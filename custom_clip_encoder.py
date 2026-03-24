"""
Custom CLIP Text Encoder  —  论文 Section 3.2，公式 3
修改 CLIP Text Encoder，同时返回全局特征 h 和序列特征 H

    { h, H } = f_T(r)
    h ∈ R^d        — EOS token 投影后 L2 归一化
    H ∈ R^{l×d}   — 所有 token 投影后的序列特征
"""

import torch
import torch.nn as nn
from typing import Tuple
import clip
from clip.model import CLIP


class CustomCLIPTextEncoder(nn.Module):
    """
    自定义 CLIP 文本编码器，返回 (h, H)。
    """

    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.transformer          = clip_model.transformer
        self.token_embedding      = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final             = clip_model.ln_final
        self.text_projection      = clip_model.text_projection
        self.dtype                = clip_model.dtype

    def forward(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text: tokenized text  [B, seq_len]

        Returns:
            h: 全局特征（EOS token，L2 归一化）  [B, feature_dim]
            H: 序列特征（所有 token，已投影）    [B, seq_len, feature_dim]
        """
        B = text.shape[0]

        x = self.token_embedding(text).type(self.dtype)       # [B, L, d_model]
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)                                # [L, B, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                                # [B, L, d_model]

        x = self.ln_final(x).type(self.dtype)                 # [B, L, d_model]

        # H：完整序列投影
        if self.text_projection is not None:
            H = x @ self.text_projection                      # [B, L, feature_dim]
        else:
            H = x

        # h：EOS token 投影后 L2 归一化
        eot_idx = text.argmax(dim=-1)                         # [B]
        h_raw   = x[torch.arange(B), eot_idx]                # [B, d_model]
        if self.text_projection is not None:
            h = h_raw @ self.text_projection                  # [B, feature_dim]
        else:
            h = h_raw
        h = h / h.norm(dim=-1, keepdim=True)

        return h, H

    def encode_text_global(self, text: torch.Tensor) -> torch.Tensor:
        """兼容原始 CLIP 接口，只返回 h"""
        h, _ = self.forward(text)
        return h

    def encode_text_sequence(self, text: torch.Tensor) -> torch.Tensor:
        """只返回序列特征 H"""
        _, H = self.forward(text)
        return H


class PromptTextEncoder(nn.Module):
    """
    编码 Dual-Grained Prompt 的 embedding 序列（非普通文本 token）。

    输入 shape：[B, seq_len, d_model]（已拼接的 [ctx, e_i]）
    关键点：需临时替换 CLIP resblock 中固定的 77×77 attn_mask，
           改为与当前 seq_len 匹配的 causal mask。
    """

    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.transformer          = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final             = clip_model.ln_final
        self.text_projection      = clip_model.text_projection
        self.dtype                = clip_model.dtype
        self._mask_cache: dict    = {}

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        if key not in self._mask_cache:
            mask = torch.empty(seq_len, seq_len, device=device)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(
        self,
        prompt_embeddings: torch.Tensor,   # [B, seq_len, d_model]
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeddings: [B, seq_len, d_model]
            return_sequence:   True → [B, seq_len, feature_dim]
                               False → [B, feature_dim]（最后一个 token）
        """
        x       = prompt_embeddings.type(self.dtype)
        seq_len = x.shape[1]
        device  = x.device

        x = x + self.positional_embedding[:seq_len].type(self.dtype)

        attn_mask = self._causal_mask(seq_len, device)

        # 临时替换所有 resblock 的 attn_mask
        x = x.permute(1, 0, 2)   # [L, B, d_model]
        orig_masks = []
        for blk in self.transformer.resblocks:
            orig_masks.append(blk.attn_mask)
            blk.attn_mask = attn_mask
        try:
            x = self.transformer(x)
        finally:
            for blk, m in zip(self.transformer.resblocks, orig_masks):
                blk.attn_mask = m

        x = x.permute(1, 0, 2)   # [B, L, d_model]
        x = self.ln_final(x).type(self.dtype)

        if return_sequence:
            out = x @ self.text_projection if self.text_projection is not None else x
            return out                                      # [B, L, feature_dim]
        else:
            feat = x[:, -1, :]                             # 最后一个 token
            if self.text_projection is not None:
                feat = feat @ self.text_projection
            return feat                                    # [B, feature_dim]


def create_custom_text_encoder(
    model_name: str = "ViT-B/32",
    device: str = "cuda",
):
    """
    工厂函数

    Returns:
        (CustomCLIPTextEncoder, clip_model, preprocess)
    """
    clip_model, preprocess = clip.load(model_name, device=device)
    encoder = CustomCLIPTextEncoder(clip_model)
    return encoder, clip_model, preprocess