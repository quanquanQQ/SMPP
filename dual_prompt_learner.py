"""
Dual-Grained Prompt Learning Module  —  论文 Section 3.2，公式 1-2

    w_i^G = [θ^G_{1:s}, e_i]       Global Prompt（s=16）
    w_i^L = [θ^L_{1:s}, e_i]       Local  Prompt（s=8）
    {G_i, L_i} = f_T(w_i^G, w_i^L)

θ^G 和 θ^L 是两套独立的可学习参数（不共享权重）。
e_i 用 CLIP token_embedding 初始化，训练时固定（register_buffer）。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import clip

from custom_clip_encoder import PromptTextEncoder


class DualGrainedPromptLearner(nn.Module):

    def __init__(
        self,
        num_classes: int = 77,
        global_ctx_length: int = 16,
        local_ctx_length: int = 8,
        ctx_dim: int = 512,
        dropout: float = 0.1,
        clip_model=None,
        class_names: Optional[list] = None,
        device: str = "cuda",
    ):
        """
        Args:
            num_classes:        类别数量
            global_ctx_length:  Global context 向量长度（论文 s=16）
            local_ctx_length:   Local  context 向量长度（论文 s=8）
            ctx_dim:            Context 向量维度（CLIP ViT-B/32 = 512）
            dropout:            Context dropout
            clip_model:         CLIP 模型，用于初始化 class_embeddings 和 PromptTextEncoder
            class_names:        类别名称列表
            device:             计算设备
        """
        super().__init__()

        self.num_classes       = num_classes
        self.global_ctx_length = global_ctx_length
        self.local_ctx_length  = local_ctx_length
        self.ctx_dim           = ctx_dim
        self.device            = device

        # ---- 可学习 context（两套独立参数，N(0, 0.02) 初始化）----
        self.global_ctx = nn.Parameter(
            torch.empty(global_ctx_length, ctx_dim, device=device)
        )
        self.local_ctx = nn.Parameter(
            torch.empty(local_ctx_length, ctx_dim, device=device)
        )
        nn.init.normal_(self.global_ctx, std=0.02)
        nn.init.normal_(self.local_ctx,  std=0.02)

        self.ctx_dropout = nn.Dropout(dropout)

        # ---- 类别标签 Embedding e_i（固定，不参与梯度）----
        if clip_model is not None and class_names is not None:
            class_emb = self._init_class_embeddings(clip_model, class_names, device)
            self.register_buffer("class_embeddings", class_emb)
        else:
            # fallback：随机可训练 embedding
            self.class_embeddings = nn.Parameter(
                torch.randn(num_classes, ctx_dim, device=device)
            )

        # ---- Prompt 文本编码器 ----
        if clip_model is not None:
            self.prompt_encoder = PromptTextEncoder(clip_model)
        else:
            self.prompt_encoder = None

    @staticmethod
    def _init_class_embeddings(clip_model, class_names: list, device: str) -> torch.Tensor:
        """
        从 CLIP token_embedding 取类名第一个有效 token（跳过 SOS，取 index=1）。
        """
        token_embedding = clip_model.token_embedding
        embeds = []
        with torch.no_grad():
            for name in class_names:
                tokens = clip.tokenize([name], truncate=True).to(device)
                emb    = token_embedding(tokens)      # [1, seq_len, ctx_dim]
                embeds.append(emb[0, 1, :])           # 第一个有效 token
        return torch.stack(embeds)                    # [K, ctx_dim]

    def construct_prompts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建 Prompt embedding 序列。

        Returns:
            global_prompts: [K, global_ctx_length+1, ctx_dim]
            local_prompts:  [K, local_ctx_length+1,  ctx_dim]
        """
        class_emb = self.class_embeddings.unsqueeze(1)    # [K, 1, ctx_dim]

        # Global: w_i^G = [θ^G, e_i]
        g = self.ctx_dropout(self.global_ctx)              # [s_g, d]
        g = g.unsqueeze(0).expand(self.num_classes, -1, -1)  # [K, s_g, d]
        global_prompts = torch.cat([g, class_emb], dim=1)   # [K, s_g+1, d]

        # Local: w_i^L = [θ^L, e_i]
        l = self.ctx_dropout(self.local_ctx)               # [s_l, d]
        l = l.unsqueeze(0).expand(self.num_classes, -1, -1)  # [K, s_l, d]
        local_prompts  = torch.cat([l, class_emb], dim=1)   # [K, s_l+1, d]

        return global_prompts, local_prompts

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码 Prompts，得到类别特征。

        Returns:
            G: [K, feature_dim]   全局类别特征，用于 L_g
            L: [K, feature_dim]   局部类别特征，用于 L_o
        """
        global_prompts, local_prompts = self.construct_prompts()

        if self.prompt_encoder is not None:
            G = self.prompt_encoder(global_prompts, return_sequence=False)
            L = self.prompt_encoder(local_prompts,  return_sequence=False)
        else:
            G = global_prompts.mean(dim=1)
            L = local_prompts.mean(dim=1)

        G = G / G.norm(dim=-1, keepdim=True)
        L = L / L.norm(dim=-1, keepdim=True)
        return G, L

    def get_prompt_embeddings(self) -> dict:
        global_prompts, local_prompts = self.construct_prompts()
        return {
            "global_prompts":   global_prompts,
            "local_prompts":    local_prompts,
            "global_ctx":       self.global_ctx,
            "local_ctx":        self.local_ctx,
            "class_embeddings": self.class_embeddings,
        }


def create_prompt_learner(
    clip_model,
    num_classes: int = 77,
    class_names: Optional[list] = None,
    global_ctx_length: int = 16,
    local_ctx_length: int = 8,
    prompt_dropout: float = 0.1,
    device: str = "cuda",
) -> DualGrainedPromptLearner:
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    ctx_dim = clip_model.token_embedding.weight.shape[1]

    return DualGrainedPromptLearner(
        num_classes=num_classes,
        global_ctx_length=global_ctx_length,
        local_ctx_length=local_ctx_length,
        ctx_dim=ctx_dim,
        dropout=prompt_dropout,
        clip_model=clip_model,
        class_names=class_names,
        device=device,
    )