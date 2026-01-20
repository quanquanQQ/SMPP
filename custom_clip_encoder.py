"""
Custom CLIP Text Encoder
修改 CLIP 的 Text Encoder，使其能返回完整 Token 序列特征 H，而不仅仅是 EOS token
"""

import torch
import torch.nn as nn
from typing import Tuple
import clip
from clip.model import CLIP


class CustomCLIPTextEncoder(nn.Module):
    """
    自定义 CLIP 文本编码器
    返回: {h, H}
    - h: 全局文本 Embedding (EOS token), shape [batch_size, feature_dim]
    - H: 序列 Token Embeddings, shape [batch_size, seq_len, feature_dim]
    """
    
    def __init__(self, clip_model: CLIP):
        """
        Args:
            clip_model: 预训练的 CLIP 模型
        """
        super().__init__()
        self.clip_model = clip_model
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            text: tokenized text, shape [batch_size, seq_len]
        
        Returns:
            h: 全局文本特征 (EOS token), shape [batch_size, feature_dim]
            H: 序列 Token 特征, shape [batch_size, seq_len, feature_dim]
        """
        # Token embedding + positional embedding
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, seq_len, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        
        # Permute for transformer: [seq_len, batch_size, d_model]
        x = x.permute(1, 0, 2)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Permute back: [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2)
        
        # Layer normalization
        x = self.ln_final(x).type(self.dtype)
        
        # H: 完整序列特征 [batch_size, seq_len, d_model]
        H = x
        
        # h: 提取 EOS token 的特征
        # text.argmax(dim=-1) 找到每个序列的 EOS token 位置
        eot_indices = text.argmax(dim=-1)
        h = x[torch.arange(x.shape[0]), eot_indices]  # [batch_size, d_model]
        
        # 投影到最终特征空间
        if self.text_projection is not None:
            h = h @ self.text_projection  # [batch_size, feature_dim]
            # 对序列特征也应用投影
            H = H @ self.text_projection  # [batch_size, seq_len, feature_dim]
        
        return h, H
    
    def encode_text_global(self, text: torch.Tensor) -> torch.Tensor:
        """只返回全局特征 h (兼容原始 CLIP)"""
        h, _ = self.forward(text)
        return h
    
    def encode_text_sequence(self, text: torch.Tensor) -> torch.Tensor:
        """只返回序列特征 H"""
        _, H = self.forward(text)
        return H


class PromptTextEncoder(nn.Module):
    """
    用于编码 Prompt 的文本编码器
    处理可学习的 Prompt 向量 + 类别标签 Embedding
    """
    
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(
        self, 
        prompt_embeddings: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        编码 Prompt
        
        Args:
            prompt_embeddings: [batch_size, seq_len, d_model] 
                              已经拼接好的 [learnable_ctx, class_embedding]
            return_sequence: 是否返回完整序列特征
        
        Returns:
            features: [batch_size, feature_dim] 或 [batch_size, seq_len, feature_dim]
        """
        # 添加位置编码
        seq_len = prompt_embeddings.shape[1]
        x = prompt_embeddings + self.positional_embedding[:seq_len].type(self.dtype)
        
        # Transformer encoding
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        
        # Layer normalization
        x = self.ln_final(x).type(self.dtype)
        
        if return_sequence:
            # 返回完整序列特征
            if self.text_projection is not None:
                x = x @ self.text_projection
            return x
        else:
            # 返回最后一个 token 的特征 (类似 EOS)
            features = x[:, -1, :]  # [batch_size, d_model]
            if self.text_projection is not None:
                features = features @ self.text_projection
            return features


def create_custom_text_encoder(model_name: str = "ViT-B/32", device: str = "cuda"):
    """
    创建自定义文本编码器的工厂函数
    
    Args:
        model_name: CLIP 模型名称
        device: 计算设备
    
    Returns:
        custom_encoder: CustomCLIPTextEncoder 实例
        clip_model: 完整的 CLIP 模型 (用于图像编码)
    """
    clip_model, preprocess = clip.load(model_name, device=device)
    custom_encoder = CustomCLIPTextEncoder(clip_model)
    
    return custom_encoder, clip_model, preprocess
