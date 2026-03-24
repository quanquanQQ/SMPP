"""
Cross-Modal Fusion Module  —  论文 Section 3.3，公式 6

    Input = Concat(P_I(x), P_I(V), P_I(T))
    [x̃, Ṽ, T̃] = SelfAttn(Input)

三路特征经统一投影后拼接成 [B, 3, hidden_dim] 序列，
经 Transformer 自注意力交互后按位置切片输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalFusion(nn.Module):

    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            image_dim:  CLIP 图像特征维度
            text_dim:   原型特征维度（与图像相同）
            hidden_dim: 统一映射维度
            num_heads:  多头注意力头数
            num_layers: Transformer 层数
            dropout:    Dropout 比例
            activation: 激活函数
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 投影层：图像特征 → hidden_dim
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # 原型投影层（视觉原型和文本原型共用同一投影）
        self.prototype_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,   # 输入格式 [B, seq, feature]
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 可学习位置编码（区分三个位置：image / visual_proto / textual_proto）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 3, hidden_dim) * 0.02
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        image_features: torch.Tensor,      # [B, image_dim]
        visual_prototypes: torch.Tensor,   # [B, text_dim]
        textual_prototypes: torch.Tensor,  # [B, text_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features:     [B, image_dim]
            visual_prototypes:  [B, text_dim]   当前 batch 的视觉原型
            textual_prototypes: [B, text_dim]   当前 batch 的文本原型

        Returns:
            x_tilde: 增强图像特征  [B, hidden_dim]
            V_tilde: 增强视觉原型  [B, hidden_dim]
            T_tilde: 增强文本原型  [B, hidden_dim]
        """
        # dtype 对齐
        proj_dtype = self.image_projection[0].weight.dtype
        image_features     = image_features.to(dtype=proj_dtype)
        visual_prototypes  = visual_prototypes.to(dtype=proj_dtype)
        textual_prototypes = textual_prototypes.to(dtype=proj_dtype)

        # 投影到统一维度
        x_proj = self.image_projection(image_features)         # [B, hidden_dim]
        V_proj = self.prototype_projection(visual_prototypes)  # [B, hidden_dim]
        T_proj = self.prototype_projection(textual_prototypes) # [B, hidden_dim]

        # 拼接为序列 [B, 3, hidden_dim]
        seq = torch.stack([x_proj, V_proj, T_proj], dim=1)

        # 位置编码
        seq = seq + self.pos_embedding.to(dtype=seq.dtype)

        # Transformer 自注意力
        fused = self.transformer(seq)        # [B, 3, hidden_dim]
        fused = self.output_projection(fused)

        x_tilde = fused[:, 0, :]             # [B, hidden_dim]
        V_tilde = fused[:, 1, :]
        T_tilde = fused[:, 2, :]

        return x_tilde, V_tilde, T_tilde