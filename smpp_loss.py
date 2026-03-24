"""
SMPP Loss Functions  —  论文 Section 3.4，公式 8

总损失：L = L_g + L_o + L_c

L_g：全局文本对齐损失
    p_i = cos<h, G_i>  →  CrossEntropy

L_o：局部文本对齐损失（空间聚合）
    P_{ij} = cos<H_j, L_i>
    p_i' = sum_j [ softmax(P_{ij}/τ_s)_j * P_{ij} ]  →  CrossEntropy

L_c：视觉-原型对齐损失
    p_V(y_i=1) = softmax(cos<x̃, Ṽ_i> / τ_v)
    p_T(y_i=1) = softmax(cos<x̃, T̃_i> / τ_t)
    logits = (logits_V + logits_T) / 2  →  CrossEntropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalTextAlignmentLoss(nn.Module):
    """L_g：全局文本对齐损失（论文公式 4）"""

    # ── 修改一A：增加 class_weights 参数，传入 CrossEntropyLoss ──────
    def __init__(self, temperature: float = 0.07,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        text_global: torch.Tensor,          # h:  [B, d]
        global_prompt_features: torch.Tensor,  # G:  [K, d]
        labels: torch.Tensor,               # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.normalize(text_global,          dim=-1)
        G = F.normalize(global_prompt_features, dim=-1)

        logits = torch.matmul(h, G.T) / self.temperature   # [B, K]
        loss   = self.ce(logits, labels)
        return loss, logits


class LocalTextAlignmentLoss(nn.Module):
    """
    L_o：局部文本对齐损失（论文公式 5）

    空间聚合：对每个类别 i，在 seq_len 维做 softmax 加权求和。
        P_{ij} = cos<H_j, L_i>
        attn   = softmax(P / τ_s, dim=seq_len)
        p_i'   = sum_j(attn_{ij} * P_{ij})
    """

    # ── 修改一B：增加 class_weights 参数 ────────────────────────────
    def __init__(
        self,
        temperature: float = 0.07,
        spatial_temperature: float = 0.1,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.temperature         = temperature
        self.spatial_temperature = spatial_temperature
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        text_sequence: torch.Tensor,           # H:  [B, seq_len, d]
        local_prompt_features: torch.Tensor,   # L:  [K, d]
        labels: torch.Tensor,                  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = F.normalize(text_sequence,          dim=-1)   # [B, L, d]
        L = F.normalize(local_prompt_features,  dim=-1)   # [K, d]

        # Token 级相似度 P_{ij} = cos<H_j, L_i>
        # [B, L, K]
        token_sim = torch.matmul(H, L.T)

        # 在 seq_len 维（dim=1）对每个类别独立 softmax
        attn = F.softmax(token_sim / self.spatial_temperature, dim=1)  # [B, L, K]

        # 加权聚合  →  [B, K]
        aggregated = (attn * token_sim).sum(dim=1)

        logits = aggregated / self.temperature
        loss   = self.ce(logits, labels)
        return loss, logits


class VisualPrototypeAlignmentLoss(nn.Module):
    """
    L_c：视觉-原型对齐损失（论文公式 7）

    p_V: softmax(cos<x̃, Ṽ_i> / τ_v)
    p_T: softmax(cos<x̃, T̃_i> / τ_t)
    logits = (logits_V + logits_T) / 2  →  CrossEntropy
    """

    # ── 修改一C：增加 class_weights 参数 ────────────────────────────
    def __init__(
        self,
        temperature_visual: float = 0.07,
        temperature_textual: float = 0.07,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.tau_v = temperature_visual
        self.tau_t = temperature_textual
        self.ce    = nn.CrossEntropyLoss(weight=class_weights)
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        enhanced_image: torch.Tensor,              # x̃:  [B, d]
        enhanced_visual_prototypes: torch.Tensor,  # Ṽ:  [B, K, d]  或  [K, d]
        enhanced_textual_prototypes: torch.Tensor, # T̃:  [B, K, d]  或  [K, d]
        labels: torch.Tensor,                      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.normalize(enhanced_image, dim=-1)    # [B, d]

        # 支持 [K, d] 和 [B, K, d] 两种输入格式
        if enhanced_visual_prototypes.dim() == 2:
            V = F.normalize(enhanced_visual_prototypes, dim=-1)   # [K, d]
            T = F.normalize(enhanced_textual_prototypes, dim=-1)

            logits_V = torch.matmul(x, V.T) / self.tau_v         # [B, K]
            logits_T = torch.matmul(x, T.T) / self.tau_t
        else:
            V = F.normalize(enhanced_visual_prototypes, dim=-1)   # [B, K, d]
            T = F.normalize(enhanced_textual_prototypes, dim=-1)

            logits_V = torch.bmm(
                x.unsqueeze(1), V.transpose(1, 2)
            ).squeeze(1) / self.tau_v                             # [B, K]
            logits_T = torch.bmm(
                x.unsqueeze(1), T.transpose(1, 2)
            ).squeeze(1) / self.tau_t

        logits = (logits_V + logits_T) / 2
        loss   = self.ce(logits, labels)
        return loss, logits


class SMPPLoss(nn.Module):
    """
    SMPP 总损失：L = L_g + L_o + L_c  （各项等权，论文未指定不同权重）
    """

    # ── 修改一D：SMPPLoss 接收 class_weights 并传给三个子损失 ────────
    def __init__(
        self,
        temperature: float = 0.07,
        spatial_temperature: float = 0.1,
        temperature_visual: float = 0.07,
        temperature_textual: float = 0.07,
        loss_weights: dict = None,
        class_weights: torch.Tensor = None,   # ← 新增
    ):
        super().__init__()
        self.global_loss = GlobalTextAlignmentLoss(
            temperature, class_weights=class_weights)
        self.local_loss  = LocalTextAlignmentLoss(
            temperature, spatial_temperature, class_weights=class_weights)
        self.visual_loss = VisualPrototypeAlignmentLoss(
            temperature_visual, temperature_textual, class_weights=class_weights)

        self.loss_weights = loss_weights or {
            "global": 1.0,
            "local":  1.0,
            "visual": 1.0,
        }
    # ────────────────────────────────────────────────────────────────

    def forward(
        self,
        text_global_features: torch.Tensor,         # h  [B, d]
        text_sequence_features: torch.Tensor,       # H  [B, L, d]
        global_prompt_features: torch.Tensor,       # G  [K, d]
        local_prompt_features: torch.Tensor,        # L  [K, d]
        enhanced_image_features: torch.Tensor,      # x̃ [B, d]
        enhanced_visual_prototypes: torch.Tensor,   # Ṽ [B, K, d] or [K, d]
        enhanced_textual_prototypes: torch.Tensor,  # T̃ [B, K, d] or [K, d]
        labels: torch.Tensor,                       # [B]
    ) -> dict:
        loss_g, logits_g = self.global_loss(
            text_global_features, global_prompt_features, labels
        )
        loss_o, logits_o = self.local_loss(
            text_sequence_features, local_prompt_features, labels
        )
        loss_c, logits_c = self.visual_loss(
            enhanced_image_features,
            enhanced_visual_prototypes,
            enhanced_textual_prototypes,
            labels,
        )

        total = (
            self.loss_weights["global"] * loss_g
            + self.loss_weights["local"]  * loss_o
            + self.loss_weights["visual"] * loss_c
        )

        return {
            "loss":          total,
            "loss_global":   loss_g,
            "loss_local":    loss_o,
            "loss_visual":   loss_c,
            "logits_global": logits_g,
            "logits_local":  logits_o,
            "logits_visual": logits_c,
        }