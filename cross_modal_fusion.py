"""
Cross-Modal Fusion Module
实现跨模态特征投影和 Transformer 交互

Input = Concat(P_I(x), P_I(V), P_I(T))
[x̃, Ṽ, T̃] = SelfAttn(Input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalFusion(nn.Module):
    """
    跨模态特征融合模块
    使用投影层和 Transformer 进行模态间的交互
    """
    
    def __init__(
        self,
        image_dim: int = 512,      # CLIP image feature 维度
        text_dim: int = 512,       # CLIP text feature 维度
        hidden_dim: int = 512,     # 统一隐藏层维度
        num_heads: int = 8,        # 多头注意力头数
        num_layers: int = 2,       # Transformer 层数
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            image_dim: 图像特征维度
            text_dim: 文本特征维度
            hidden_dim: 统一映射后的维度
            num_heads: 多头注意力头数
            num_layers: Transformer 层数
            dropout: Dropout 比例
            activation: 激活函数类型
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 投影层: 将不同模态映射到统一维度
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 视觉原型和文本原型使用相同的投影
        self.prototype_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # [batch, seq, feature]
            norm_first=False
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 可学习的位置编码 (可选)
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            # 3个位置: [image, visual_prototype, textual_prototype]
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 3, hidden_dim) * 0.02
            )
        
        # 输出投影 (可选，用于进一步映射)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        visual_prototypes: torch.Tensor,
        textual_prototypes: torch.Tensor,
        return_separate: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            image_features: [batch_size, image_dim] 图像特征
            visual_prototypes: [batch_size, num_classes] 或 [batch_size, proto_dim]
                              视觉原型特征
            textual_prototypes: [batch_size, num_classes] 或 [batch_size, proto_dim]
                               文本原型特征
            return_separate: 是否分别返回三个特征，否则返回拼接后的特征
        
        Returns:
            x_tilde: 增强后的图像特征 [batch_size, hidden_dim]
            V_tilde: 增强后的视觉原型特征 [batch_size, hidden_dim]
            T_tilde: 增强后的文本原型特征 [batch_size, hidden_dim]
        """
        batch_size = image_features.shape[0]
        
        # 投影到统一维度
        x_proj = self.image_projection(image_features)  # [B, hidden_dim]
        V_proj = self.prototype_projection(visual_prototypes)  # [B, hidden_dim]
        T_proj = self.prototype_projection(textual_prototypes)  # [B, hidden_dim]
        
        # 拼接成序列: [batch_size, 3, hidden_dim]
        multimodal_input = torch.stack([x_proj, V_proj, T_proj], dim=1)
        
        # 添加位置编码
        if self.use_pos_encoding:
            multimodal_input = multimodal_input + self.pos_embedding
        
        # Transformer 自注意力交互
        # [batch_size, 3, hidden_dim]
        fused_features = self.transformer(multimodal_input)
        
        # 输出投影
        fused_features = self.output_projection(fused_features)
        
        if return_separate:
            # 分离三个特征
            x_tilde = fused_features[:, 0, :]  # [batch_size, hidden_dim]
            V_tilde = fused_features[:, 1, :]  # [batch_size, hidden_dim]
            T_tilde = fused_features[:, 2, :]  # [batch_size, hidden_dim]
            return x_tilde, V_tilde, T_tilde
        else:
            # 返回拼接后的特征
            return fused_features.reshape(batch_size, -1)  # [batch_size, 3*hidden_dim]


class AdaptiveCrossModalFusion(nn.Module):
    """
    自适应跨模态融合模块
    根据不同样本动态调整融合权重
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 基础融合模块
        self.base_fusion = CrossModalFusion(
            image_dim, text_dim, hidden_dim, num_heads, num_layers=2, dropout=dropout
        )
        
        # 自适应权重网络
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        visual_prototypes: torch.Tensor,
        textual_prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        自适应融合
        
        Returns:
            融合后的特征，带有自适应权重
        """
        # 基础融合
        x_tilde, V_tilde, T_tilde = self.base_fusion(
            image_features, visual_prototypes, textual_prototypes
        )
        
        # 计算自适应权重
        concat_features = torch.cat([x_tilde, V_tilde, T_tilde], dim=-1)
        weights = self.weight_network(concat_features)  # [batch_size, 3]
        
        # 加权融合
        x_tilde = x_tilde * weights[:, 0:1]
        V_tilde = V_tilde * weights[:, 1:2]
        T_tilde = T_tilde * weights[:, 2:3]
        
        return x_tilde, V_tilde, T_tilde


class PrototypeSelector(nn.Module):
    """
    原型选择模块
    根据输入图像选择最相关的原型
    """
    
    def __init__(
        self,
        num_classes: int = 77,
        feature_dim: int = 512,
        top_k: int = 5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(
        self,
        image_features: torch.Tensor,
        all_prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择 top-k 相关原型
        
        Args:
            image_features: [batch_size, feature_dim]
            all_prototypes: [num_classes, feature_dim]
        
        Returns:
            selected_prototypes: [batch_size, top_k, feature_dim]
            selected_indices: [batch_size, top_k]
        """
        # 计算相似度
        similarity = torch.matmul(
            image_features, all_prototypes.T
        ) / self.temperature  # [batch_size, num_classes]
        
        # 选择 top-k
        top_k_values, top_k_indices = torch.topk(
            similarity, self.top_k, dim=-1
        )  # [batch_size, top_k]
        
        # 获取对应的原型
        batch_size = image_features.shape[0]
        selected_prototypes = []
        for i in range(batch_size):
            selected = all_prototypes[top_k_indices[i]]  # [top_k, feature_dim]
            selected_prototypes.append(selected)
        
        selected_prototypes = torch.stack(selected_prototypes)  # [B, top_k, feature_dim]
        
        return selected_prototypes, top_k_indices


class MultiModalInteraction(nn.Module):
    """
    多模态交互模块（扩展版本）
    支持更复杂的跨模态交互模式
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        interaction_type: str = "cross_attention"
    ):
        super().__init__()
        self.interaction_type = interaction_type
        
        if interaction_type == "cross_attention":
            # 图像作为 query，原型作为 key/value
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True
            )
        
        elif interaction_type == "self_attention":
            # 全局自注意力
            self.self_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True
            )
    
    def forward(
        self,
        image_features: torch.Tensor,
        prototype_features: torch.Tensor
    ) -> torch.Tensor:
        """
        模态交互
        
        Args:
            image_features: [batch_size, feature_dim]
            prototype_features: [batch_size, num_prototypes, feature_dim]
        
        Returns:
            enhanced_features: [batch_size, feature_dim]
        """
        if self.interaction_type == "cross_attention":
            # Image as query, prototypes as key/value
            query = image_features.unsqueeze(1)  # [B, 1, feature_dim]
            key = value = prototype_features  # [B, num_proto, feature_dim]
            
            enhanced, _ = self.cross_attn(query, key, value)
            enhanced = enhanced.squeeze(1)  # [B, feature_dim]
            
        elif self.interaction_type == "self_attention":
            # Concat and self-attention
            concat = torch.cat([
                image_features.unsqueeze(1),
                prototype_features
            ], dim=1)  # [B, 1+num_proto, feature_dim]
            
            enhanced, _ = self.self_attn(concat, concat, concat)
            enhanced = enhanced[:, 0, :]  # Take the first (image) token
        
        return enhanced
