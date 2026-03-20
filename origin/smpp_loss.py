"""
SMPP Loss Functions
实现三个损失函数:
1. L_g: 全局文本对齐损失 (Global Text Alignment Loss)
2. L_o: 局部文本对齐损失 (Local Text Alignment Loss with Spatial Aggregation)
3. L_c: 视觉与原型对齐损失 (Visual-Prototype Alignment Loss)

总损失: L = L_g + L_o + L_c
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalTextAlignmentLoss(nn.Module):
    """
    全局文本对齐损失 L_g
    
    计算输入文本全局特征 h 与全局 Prompt 特征 G_i 的余弦相似度
    p_i = cos<h, G_i>
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        text_global_features: torch.Tensor,  # h: [batch_size, feature_dim]
        global_prompt_features: torch.Tensor,  # G: [num_classes, feature_dim]
        labels: torch.Tensor  # [batch_size]
    ) -> torch.Tensor:
        """
        Args:
            text_global_features: 输入文本的全局特征 (EOS token)
            global_prompt_features: 所有类别的 Global Prompt 特征
            labels: 真实类别标签
        
        Returns:
            loss: Cross-Entropy Loss
        """
        # L2 归一化
        text_features = F.normalize(text_global_features, dim=-1)
        prompt_features = F.normalize(global_prompt_features, dim=-1)
        
        # 计算余弦相似度: [batch_size, num_classes]
        logits = torch.matmul(text_features, prompt_features.T) / self.temperature
        
        # Cross-Entropy Loss
        loss = self.ce_loss(logits, labels)
        
        return loss, logits


class LocalTextAlignmentLoss(nn.Module):
    """
    局部文本对齐损失 L_o
    
    对文本序列特征 H 和局部 Prompt 特征 L_i 进行空间聚合:
    1. 计算 Token 级相似度: P_ij = cos<H_j, L_i>
    2. 加权聚合: p_i' = Σ_j (exp(P_ij/τ_s) / Σ_j exp(P_ij/τ_s)) * P_ij
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        spatial_temperature: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.spatial_temperature = spatial_temperature  # τ_s
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        text_sequence_features: torch.Tensor,  # H: [batch_size, seq_len, feature_dim]
        local_prompt_features: torch.Tensor,  # L: [num_classes, feature_dim]
        labels: torch.Tensor  # [batch_size]
    ) -> torch.Tensor:
        """
        Args:
            text_sequence_features: 输入文本的序列特征 (所有 tokens)
            local_prompt_features: 所有类别的 Local Prompt 特征
            labels: 真实类别标签
        
        Returns:
            loss: Cross-Entropy Loss after spatial aggregation
        """
        batch_size, seq_len, feature_dim = text_sequence_features.shape
        num_classes = local_prompt_features.shape[0]
        
        # L2 归一化
        text_features = F.normalize(text_sequence_features, dim=-1)
        # [batch_size, seq_len, feature_dim]
        
        prompt_features = F.normalize(local_prompt_features, dim=-1)
        # [num_classes, feature_dim]
        
        # 计算 Token 级相似度: P_ij = cos<H_j, L_i>
        # [batch_size, seq_len, num_classes]
        token_similarities = torch.matmul(
            text_features, prompt_features.T
        )
        
        # 空间聚合 (Spatial Aggregation)
        # 计算每个 token 对每个类别的注意力权重
        attention_weights = F.softmax(
            token_similarities / self.spatial_temperature, dim=1
        )  # [batch_size, seq_len, num_classes]
        
        # 加权求和: p_i' = Σ_j w_ij * P_ij
        aggregated_logits = (attention_weights * token_similarities).sum(dim=1)
        # [batch_size, num_classes]
        
        # 应用 temperature scaling
        logits = aggregated_logits / self.temperature
        
        # Cross-Entropy Loss
        loss = self.ce_loss(logits, labels)
        
        return loss, logits


class VisualPrototypeAlignmentLoss(nn.Module):
    """
    视觉与原型对齐损失 L_c
    
    基于交互后的特征 x̃ 进行分类预测:
    
    p_V(y_i=1) = exp(cos<x̃, Ṽ_i> / τ_v) / Σ_j exp(cos<x̃, Ṽ_j> / τ_v)
    p_T(y_i=1) = exp(cos<x̃, T̃_i> / τ_t) / Σ_j exp(cos<x̃, T̃_i> / τ_t)
    
    最终 L_c: 对 p_V 和 p_T 的 logits 取平均后计算 Cross-Entropy
    """
    
    def __init__(
        self,
        temperature_visual: float = 0.07,
        temperature_textual: float = 0.07,
        fusion_method: str = "mean"  # "mean", "weighted", "max"
    ):
        super().__init__()
        self.temperature_visual = temperature_visual  # τ_v
        self.temperature_textual = temperature_textual  # τ_t
        self.fusion_method = fusion_method
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 如果使用加权融合，学习权重
        if fusion_method == "weighted":
            self.fusion_weight = nn.Parameter(torch.tensor([0.5]))
    
    def forward(
        self,
        enhanced_image_features: torch.Tensor,  # x̃: [batch_size, feature_dim]
        enhanced_visual_prototypes: torch.Tensor,  # Ṽ: [batch_size, num_classes, feature_dim]
        enhanced_textual_prototypes: torch.Tensor,  # T̃: [batch_size, num_classes, feature_dim]
        labels: torch.Tensor  # [batch_size]
    ) -> torch.Tensor:
        """
        Args:
            enhanced_image_features: 交互后的增强图像特征
            enhanced_visual_prototypes: 交互后的增强视觉原型
            enhanced_textual_prototypes: 交互后的增强文本原型
            labels: 真实类别标签
        
        Returns:
            loss: Cross-Entropy Loss
        """
        # L2 归一化
        image_features = F.normalize(enhanced_image_features, dim=-1)
        # [batch_size, feature_dim]
        
        # 如果原型是 [num_classes, feature_dim]，需要扩展
        if enhanced_visual_prototypes.dim() == 2:
            visual_prototypes = enhanced_visual_prototypes.unsqueeze(0).expand(
                image_features.shape[0], -1, -1
            )
            textual_prototypes = enhanced_textual_prototypes.unsqueeze(0).expand(
                image_features.shape[0], -1, -1
            )
        else:
            visual_prototypes = enhanced_visual_prototypes
            textual_prototypes = enhanced_textual_prototypes
        
        visual_prototypes = F.normalize(visual_prototypes, dim=-1)
        # [batch_size, num_classes, feature_dim]
        
        textual_prototypes = F.normalize(textual_prototypes, dim=-1)
        # [batch_size, num_classes, feature_dim]
        
        # 计算视觉原型预测概率
        # cos<x̃, Ṽ_i>
        visual_similarities = torch.bmm(
            image_features.unsqueeze(1),  # [batch_size, 1, feature_dim]
            visual_prototypes.transpose(1, 2)  # [batch_size, feature_dim, num_classes]
        ).squeeze(1)  # [batch_size, num_classes]
        
        logits_visual = visual_similarities / self.temperature_visual
        
        # 计算文本原型预测概率
        # cos<x̃, T̃_i>
        textual_similarities = torch.bmm(
            image_features.unsqueeze(1),
            textual_prototypes.transpose(1, 2)
        ).squeeze(1)  # [batch_size, num_classes]
        
        logits_textual = textual_similarities / self.temperature_textual
        
        # 融合 logits
        if self.fusion_method == "mean":
            logits = (logits_visual + logits_textual) / 2
        elif self.fusion_method == "weighted":
            alpha = torch.sigmoid(self.fusion_weight)
            logits = alpha * logits_visual + (1 - alpha) * logits_textual
        elif self.fusion_method == "max":
            logits = torch.max(logits_visual, logits_textual)
        
        # Cross-Entropy Loss
        loss = self.ce_loss(logits, labels)
        
        return loss, logits


class SMPPLoss(nn.Module):
    """
    Social Media Popularity Prediction 总损失
    
    L = L_g + L_o + L_c
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        spatial_temperature: float = 0.1,
        temperature_visual: float = 0.07,
        temperature_textual: float = 0.07,
        loss_weights: dict = None
    ):
        """
        Args:
            temperature: 全局和局部文本对齐的 temperature
            spatial_temperature: 局部对齐的空间聚合 temperature
            temperature_visual: 视觉原型对齐的 temperature
            temperature_textual: 文本原型对齐的 temperature
            loss_weights: 各损失的权重 {"global": 1.0, "local": 1.0, "visual": 1.0}
        """
        super().__init__()
        
        # 三个损失函数
        self.global_loss = GlobalTextAlignmentLoss(temperature)
        self.local_loss = LocalTextAlignmentLoss(temperature, spatial_temperature)
        self.visual_loss = VisualPrototypeAlignmentLoss(
            temperature_visual, temperature_textual
        )
        
        # 损失权重
        if loss_weights is None:
            loss_weights = {"global": 1.0, "local": 1.0, "visual": 1.0}
        self.loss_weights = loss_weights
    
    def forward(
        self,
        text_global_features: torch.Tensor,  # h
        text_sequence_features: torch.Tensor,  # H
        global_prompt_features: torch.Tensor,  # G
        local_prompt_features: torch.Tensor,  # L
        enhanced_image_features: torch.Tensor,  # x̃
        enhanced_visual_prototypes: torch.Tensor,  # Ṽ
        enhanced_textual_prototypes: torch.Tensor,  # T̃
        labels: torch.Tensor
    ) -> dict:
        """
        计算总损失
        
        Returns:
            dict: {
                "loss": 总损失,
                "loss_global": L_g,
                "loss_local": L_o,
                "loss_visual": L_c,
                "logits_global": 全局 logits,
                "logits_local": 局部 logits,
                "logits_visual": 视觉 logits
            }
        """
        # L_g: 全局文本对齐
        loss_g, logits_g = self.global_loss(
            text_global_features, global_prompt_features, labels
        )
        
        # L_o: 局部文本对齐
        loss_o, logits_o = self.local_loss(
            text_sequence_features, local_prompt_features, labels
        )
        
        # L_c: 视觉与原型对齐
        loss_c, logits_c = self.visual_loss(
            enhanced_image_features,
            enhanced_visual_prototypes,
            enhanced_textual_prototypes,
            labels
        )
        
        # 总损失
        total_loss = (
            self.loss_weights["global"] * loss_g +
            self.loss_weights["local"] * loss_o +
            self.loss_weights["visual"] * loss_c
        )
        
        return {
            "loss": total_loss,
            "loss_global": loss_g,
            "loss_local": loss_o,
            "loss_visual": loss_c,
            "logits_global": logits_g,
            "logits_local": logits_o,
            "logits_visual": logits_c
        }


class FocalLoss(nn.Module):
    """
    Focal Loss (可选)
    用于处理类别不平衡问题
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            labels: [batch_size]
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """
    对比损失 (可选)
    进一步增强特征判别力
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim]
            labels: [batch_size]
        """
        # 归一化特征
        features = F.normalize(features, dim=-1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建正样本 mask
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # 对角线设为 0 (排除自己)
        mask = mask - torch.eye(mask.shape[0], device=mask.device)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只考虑正样本对
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss
