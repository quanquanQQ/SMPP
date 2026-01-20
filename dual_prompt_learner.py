"""
Dual-Grained Prompt Learning Module
实现可学习的 Global 和 Local Prompts

Global Prompt: w_i^G = [θ_{1:k}^G, e_i]
Local Prompt:  w_i^L = [θ_{1:s}^L, e_i]
"""

import torch
import torch.nn as nn
from typing import Tuple
import clip


class DualGrainedPromptLearner(nn.Module):
    """
    双粒度 Prompt 学习模块
    包含可学习的 global_ctx 和 local_ctx 参数
    """
    
    def __init__(
        self,
        num_classes: int = 77,
        global_ctx_length: int = 16,  # k: global prompt 长度
        local_ctx_length: int = 8,    # s: local prompt 长度
        ctx_dim: int = 512,            # CLIP 的 token embedding 维度
        clip_model=None,
        class_names: list = None,
        device: str = "cuda"
    ):
        """
        Args:
            num_classes: 类别数量 (77个细粒度子类)
            global_ctx_length: Global Prompt 的可学习向量长度
            local_ctx_length: Local Prompt 的可学习向量长度
            ctx_dim: Context 向量的维度 (CLIP token embedding dim)
            clip_model: CLIP 模型 (用于获取 token embedding)
            class_names: 类别名称列表
            device: 计算设备
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.global_ctx_length = global_ctx_length
        self.local_ctx_length = local_ctx_length
        self.ctx_dim = ctx_dim
        self.device = device
        
        # 初始化可学习的 context vectors
        # Global context: θ^G
        self.global_ctx = nn.Parameter(
            torch.randn(global_ctx_length, ctx_dim, device=device)
        )
        # Local context: θ^L
        self.local_ctx = nn.Parameter(
            torch.randn(local_ctx_length, ctx_dim, device=device)
        )
        
        # 初始化类别名称的 embeddings
        if clip_model is not None and class_names is not None:
            self.class_embeddings = self._initialize_class_embeddings(
                clip_model, class_names
            )
        else:
            # 如果没有提供类别名称，使用随机初始化
            self.class_embeddings = nn.Parameter(
                torch.randn(num_classes, ctx_dim, device=device)
            )
        
        # 用于编码 prompt 的文本编码器
        if clip_model is not None:
            from custom_clip_encoder import PromptTextEncoder
            self.prompt_encoder = PromptTextEncoder(clip_model)
        else:
            self.prompt_encoder = None
        
        self._initialize_parameters()
    
    def _initialize_class_embeddings(self, clip_model, class_names):
        """使用 CLIP 初始化类别名称的 embeddings"""
        with torch.no_grad():
            # 将类别名称转换为 token embeddings
            # 例如: "sports", "politics", "entertainment" 等
            token_embedding = clip_model.token_embedding
            
            class_embeds = []
            for name in class_names:
                # Tokenize class name
                text_tokens = clip.tokenize([name], truncate=True).to(self.device)
                # 获取第一个实际 token 的 embedding (跳过 SOS token)
                embeds = token_embedding(text_tokens)  # [1, seq_len, ctx_dim]
                class_embed = embeds[0, 1, :]  # 取第一个有意义的 token
                class_embeds.append(class_embed)
            
            class_embeddings = torch.stack(class_embeds)  # [num_classes, ctx_dim]
        
        return nn.Parameter(class_embeddings)
    
    def _initialize_parameters(self):
        """参数初始化"""
        # 使用正态分布初始化 context vectors
        nn.init.normal_(self.global_ctx, std=0.02)
        nn.init.normal_(self.local_ctx, std=0.02)
    
    def construct_prompts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建 Global 和 Local Prompts
        
        Returns:
            global_prompts: [num_classes, global_ctx_length + 1, ctx_dim]
            local_prompts: [num_classes, local_ctx_length + 1, ctx_dim]
        """
        # Global Prompt: w_i^G = [θ_{1:k}^G, e_i]
        global_ctx = self.global_ctx.unsqueeze(0).expand(
            self.num_classes, -1, -1
        )  # [num_classes, k, ctx_dim]
        
        class_embeds = self.class_embeddings.unsqueeze(1)  # [num_classes, 1, ctx_dim]
        global_prompts = torch.cat([global_ctx, class_embeds], dim=1)
        # [num_classes, k+1, ctx_dim]
        
        # Local Prompt: w_i^L = [θ_{1:s}^L, e_i]
        local_ctx = self.local_ctx.unsqueeze(0).expand(
            self.num_classes, -1, -1
        )  # [num_classes, s, ctx_dim]
        
        local_prompts = torch.cat([local_ctx, class_embeds], dim=1)
        # [num_classes, s+1, ctx_dim]
        
        return global_prompts, local_prompts
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，编码 Prompts 得到类别特征
        
        Returns:
            G: Global 类别特征 [num_classes, feature_dim]
            L: Local 类别特征 [num_classes, feature_dim]
        """
        # 构建 prompts
        global_prompts, local_prompts = self.construct_prompts()
        
        if self.prompt_encoder is not None:
            # 使用 CLIP Text Encoder 编码
            G = self.prompt_encoder(global_prompts, return_sequence=False)
            L = self.prompt_encoder(local_prompts, return_sequence=False)
        else:
            # 如果没有编码器，直接使用平均池化
            G = global_prompts.mean(dim=1)  # [num_classes, ctx_dim]
            L = local_prompts.mean(dim=1)  # [num_classes, ctx_dim]
        
        # L2 归一化
        G = G / G.norm(dim=-1, keepdim=True)
        L = L / L.norm(dim=-1, keepdim=True)
        
        return G, L
    
    def get_prompt_embeddings(self) -> dict:
        """
        获取 Prompt embeddings (用于特征提取阶段)
        
        Returns:
            dict: 包含 global 和 local prompt embeddings
        """
        global_prompts, local_prompts = self.construct_prompts()
        
        return {
            'global_prompts': global_prompts,
            'local_prompts': local_prompts,
            'global_ctx': self.global_ctx,
            'local_ctx': self.local_ctx,
            'class_embeddings': self.class_embeddings
        }


class PromptPooling(nn.Module):
    """
    用于对 prompt 特征进行池化/聚合
    可选的增强模块
    """
    
    def __init__(self, feature_dim: int, pooling_type: str = "attention"):
        super().__init__()
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            # 使用注意力机制进行加权池化
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feature_dim]
        
        Returns:
            pooled: [batch_size, feature_dim]
        """
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        
        elif self.pooling_type == "max":
            return x.max(dim=1)[0]
        
        elif self.pooling_type == "attention":
            # 计算注意力权重
            weights = self.attention(x)  # [batch_size, seq_len, 1]
            weights = torch.softmax(weights, dim=1)
            # 加权求和
            pooled = (x * weights).sum(dim=1)  # [batch_size, feature_dim]
            return pooled
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


def create_prompt_learner(
    clip_model,
    num_classes: int = 77,
    class_names: list = None,
    global_ctx_length: int = 16,
    local_ctx_length: int = 8,
    device: str = "cuda"
):
    """
    创建 Prompt Learner 的工厂函数
    
    Args:
        clip_model: CLIP 模型实例
        num_classes: 类别数量
        class_names: 类别名称列表
        global_ctx_length: Global context 长度
        local_ctx_length: Local context 长度
        device: 计算设备
    
    Returns:
        prompt_learner: DualGrainedPromptLearner 实例
    """
    # 如果没有提供类别名称，生成默认名称
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    # 获取 CLIP 的 token embedding 维度
    ctx_dim = clip_model.token_embedding.weight.shape[1]
    
    prompt_learner = DualGrainedPromptLearner(
        num_classes=num_classes,
        global_ctx_length=global_ctx_length,
        local_ctx_length=local_ctx_length,
        ctx_dim=ctx_dim,
        clip_model=clip_model,
        class_names=class_names,
        device=device
    )
    
    return prompt_learner
