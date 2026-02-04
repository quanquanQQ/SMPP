"""
Main SMPP Model
整合所有模块的完整社交媒体流行度预测模型
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import clip

from custom_clip_encoder import CustomCLIPTextEncoder, create_custom_text_encoder
from dual_prompt_learner import DualGrainedPromptLearner, create_prompt_learner
from cross_modal_fusion import CrossModalFusion
from smpp_loss import SMPPLoss


class SMPPModel(nn.Module):
    """
    Social Media Popularity Prediction Model
    
    完整的多模态流行度预测模型，包含:
    1. CLIP 图像/文本编码器
    2. 双粒度 Prompt Learning
    3. 跨模态融合
    4. 多任务损失函数
    """
    
    def __init__(
        self,
        num_classes: int = 77,
        clip_model_name: str = "ViT-B/32",
        global_ctx_length: int = 16,
        local_ctx_length: int = 8,
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 2,
        prompt_dropout: float = 0.1,
        class_names: list = None,
        device: str = "cuda",
        freeze_image: bool = False,
        freeze_text: bool = True,
        freeze_text_embedding: bool = False
    ):
        """
        Args:
            num_classes: 类别数量 (77个细粒度子类)
            clip_model_name: CLIP 模型名称
            global_ctx_length: Global Prompt 长度
            local_ctx_length: Local Prompt 长度
            fusion_hidden_dim: 融合模块隐藏层维度
            fusion_num_heads: 融合模块注意力头数
            fusion_num_layers: 融合模块 Transformer 层数
            class_names: 类别名称列表
            device: 计算设备
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        
        # 1. 加载 CLIP 模型
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        # 强制使用 float32，减少数值溢出导致的 NaN
        self.clip_model = self.clip_model.float()
        self.feature_dim = self.clip_model.visual.output_dim
        
        # 2. 自定义文本编码器 (返回 h 和 H)
        self.text_encoder = CustomCLIPTextEncoder(self.clip_model)
        
        # 3. 双粒度 Prompt Learner
        self.prompt_learner = create_prompt_learner(
            clip_model=self.clip_model,
            num_classes=num_classes,
            class_names=class_names,
            global_ctx_length=global_ctx_length,
            local_ctx_length=local_ctx_length,
            prompt_dropout=prompt_dropout,
            device=device
        )
        
        # 4. 跨模态融合模块
        self.cross_modal_fusion = CrossModalFusion(
            image_dim=self.feature_dim,
            text_dim=self.feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers
        )
        
        # 5. 原型 (需要在训练前加载或初始化)
        self.register_buffer('visual_prototypes', torch.zeros(num_classes, self.feature_dim))
        self.register_buffer('textual_prototypes', torch.zeros(num_classes, self.feature_dim))
        
        # 6. 损失函数
        self.criterion = SMPPLoss(
            temperature=0.07,
            spatial_temperature=0.1,
            temperature_visual=0.07,
            temperature_textual=0.07
        )
        
        # 冻结 CLIP 的部分参数 (可选)
        self._freeze_clip_backbone(
            freeze_image=freeze_image,
            freeze_text=freeze_text,
            freeze_text_embedding=freeze_text_embedding
        )
    
    def _freeze_clip_backbone(
        self,
        freeze_image: bool = False,
        freeze_text: bool = False,
        freeze_text_embedding: bool = False
    ):
        """冻结 CLIP 的主干网络"""
        if freeze_image:
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
        
        if freeze_text:
            # 只冻结 transformer，保留 token_embedding 可训练
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False

        if freeze_text_embedding:
            if hasattr(self.clip_model, "token_embedding"):
                for param in self.clip_model.token_embedding.parameters():
                    param.requires_grad = False
            if hasattr(self.clip_model, "positional_embedding"):
                self.clip_model.positional_embedding.requires_grad = False
            if hasattr(self.clip_model, "ln_final"):
                for param in self.clip_model.ln_final.parameters():
                    param.requires_grad = False
            if hasattr(self.clip_model, "text_projection"):
                self.clip_model.text_projection.requires_grad = False
    
    def load_prototypes(self, visual_prototypes: torch.Tensor, textual_prototypes: torch.Tensor):
        """
        加载预先计算好的原型
        
        Args:
            visual_prototypes: [num_classes, feature_dim]
            textual_prototypes: [num_classes, feature_dim]
        """
        self.visual_prototypes = visual_prototypes.to(self.device)
        self.textual_prototypes = textual_prototypes.to(self.device)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        image_features = self.clip_model.encode_image(images)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def encode_text(self, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本
        
        Returns:
            h: 全局文本特征 [batch_size, feature_dim]
            H: 序列文本特征 [batch_size, seq_len, feature_dim]
        """
        h, H = self.text_encoder(text_tokens)
        return h, H
    
    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict:
        """
        前向传播
        
        Args:
            images: [batch_size, 3, H, W] 图像数据
            text_tokens: [batch_size, seq_len] tokenized 文本
            labels: [batch_size] 类别标签 (训练时必须提供)
            return_features: 是否返回中间特征 (用于特征提取)
        
        Returns:
            dict: 包含损失和预测结果
        """
        batch_size = images.shape[0]
        
        # 1. 编码图像
        image_features = self.encode_image(images)  # [B, feature_dim]
        
        # 2. 编码文本 (h: 全局, H: 序列)
        text_global, text_sequence = self.encode_text(text_tokens)
        # h: [B, feature_dim], H: [B, seq_len, feature_dim]
        
        # 3. 获取 Prompt 特征 (G: 全局, L: 局部)
        global_prompt_features, local_prompt_features = self.prompt_learner()
        # G: [num_classes, feature_dim], L: [num_classes, feature_dim]
        
        # 4. 为每个样本选择对应的原型 (或使用全部原型)
        # 这里简化处理，使用全部原型
        V = self.visual_prototypes.to(dtype=image_features.dtype)  # [num_classes, feature_dim]
        T = self.textual_prototypes.to(dtype=image_features.dtype)  # [num_classes, feature_dim]
        
        # 对于每个样本，计算其与所有原型的相似度，选择 top-k 或使用所有
        # 这里为了简化，使用平均原型或最相似的原型
        # 计算图像与视觉原型的相似度
        image_proto_sim = torch.matmul(image_features, V.T)  # [B, num_classes]
        top_k = 5  # 选择 top-5 最相关的原型
        _, top_indices = torch.topk(image_proto_sim, k=top_k, dim=1)  # [B, top_k]
        
        # 为每个样本选择其最相关的视觉和文本原型
        # 简化版本：使用加权平均
        weights = torch.softmax(image_proto_sim, dim=1)  # [B, num_classes]
        weighted_V = torch.matmul(weights, V)  # [B, feature_dim]
        weighted_T = torch.matmul(weights, T)  # [B, feature_dim]
        
        # 5. 跨模态融合
        enhanced_image, enhanced_visual_proto, enhanced_textual_proto = \
            self.cross_modal_fusion(
                image_features,
                weighted_V,
                weighted_T
            )
        # x̃, Ṽ, T̃: [B, fusion_hidden_dim]
        
        # 准备输出
        output = {
            'image_features': image_features,
            'text_global': text_global,
            'text_sequence': text_sequence,
            'enhanced_image': enhanced_image,
            'global_prompt_features': global_prompt_features,
            'local_prompt_features': local_prompt_features
        }
        
        # 6. 计算损失 (如果提供了标签)
        if labels is not None:
            # 需要为每个样本准备对应的原型
            # 简化处理：使用所有类别的原型
            enhanced_V_all = enhanced_visual_proto.unsqueeze(1).expand(-1, self.num_classes, -1)
            enhanced_T_all = enhanced_textual_proto.unsqueeze(1).expand(-1, self.num_classes, -1)
            
            # 更精确的做法：对所有原型进行融合
            # 这里为了与训练逻辑一致，对所有原型进行融合
            all_enhanced_V = []
            all_enhanced_T = []
            for i in range(self.num_classes):
                _, eV, eT = self.cross_modal_fusion(
                    image_features,
                    V[i:i+1].expand(batch_size, -1),
                    T[i:i+1].expand(batch_size, -1)
                )
                all_enhanced_V.append(eV)
                all_enhanced_T.append(eT)
            
            enhanced_V_all = torch.stack(all_enhanced_V, dim=1)  # [B, num_classes, hidden_dim]
            enhanced_T_all = torch.stack(all_enhanced_T, dim=1)  # [B, num_classes, hidden_dim]
            
            loss_dict = self.criterion(
                text_global_features=text_global,
                text_sequence_features=text_sequence,
                global_prompt_features=global_prompt_features,
                local_prompt_features=local_prompt_features,
                enhanced_image_features=enhanced_image,
                enhanced_visual_prototypes=enhanced_V_all,
                enhanced_textual_prototypes=enhanced_T_all,
                labels=labels
            )
            
            output.update(loss_dict)
        
        # 7. 如果需要返回所有特征（用于下游任务）
        if return_features:
            output['all_features'] = self.extract_all_features(
                image_features, text_global, text_sequence,
                enhanced_image, enhanced_visual_proto, enhanced_textual_proto,
                global_prompt_features, local_prompt_features
            )
        
        return output
    
    def extract_all_features(
        self,
        image_features,
        text_global,
        text_sequence,
        enhanced_image,
        enhanced_V,
        enhanced_T,
        global_prompts,
        local_prompts
    ) -> torch.Tensor:
        """
        提取所有特征用于下游回归任务
        
        F = [f_I(x), f_I(t), w_i^G, w_i^L, x̃, Ṽ, T̃, s]
        
        Returns:
            features: [batch_size, total_feature_dim]
        """
        batch_size = image_features.shape[0]
        
        # 获取每个样本对应类别的 prompt (需要预测或使用真实标签)
        # 这里简化：使用所有 prompt 的平均
        global_prompt_avg = global_prompts.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        local_prompt_avg = local_prompts.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        
        # 拼接所有特征
        all_features = torch.cat([
            image_features,         # f_I(x)
            text_global,            # f_T(t)
            global_prompt_avg,      # w^G
            local_prompt_avg,       # w^L
            enhanced_image,         # x̃
            enhanced_V,             # Ṽ
            enhanced_T              # T̃
            # 注意: 用户行为特征 s 需要从外部提供
        ], dim=-1)
        
        return all_features
    
    def predict_class(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Returns:
            predictions: [batch_size] 预测的类别
        """
        with torch.no_grad():
            output = self.forward(images, text_tokens, return_features=False)
            # 使用三个 logits 的平均进行预测
            logits = (
                output['logits_global'] +
                output['logits_local'] +
                output['logits_visual']
            ) / 3
            predictions = logits.argmax(dim=-1)
        
        return predictions
    
    def extract_features_for_regression(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        提取特征用于下游回归任务 (GBDT)
        
        Args:
            images: 图像数据
            text_tokens: 文本 tokens
            user_features: 用户行为特征 (可选)
        
        Returns:
            features: [batch_size, feature_dim] 用于回归的特征向量
        """
        with torch.no_grad():
            output = self.forward(images, text_tokens, return_features=True)
            features = output['all_features']
            
            # 如果提供了用户特征，拼接上
            if user_features is not None:
                features = torch.cat([features, user_features], dim=-1)
        
        return features


class SMPPTrainer:
    """
    SMPP 模型训练器
    """
    
    def __init__(
        self,
        model: SMPPModel,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, epoch: int):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_loss_g = 0
        total_loss_o = 0
        total_loss_c = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch depending on content
            if len(batch) == 3:
                images, text_tokens, labels = batch
            elif len(batch) >= 4:
                images, text_tokens, labels, popularities, *extra = batch
                
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            output = self.model(images, text_tokens, labels)
            
            # 反向传播
            loss = output['loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_loss_g += output['loss_global'].item()
            total_loss_o += output['loss_local'].item()
            total_loss_c += output['loss_visual'].item()
            
            # 计算准确率 (使用平均 logits)
            logits = (
                output['logits_global'] +
                output['logits_local'] +
                output['logits_visual']
            ) / 3
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100 * correct / total:.2f}%")
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100 * correct / total
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  L_g: {total_loss_g / len(dataloader):.4f}")
        print(f"  L_o: {total_loss_o / len(dataloader):.4f}")
        print(f"  L_c: {total_loss_c / len(dataloader):.4f}")
        print(f"  Accuracy: {avg_acc:.2f}%")
        
        return avg_loss, avg_acc
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch depending on content
                if len(batch) == 3:
                    images, text_tokens, labels = batch
                elif len(batch) >= 4:
                    images, text_tokens, labels, popularities, *extra = batch

                images = images.to(self.device)
                text_tokens = text_tokens.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(images, text_tokens, labels)
                
                total_loss += output['loss'].item()
                
                logits = (
                    output['logits_global'] +
                    output['logits_local'] +
                    output['logits_visual']
                ) / 3
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100 * correct / total
        
        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%")
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        model_is_dp = isinstance(self.model, torch.nn.DataParallel)
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

        # Align state_dict keys with model
        if model_is_dp and not has_module_prefix:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif (not model_is_dp) and has_module_prefix:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"Warning: Skipping optimizer state load due to mismatch: {e}")
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {path}, Epoch: {epoch}, Loss: {loss:.4f}")
        return epoch, loss
