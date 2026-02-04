"""
Prototype Builder Module
负责从数据集中构建 256-shot 的视觉和文本原型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import clip


class PrototypeBuilder:
    """
    构建多模态原型 (Visual & Textual Prototypes)
    为77个细粒度子类别各选取256个代表性样本
    """
    
    def __init__(
        self,
        num_classes: int = 77,
        num_shots: int = 256,
        device: str = "cuda"
    ):
        """
        Args:
            num_classes: 类别数量 (77个细粒度子类)
            num_shots: 每类采样数量
            device: 计算设备
        """
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.device = device
        
        # 加载 CLIP 模型用于特征提取
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
    @torch.no_grad()
    def build_visual_prototypes(
        self, 
        image_samples: Dict[int, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        构建视觉原型
        
        V_i = (1/256) * Σ f_I(x_n)
        
        Args:
            image_samples: {class_id: [image_tensor1, image_tensor2, ...]}
                          每个类别包含256张代表性图片
        
        Returns:
            visual_prototypes: [num_classes, feature_dim] 视觉原型矩阵
        """
        visual_prototypes = []
        
        for class_id in range(self.num_classes):
            if class_id not in image_samples:
                raise ValueError(f"Missing samples for class {class_id}")
            
            images = image_samples[class_id]
            if len(images) != self.num_shots:
                print(f"Warning: Class {class_id} has {len(images)} samples, expected {self.num_shots}")
            
            # 批量编码图像
            image_batch = torch.stack(images).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 取平均作为原型
            prototype = image_features.mean(dim=0)
            prototype = prototype / prototype.norm()  # 归一化
            visual_prototypes.append(prototype)
        
        visual_prototypes = torch.stack(visual_prototypes)  # [num_classes, feature_dim]
        return visual_prototypes
    
    @torch.no_grad()
    def build_textual_prototypes(
        self,
        text_samples: Dict[int, List[str]]
    ) -> torch.Tensor:
        """
        构建文本原型
        
        T_i = (1/256) * Σ f_T(t_n)
        
        Args:
            text_samples: {class_id: [title1, title2, ...]}
                         每个类别包含256个描述性标题
        
        Returns:
            textual_prototypes: [num_classes, feature_dim] 文本原型矩阵
        """
        textual_prototypes = []
        
        for class_id in range(self.num_classes):
            if class_id not in text_samples:
                raise ValueError(f"Missing text samples for class {class_id}")
            
            texts = text_samples[class_id]
            if len(texts) != self.num_shots:
                print(f"Warning: Class {class_id} has {len(texts)} samples, expected {self.num_shots}")
            
            # 批量编码文本
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 取平均作为原型
            prototype = text_features.mean(dim=0)
            prototype = prototype / prototype.norm()  # 归一化
            textual_prototypes.append(prototype)
        
        textual_prototypes = torch.stack(textual_prototypes)  # [num_classes, feature_dim]
        return textual_prototypes
    
    def sample_diverse_data(
        self,
        dataset,
        class_id: int,
        sampling_strategy: str = "diverse"
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        采样策略: 考虑时间多样性、语义多样性和用户多样性
        
        Args:
            dataset: 数据集对象
            class_id: 类别ID
            sampling_strategy: 采样策略 ("diverse", "random", "temporal")
        
        Returns:
            images: 采样的图像列表
            texts: 采样的文本列表
        """
        # 获取该类别的所有样本
        class_samples = dataset.get_samples_by_class(class_id)
        if len(class_samples) == 0:
            raise ValueError(f"No samples found for class {class_id}")
        
        if sampling_strategy == "diverse":
            # 时间多样性: 从不同时间段采样
            samples = self._temporal_diverse_sampling(class_samples)
            
            # 用户多样性: 尽量选择不同用户的帖子
            samples = self._user_diverse_sampling(samples)
            
            # 语义多样性: 子话题过滤
            samples = self._semantic_diverse_sampling(samples)
            samples = self._pad_to_num_shots(samples, class_samples)
            
        elif sampling_strategy == "random":
            # Perform random sampling if the strategy is set to random
            samples = list(np.random.choice(class_samples, min(len(class_samples), self.num_shots), replace=False))
            samples = self._pad_to_num_shots(samples, class_samples)
        
        elif sampling_strategy == "temporal":
            # 仅时间采样容易丢失用户/语义多样性，这里做混合采样增强鲁棒性
            samples = self._temporal_diverse_sampling(class_samples)
            samples = self._user_diverse_sampling(samples)
            samples = self._semantic_diverse_sampling(samples)
            samples = self._pad_to_num_shots(samples, class_samples)
        
        else:
            # Default to random sampling if no valid strategy is provided
            samples = list(np.random.choice(class_samples, min(len(class_samples), self.num_shots), replace=False))
            samples = self._pad_to_num_shots(samples, class_samples)
        
        images = [sample['image'] for sample in samples]
        texts = [sample['title'] for sample in samples]
        
        return images, texts
    
    def _temporal_diverse_sampling(self, samples):
        """时间多样性采样（对缺失时间戳更鲁棒）"""
        if len(samples) <= self.num_shots:
            return list(samples)

        # 过滤无效时间戳，避免 None 集中在最前导致采样失真
        valid = [s for s in samples if s.get('timestamp') is not None]
        invalid = [s for s in samples if s.get('timestamp') is None]

        # 如果有效时间戳过少，退化为随机补齐
        if len(valid) == 0:
            return list(np.random.choice(samples, self.num_shots, replace=False))

        # 按时间戳排序并均匀采样
        sorted_samples = sorted(valid, key=lambda x: x.get('timestamp'))
        if len(sorted_samples) >= self.num_shots:
            indices = np.linspace(0, len(sorted_samples) - 1, num=self.num_shots, dtype=int)
            selected = [sorted_samples[i] for i in indices]
        else:
            selected = list(sorted_samples)

        # 如不足 num_shots，用剩余样本随机补齐（优先使用无效时间戳样本）
        if len(selected) < self.num_shots:
            needed = self.num_shots - len(selected)
            pool = invalid + [s for s in valid if id(s) not in set(id(x) for x in selected)]
            if pool:
                selected.extend(list(np.random.choice(pool, min(needed, len(pool)), replace=False)))
        return selected
    
    def _user_diverse_sampling(self, samples):
        """用户多样性采样"""
        # 优先选择来自不同用户的样本
        user_dict = {}
        for sample in samples:
            user_id = sample['user_id']
            if user_id not in user_dict:
                user_dict[user_id] = []
            user_dict[user_id].append(sample)
        
        selected = []
        users = list(user_dict.keys())
        idx = 0
        max_iters = len(samples) * 2
        while len(selected) < self.num_shots and idx < max_iters and users:
            user = users[idx % len(users)]
            if user_dict.get(user):
                selected.append(user_dict[user].pop(0))
            idx += 1

        return selected
    
    def _semantic_diverse_sampling(self, samples):
        """语义多样性采样 (子话题过滤)"""
        # 使用聚类或其他方法确保样本覆盖不同子话题
        # 简化版本: 基于文本相似度筛选
        if len(samples) <= self.num_shots:
            return list(samples)

        def _tokens(text: str):
            return set(t for t in text.lower().split() if t)

        def _jaccard(a, b):
            if not a or not b:
                return 0.0
            return len(a & b) / max(1, len(a | b))

        selected = [samples[0]]
        selected_tokens = [_tokens(samples[0].get("title", ""))]
        for sample in samples[1:]:
            if len(selected) >= self.num_shots:
                break
            toks = _tokens(sample.get("title", ""))
            sim = max((_jaccard(toks, t) for t in selected_tokens), default=0.0)
            if sim < 0.5:
                selected.append(sample)
                selected_tokens.append(toks)

        return selected

    def _pad_to_num_shots(self, selected, all_samples):
        """如果样本不足，随机补齐到 num_shots。"""
        selected = list(selected)
        if len(selected) == 0:
            raise ValueError("No samples available to pad.")
        if len(selected) >= self.num_shots:
            return selected[: self.num_shots]

        # 补齐：优先使用未选样本，不足则有放回采样
        selected_set = set(id(s) for s in selected)
        remaining = [s for s in all_samples if id(s) not in selected_set]
        if remaining:
            needed = self.num_shots - len(selected)
            take = remaining[:needed]
            selected.extend(take)
        if len(selected) < self.num_shots:
            needed = self.num_shots - len(selected)
            selected.extend(list(np.random.choice(all_samples, needed, replace=True)))
        return selected
    
    def save_prototypes(self, visual_prototypes, textual_prototypes, save_path):
        """保存原型到文件"""
        torch.save({
            'visual_prototypes': visual_prototypes,
            'textual_prototypes': textual_prototypes,
            'num_classes': self.num_classes,
            'num_shots': self.num_shots
        }, save_path)
        print(f"Prototypes saved to {save_path}")
    
    def load_prototypes(self, load_path):
        """从文件加载原型"""
        checkpoint = torch.load(load_path)
        return checkpoint['visual_prototypes'], checkpoint['textual_prototypes']

