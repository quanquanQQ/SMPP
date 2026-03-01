"""
Dataset and DataLoader Implementation
数据集和数据加载器实现
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import json
import os
from collections import defaultdict
from torchvision import transforms

class JSONDataset(Dataset):
    """
    Dataset loader for JSON/TXT file structure
    """
    def __init__(
        self,
        metadata_dir: str,
        image_dir: str,
        clip_preprocess,
        split: str = "train", # "train" or "test"
        text_field: str = "Title",
        include_user_features: bool = False,
        class_mapping: Optional[Dict[str, int]] = None,
        is_training: Optional[bool] = None
    ):
        """
        Args:
            metadata_dir: Directory containing JSON/TXT files
            image_dir: Directory containing images
            split: "train" or "test"
            ...
        """
        super().__init__()
        self.metadata_dir = Path(metadata_dir)
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess
        self.text_field = text_field
        self.include_user_features = False
        self.is_training = (split == "train") if is_training is None else is_training

        # Data augmentation for training only
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
        # Load data
        self.data_prefix = split
        self._load_data()
        
        # Handle categories
        self.class_mapping = class_mapping
        if self.class_mapping is None:
            # Create mapping or use existing
            unique_cats = sorted(list(set(item['Category'] for item in self.categories)))
            self.class_mapping = {cat: idx for idx, cat in enumerate(unique_cats)}
        
        # Build class indices for sampling
        self.class_indices = defaultdict(list)
        for idx, cat_item in enumerate(self.categories):
            cat_name = cat_item.get('Category') if isinstance(cat_item, dict) else None
            if cat_name is None:
                continue
            class_id = self.class_mapping.get(cat_name)
            if class_id is None:
                continue
            self.class_indices[class_id].append(idx)
            
        print(f"Loaded {len(self.image_paths)} samples from {metadata_dir}")

    def _load_data(self):
        # 1. Image Paths
        with open(self.metadata_dir / f"{self.data_prefix}_img_filepath.txt", 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
            
        # 2. Text Data
        with open(self.metadata_dir / f"{self.data_prefix}_text.json", 'r') as f:
            self.text_data = json.load(f)
            
        # 3. Category Data
        with open(self.metadata_dir / f"{self.data_prefix}_category.json", 'r') as f:
            self.categories = json.load(f)
            
        # 4. Label/Popularity Data
        # Note: train_label.txt contains popularity scores
        label_file = self.metadata_dir / f"{self.data_prefix}_label.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                self.popularities = [float(line.strip()) for line in f.readlines()]
        else:
             # For test set, might not have labels or might be in different format
             # If missing, use zeros
             self.popularities = [0.0] * len(self.image_paths)


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        # 1. Image
        # Image paths are like "train/..." or "test/..."
        # image_dir should be parent of "train" and "test" folders
        img_rel_path = self.image_paths[idx]
        image_path = self.image_dir / img_rel_path
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.is_training:
                image = self.augment(image)
            image = self.clip_preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy image or handle error
            image = torch.zeros((3, 224, 224)) 

        # 2. Text
        text_item = self.text_data[idx]
        text_content = text_item.get(self.text_field, "")
        if not isinstance(text_content, str):
            text_content = ""
        text_tokens = clip.tokenize([text_content[:77]], truncate=True)[0]

        # 3. Label (Category)
        cat_item = self.categories[idx]
        cat_name = cat_item['Category']
        label = self.class_mapping.get(cat_name, 0)
        
        # 4. Popularity
        popularity = self.popularities[idx]
        
        return image, text_tokens, label, popularity

    def get_samples_by_class(self, class_id: int) -> List[Dict]:
        """
        Return raw samples for prototype building.
        Each sample contains image tensor and title text.
        """
        indices = self.class_indices.get(class_id, [])
        samples: List[Dict] = []
        for idx in indices:
            # Image
            img_rel_path = self.image_paths[idx]
            image_path = self.image_dir / img_rel_path
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.clip_preprocess(image)
            except Exception:
                image = torch.zeros((3, 224, 224))

            # Text
            text_item = self.text_data[idx] if idx < len(self.text_data) else {}
            if isinstance(text_item, dict):
                title = text_item.get(self.text_field, "")
            else:
                title = ""
            if not isinstance(title, str):
                title = ""

            samples.append({
                "image": image,
                "title": title
            })

        return samples


class SMPDataset(Dataset):
    """
    Social Media Popularity Dataset
    
    数据格式:
    - image_path: 图像文件路径
    - title: 帖子标题
    - tags: 所有标签 (可选)
    - category: 类别标签 (0-76)
    - popularity: 流行度数值 (点赞数、转发数等)
    - user_features: 用户行为特征 (可选)
    """
    
    def __init__(
        self,
        data_csv: str,
        image_dir: str,
        clip_preprocess,
        text_field: str = "title",  # "title" or "all_tags"
        max_text_length: int = 77,
        include_user_features: bool = False,
        user_feature_dim: int = 20
    ):
        """
        Args:
            data_csv: CSV 文件路径，包含所有样本信息
            image_dir: 图像文件目录
            clip_preprocess: CLIP 图像预处理函数
            text_field: 使用的文本字段 ("title" 或 "all_tags")
            max_text_length: 最大文本长度
            include_user_features: 是否包含用户特征
            user_feature_dim: 用户特征维度
        """
        super().__init__()
        
        self.data = pd.read_csv(data_csv)
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess
        self.text_field = text_field
        self.max_text_length = max_text_length
        self.include_user_features = include_user_features
        self.user_feature_dim = user_feature_dim
        
        print(f"Loaded {len(self.data)} samples from {data_csv}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            image: [3, H, W] 预处理后的图像
            text_tokens: [seq_len] tokenized 文本
            label: 类别标签 (0-76)
            popularity: 流行度数值
            user_features: [user_feature_dim] 用户特征 (可选)
        """
        row = self.data.iloc[idx]
        
        # 1. 加载并预处理图像
        image_path = self.image_dir / row['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.clip_preprocess(image)
        
        # 2. 处理文本
        text = row[self.text_field] if pd.notna(row[self.text_field]) else ""
        text_tokens = clip.tokenize([text], truncate=True)[0]  # [seq_len]
        
        # 3. 类别标签
        label = int(row['category'])
        
        # 4. 流行度数值
        popularity = float(row['popularity'])
        
        # 5. 用户特征 (可选)
        if self.include_user_features:
            # 从数据中提取用户特征
            user_features = self._extract_user_features(row)
            return image, text_tokens, label, popularity, user_features
        
        return image, text_tokens, label, popularity
    
    def _extract_user_features(self, row) -> torch.Tensor:
        """
        提取用户行为特征
        
        根据文献 [33] 的用户特征定义
        可能包括: 粉丝数、发帖数、账号年龄等
        """
        feature_names = [
            'follower_count', 'following_count', 'post_count',
            'verified', 'account_age_days', 'avg_likes',
            'avg_retweets', 'avg_comments', 'engagement_rate',
            'posting_frequency', 'reply_rate', 'quote_rate',
            # ... 更多特征
        ]
        
        features = []
        for feat_name in feature_names[:self.user_feature_dim]:
            if feat_name in row and pd.notna(row[feat_name]):
                features.append(float(row[feat_name]))
            else:
                features.append(0.0)
        
        # 归一化
        features = torch.tensor(features, dtype=torch.float32)
        # features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features
    
    def get_samples_by_class(self, class_id: int) -> list:
        """获取指定类别的所有样本"""
        class_data = self.data[self.data['category'] == class_id]
        return [self[i] for i in class_data.index]


class PrototypeDataset(Dataset):
    """
    用于构建原型的特殊数据集
    每个类别返回 256 个代表性样本
    """
    
    def __init__(
        self,
        data_csv: str,
        image_dir: str,
        clip_preprocess,
        num_classes: int = 77,
        num_shots: int = 256
    ):
        super().__init__()
        
        self.data = pd.read_csv(data_csv)
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess
        self.num_classes = num_classes
        self.num_shots = num_shots
        
        # 为每个类别采样数据
        self.class_samples = self._sample_prototypes()
    
    def _sample_prototypes(self) -> dict:
        """
        为每个类别采样代表性样本
        使用随机采样（移除时间序列采样）
        """
        class_samples = {}
        
        for class_id in range(self.num_classes):
            class_data = self.data[self.data['category'] == class_id]
            
            if len(class_data) < self.num_shots:
                print(f"Warning: Class {class_id} has only {len(class_data)} samples")
                sampled = class_data
            else:
                # 随机采样
                sampled = class_data.sample(n=self.num_shots, replace=False, random_state=None)
            
            class_samples[class_id] = sampled.index.tolist()
        
        return class_samples
    
    def get_class_samples(self, class_id: int) -> Tuple[list, list]:
        """
        获取指定类别的图像和文本样本
        
        Returns:
            images: 预处理后的图像列表
            texts: 文本列表
        """
        indices = self.class_samples[class_id]
        
        images = []
        texts = []
        
        for idx in indices:
            row = self.data.iloc[idx]
            
            # 加载图像
            image_path = self.image_dir / row['image_path']
            image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(image)
            images.append(image)
            
            # 提取文本
            text = row['title'] if pd.notna(row['title']) else ""
            texts.append(text)
        
        return images, texts


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_dir: str,
    clip_preprocess,
    batch_size: int = 128,
    num_workers: int = 0,
    include_user_features: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_csv: 训练集 CSV
        val_csv: 验证集 CSV
        test_csv: 测试集 CSV
        image_dir: 图像目录
        clip_preprocess: CLIP 预处理函数
        batch_size: 批次大小
        num_workers: 数据加载线程数
        include_user_features: 是否包含用户特征
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = SMPDataset(
        train_csv, image_dir, clip_preprocess,
        include_user_features=include_user_features
    )
    
    val_dataset = SMPDataset(
        val_csv, image_dir, clip_preprocess,
        include_user_features=include_user_features
    )
    
    test_dataset = SMPDataset(
        test_csv, image_dir, clip_preprocess,
        include_user_features=include_user_features
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_fn_with_user_features(batch):
    """
    自定义 collate 函数，处理包含用户特征的数据
    """
    if len(batch[0]) == 5:  # 包含用户特征
        images, text_tokens, labels, popularities, user_features = zip(*batch)
        
        images = torch.stack(images)
        text_tokens = torch.stack(text_tokens)
        labels = torch.tensor(labels)
        popularities = torch.tensor(popularities)
        user_features = torch.stack(user_features)
        
        return images, text_tokens, labels, popularities, user_features
    
    else:  # 不包含用户特征
        images, text_tokens, labels, popularities = zip(*batch)
        
        images = torch.stack(images)
        text_tokens = torch.stack(text_tokens)
        labels = torch.tensor(labels)
        popularities = torch.tensor(popularities)
        
        return images, text_tokens, labels, popularities


# 数据集统计和可视化工具
class DatasetStatistics:
    """数据集统计工具"""
    
    def __init__(self, dataset: SMPDataset):
        self.dataset = dataset
        self.data = dataset.data
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print("\n" + "="*50)
        print("Dataset Statistics")
        print("="*50)
        
        print(f"Total samples: {len(self.data)}")
        print(f"Number of classes: {self.data['category'].nunique()}")
        
        print("\nClass distribution:")
        class_counts = self.data['category'].value_counts().sort_index()
        for class_id, count in class_counts.items():
            print(f"  Class {class_id}: {count} samples")
        
        print(f"\nPopularity statistics:")
        print(f"  Mean: {self.data['popularity'].mean():.2f}")
        print(f"  Median: {self.data['popularity'].median():.2f}")
        print(f"  Std: {self.data['popularity'].std():.2f}")
        print(f"  Min: {self.data['popularity'].min():.2f}")
        print(f"  Max: {self.data['popularity'].max():.2f}")
        
        if 'title' in self.data.columns:
            avg_title_length = self.data['title'].str.len().mean()
            print(f"\nAverage title length: {avg_title_length:.1f} characters")
        
        print("="*50 + "\n")
    
    def plot_distribution(self, save_path: str = None):
        """绘制数据分布图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 类别分布
        self.data['category'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], title='Class Distribution'
        )
        axes[0, 0].set_xlabel('Class ID')
        axes[0, 0].set_ylabel('Count')
        
        # 流行度分布
        self.data['popularity'].hist(bins=50, ax=axes[0, 1])
        axes[0, 1].set_title('Popularity Distribution')
        axes[0, 1].set_xlabel('Popularity')
        axes[0, 1].set_ylabel('Frequency')
        
        # 流行度对数分布
        self.data['popularity'].apply(np.log1p).hist(bins=50, ax=axes[1, 0])
        axes[1, 0].set_title('Log Popularity Distribution')
        axes[1, 0].set_xlabel('Log(Popularity + 1)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 类别 vs 平均流行度
        avg_pop = self.data.groupby('category')['popularity'].mean()
        avg_pop.plot(kind='bar', ax=axes[1, 1], title='Avg Popularity by Class')
        axes[1, 1].set_xlabel('Class ID')
        axes[1, 1].set_ylabel('Avg Popularity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
