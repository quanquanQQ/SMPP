"""
Dataset  —  论文 Section 4.1.1

SMPD-Image: 486k 图文帖子，70k 用户，Flickr 来源
类别体系：原始 11 大类 -> 本框架使用 77 细粒度子类

用户行为特征 s（论文公式 11，来自 HVLA [33]）：
    hashtag 频率向量（hashtag_top_k + 1 维）
    通过 include_user_features=True 启用
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class JSONDataset(Dataset):
    """
    基于 JSON/TXT 文件结构的数据集加载器。

    目录结构期望：
        metadata_dir/
            {split}_img_filepath.txt
            {split}_text.json
            {split}_category.json
            {split}_label.txt
    """

    def __init__(
        self,
        metadata_dir: str,
        image_dir: str,
        clip_preprocess,
        split: str = "train",
        text_field: str = "Title",
        include_user_features: bool = False,
        class_mapping: Optional[Dict[str, int]] = None,
        is_training: Optional[bool] = None,
        hashtag_min_freq: int = 5,
        hashtag_top_k: int = 300,
        hashtag_other_name: str = "other",
    ):
        super().__init__()
        self.metadata_dir    = Path(metadata_dir)
        self.image_dir       = Path(image_dir)
        self.clip_preprocess = clip_preprocess
        self.text_field      = text_field
        self.include_user_features = include_user_features
        self.is_training     = (split == "train") if is_training is None else is_training
        self.data_prefix     = split

        self.hashtag_min_freq  = hashtag_min_freq
        self.hashtag_top_k     = hashtag_top_k
        self.hashtag_other_name = hashtag_other_name

        # 训练时数据增强
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

        self._missing_image_logs = 0
        self._load_data()

        # 类别映射
        self.class_mapping = class_mapping
        if self.class_mapping is None:
            unique_cats = sorted({
                item["Category"] for item in self.categories
                if isinstance(item, dict) and "Category" in item
            })
            self.class_mapping = {cat: i for i, cat in enumerate(unique_cats)}

        # 按类别建立索引
        self.class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, cat_item in enumerate(self.categories):
            cat_name = cat_item.get("Category") if isinstance(cat_item, dict) else None
            if cat_name is None:
                continue
            class_id = self.class_mapping.get(cat_name)
            if class_id is not None:
                self.class_indices[class_id].append(idx)

        # 用户行为特征：构建 hashtag 词表
        if include_user_features:
            self._build_hashtag_vocab()

        print(f"[JSONDataset] {split}: {len(self.image_paths)} samples, "
              f"{len(self.class_mapping)} classes")

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load_data(self):
        with open(self.metadata_dir / f"{self.data_prefix}_img_filepath.txt") as f:
            self.image_paths = [line.strip() for line in f]

        with open(self.metadata_dir / f"{self.data_prefix}_text.json") as f:
            self.text_data = json.load(f)

        with open(self.metadata_dir / f"{self.data_prefix}_category.json") as f:
            self.categories = json.load(f)

        label_file = self.metadata_dir / f"{self.data_prefix}_label.txt"
        if label_file.exists():
            with open(label_file) as f:
                self.popularities = [float(line.strip()) for line in f]
        else:
            self.popularities = [0.0] * len(self.image_paths)

    # ------------------------------------------------------------------
    # 图片路径解析（带 split-prefix fallback）
    # ------------------------------------------------------------------

    def _resolve_image_path(self, img_rel_path: str) -> Path:
        rel  = Path(img_rel_path)
        candidates = []

        if rel.is_absolute():
            candidates.append(rel)
        else:
            candidates.append(self.image_dir / rel)

        if rel.parts:
            prefix = rel.parts[0]
            rest   = rel.parts[1:]
            alt = {
                "train": "train_new", "train_new": "train",
                "test":  "test_new",  "test_new":  "test",
            }.get(prefix)
            if alt:
                candidates.append(self.image_dir / Path(alt).joinpath(*rest))
            if prefix in {"train", "train_new", "test", "test_new"}:
                candidates.append(self.image_dir / Path(self.data_prefix).joinpath(*rest))
        else:
            candidates.append(self.image_dir / self.data_prefix / rel)

        seen = set()
        for c in candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                if c.exists():
                    return c
        return candidates[0]

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 图像
        img_path = self._resolve_image_path(self.image_paths[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.is_training:
                image = self.augment(image)
            image = self.clip_preprocess(image)
        except Exception as e:
            if self._missing_image_logs < 20:
                print(f"[Dataset] Image load error {img_path}: {e}")
                self._missing_image_logs += 1
            image = torch.zeros(3, 224, 224)

        # 文本
        text_item    = self.text_data[idx] if idx < len(self.text_data) else {}
        text_content = text_item.get(self.text_field, "") if isinstance(text_item, dict) else ""
        if not isinstance(text_content, str):
            text_content = ""
        text_tokens = clip.tokenize([text_content[:77]], truncate=True)[0]

        # 类别标签
        cat_item = self.categories[idx]
        cat_name = cat_item["Category"] if isinstance(cat_item, dict) else ""
        label    = self.class_mapping.get(cat_name, 0)

        # 流行度
        popularity = float(self.popularities[idx])

        if self.include_user_features:
            user_feat = self._extract_user_features(idx)
            return image, text_tokens, label, popularity, user_feat

        return image, text_tokens, label, popularity

    # ------------------------------------------------------------------
    # 原型构建辅助
    # ------------------------------------------------------------------

    def get_samples_by_class(self, class_id: int) -> List[Dict]:
        """
        返回指定类别的样本列表，每个样本为 dict：
            {"image": tensor, "title": str, "user_id": str|None, "timestamp": None}
        """
        indices = self.class_indices.get(class_id, [])
        samples = []
        for idx in indices:
            img_path = self._resolve_image_path(self.image_paths[idx])
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.clip_preprocess(image)
            except Exception:
                image = torch.zeros(3, 224, 224)

            text_item = self.text_data[idx] if idx < len(self.text_data) else {}
            if isinstance(text_item, dict):
                title   = text_item.get(self.text_field, "")
                user_id = text_item.get("Uid")
            else:
                title   = ""
                user_id = None
            if not isinstance(title, str):
                title = ""

            samples.append({
                "image":     image,
                "title":     title,
                "user_id":   user_id,
                "timestamp": None,
            })
        return samples

    # ------------------------------------------------------------------
    # 用户行为特征 s（论文公式 11，HVLA [33]）
    # ------------------------------------------------------------------

    def _build_hashtag_vocab(self):
        """
        统计所有样本中 tag 的出现频率，构建 hashtag 词表。
        词表大小 = hashtag_top_k + 1（最后一维为 'other'）。
        """
        counter = Counter()
        tag_field = "All_tags" if self.text_field == "Title" else self.text_field

        for item in self.text_data:
            tags_str = ""
            if isinstance(item, dict):
                tags_str = item.get("All_tags", item.get("tags", ""))
            if isinstance(tags_str, str) and tags_str.strip():
                for tag in tags_str.replace(",", " ").split():
                    t = tag.strip().lower()
                    if t:
                        counter[t] += 1

        filtered  = {t: c for t, c in counter.items()
                     if c >= self.hashtag_min_freq}
        top_tags  = sorted(filtered, key=lambda t: filtered[t], reverse=True)
        top_tags  = top_tags[: self.hashtag_top_k]

        self.hashtag_vocab      = {tag: i for i, tag in enumerate(top_tags)}
        self.hashtag_vocab_size = len(top_tags) + 1        # +1 for 'other'
        self.hashtag_other_idx  = len(top_tags)

        print(f"[Dataset] Hashtag vocab: {len(top_tags)} tags "
              f"(min_freq={self.hashtag_min_freq})")

    def _extract_user_features(self, idx: int) -> torch.Tensor:
        """
        提取 hashtag 频率向量（L1 归一化）。
        维度 = hashtag_vocab_size = hashtag_top_k + 1
        """
        if not hasattr(self, "hashtag_vocab"):
            return torch.zeros(1)

        text_item = self.text_data[idx] if idx < len(self.text_data) else {}
        tags_str  = ""
        if isinstance(text_item, dict):
            tags_str = text_item.get("All_tags", text_item.get("tags", ""))

        feat = torch.zeros(self.hashtag_vocab_size)
        if isinstance(tags_str, str) and tags_str.strip():
            for tag in tags_str.replace(",", " ").split():
                t = tag.strip().lower()
                if t:
                    i = self.hashtag_vocab.get(t, self.hashtag_other_idx)
                    feat[i] += 1.0
            total = feat.sum()
            if total > 0:
                feat = feat / total

        return feat

    # ------------------------------------------------------------------
    # 统计工具
    # ------------------------------------------------------------------

    def print_statistics(self):
        print("\n" + "=" * 50)
        print(f"Dataset split: {self.data_prefix}")
        print(f"Total samples: {len(self.image_paths)}")
        print(f"Classes:       {len(self.class_mapping)}")

        pop = self.popularities
        if pop:
            arr = np.array(pop)
            print(f"Popularity — mean={arr.mean():.2f}  std={arr.std():.2f}  "
                  f"min={arr.min():.2f}  max={arr.max():.2f}")
        print("=" * 50 + "\n")

    def plot_distribution(self):
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # 类别分布
            counts = [len(self.class_indices.get(i, []))
                      for i in range(len(self.class_mapping))]
            axes[0].bar(range(len(counts)), counts)
            axes[0].set_title(f"Class Distribution ({self.data_prefix})")
            axes[0].set_xlabel("Class ID")
            axes[0].set_ylabel("Count")

            # 流行度分布
            axes[1].hist(self.popularities, bins=50)
            axes[1].set_title("Popularity Distribution")
            axes[1].set_xlabel("Popularity")

            plt.tight_layout()
            plt.savefig(f"distribution_{self.data_prefix}.png")
            plt.close()
            print(f"[Dataset] Distribution saved to distribution_{self.data_prefix}.png")
        except Exception as e:
            print(f"[Dataset] plot_distribution failed: {e}")


# Alias for backward compatibility
DatasetStatistics = JSONDataset