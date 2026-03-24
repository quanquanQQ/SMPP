"""
Prototype Builder Module  —  论文 Section 3.1
构建 256-shot 多模态原型（Visual & Textual）

公式：
    V_i = (1/256) * sum_{n=1}^{256} f_I(x_n)   （均值后 L2 归一化）
    T_i = (1/256) * sum_{n=1}^{256} f_T(t_n)   （均值后 L2 归一化）

采样策略（三阶段有序执行）：
    Stage 1: Temporal Diversity   — 时间轴等分，每段随机取 1 个样本
    Stage 2: Semantic Diversity   — 按 subtopic 分组轮询，覆盖多个子话题
    Stage 3: User Diversity       — 按 user_id 分组轮询，避免单一用户偏差
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import clip


class PrototypeBuilder:
    def __init__(
        self,
        num_classes: int = 77,
        num_shots: int = 256,
        device: str = "cuda",
    ):
        self.num_classes = num_classes
        self.num_shots   = num_shots
        self.device      = device

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()

    # ---------------------------------------------------------------
    # 原型构建
    # ---------------------------------------------------------------

    @torch.no_grad()
    def build_visual_prototypes(
        self,
        image_samples: Dict[int, List[torch.Tensor]],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        构建视觉原型矩阵。

        Steps:
            1. 批量送入 clip_model.encode_image()
            2. 每张图 L2 归一化
            3. 取均值
            4. 均值再 L2 归一化

        Returns:
            [num_classes, feature_dim]
        """
        feature_dim = self.clip_model.visual.output_dim
        prototypes  = []

        for class_id in range(self.num_classes):
            images = image_samples.get(class_id, [])
            if not images:
                prototypes.append(torch.zeros(feature_dim, device=self.device))
                continue

            if len(images) != self.num_shots:
                print(f"[PrototypeBuilder] Class {class_id}: "
                      f"{len(images)} images (expected {self.num_shots})")

            all_feats = []
            for i in range(0, len(images), batch_size):
                batch = torch.stack(images[i: i + batch_size]).to(self.device)
                feats = self.clip_model.encode_image(batch).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats)

            all_feats = torch.cat(all_feats, dim=0)   # [N, d]
            proto     = all_feats.mean(dim=0)          # [d]
            proto     = proto / proto.norm()
            prototypes.append(proto)

        return torch.stack(prototypes)   # [K, d]

    @torch.no_grad()
    def build_textual_prototypes(
        self,
        text_samples: Dict[int, List[str]],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        构建文本原型矩阵。

        Steps:
            1. clip.tokenize()
            2. clip_model.encode_text()
            3. L2 归一化，取均值，再归一化

        Returns:
            [num_classes, feature_dim]
        """
        feature_dim = self.clip_model.visual.output_dim
        prototypes  = []

        for class_id in range(self.num_classes):
            texts = text_samples.get(class_id, [])
            if not texts:
                prototypes.append(torch.zeros(feature_dim, device=self.device))
                continue

            if len(texts) != self.num_shots:
                print(f"[PrototypeBuilder] Class {class_id}: "
                      f"{len(texts)} texts (expected {self.num_shots})")

            all_feats = []
            for i in range(0, len(texts), batch_size):
                tokens = clip.tokenize(texts[i: i + batch_size],
                                       truncate=True).to(self.device)
                feats = self.clip_model.encode_text(tokens).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats)

            all_feats = torch.cat(all_feats, dim=0)
            proto     = all_feats.mean(dim=0)
            proto     = proto / proto.norm()
            prototypes.append(proto)

        return torch.stack(prototypes)

    # ---------------------------------------------------------------
    # 三阶段多样性采样入口
    # ---------------------------------------------------------------

    def sample_diverse_data(
        self,
        dataset,
        class_id: int,
        sampling_strategy: str = "diverse",
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        采样策略入口。

        Args:
            dataset:            实现了 get_samples_by_class(class_id) 的数据集
            class_id:           类别 ID
            sampling_strategy:  "diverse"（三阶段）或 "random"

        Returns:
            (images, texts)  列表长度 <= num_shots
        """
        class_samples = dataset.get_samples_by_class(class_id)
        if not class_samples:
            return [], []

        if sampling_strategy == "diverse":
            # --- Stage 1: Temporal ---
            try:
                samples = self._temporal_diverse_sampling(class_samples)
            except Exception as e:
                print(f"[Sampling] Class {class_id} temporal stage failed: {e}. "
                      "Using all samples.")
                samples = class_samples

            # --- Stage 2: Semantic (subtopic) ---
            try:
                samples = self._semantic_diverse_sampling(samples)
            except Exception as e:
                print(f"[Sampling] Class {class_id} semantic stage failed: {e}.")

            # --- Stage 3: User ---
            try:
                samples = self._user_diverse_sampling(samples)
            except Exception as e:
                print(f"[Sampling] Class {class_id} user stage failed: {e}.")

            # 不足 num_shots 时，从原始集合随机补充
            if len(samples) < self.num_shots:
                sampled_ids = {id(s) for s in samples}
                remaining   = [s for s in class_samples if id(s) not in sampled_ids]
                if remaining:
                    extra_n = min(self.num_shots - len(samples), len(remaining))
                    extra_i = np.random.choice(len(remaining), extra_n, replace=False)
                    samples = samples + [remaining[i] for i in extra_i]

        elif sampling_strategy == "random":
            if len(class_samples) <= self.num_shots:
                samples = class_samples
            else:
                idxs    = np.random.choice(len(class_samples),
                                           self.num_shots, replace=False)
                samples = [class_samples[i] for i in idxs]
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy!r}")

        images = [s["image"] for s in samples]
        texts  = [s["title"] for s in samples]
        return images, texts

    # ---------------------------------------------------------------
    # Stage 1: Temporal Diversity
    # ---------------------------------------------------------------

    def _temporal_diverse_sampling(self, samples: list) -> list:
        """
        将时间轴等分为 num_shots 段，每段随机取 1 个样本。
        无时间戳样本先过滤；全部缺失则 raise ValueError。
        """
        timestamped = [s for s in samples if s.get("timestamp") is not None]
        if not timestamped:
            raise ValueError("No timestamped samples available.")

        sorted_s = sorted(timestamped, key=lambda x: x["timestamp"])
        n        = len(sorted_s)

        if n <= self.num_shots:
            return sorted_s

        seg_size = n / self.num_shots
        selected = []
        for i in range(self.num_shots):
            start = int(i * seg_size)
            end   = min(int((i + 1) * seg_size), n)
            if start >= end:
                start = min(start, n - 1)
                end   = start + 1
            idx = np.random.randint(start, end)
            selected.append(sorted_s[idx])

        return selected

    # ---------------------------------------------------------------
    # Stage 2: Semantic Diversity
    # ---------------------------------------------------------------

    def _semantic_diverse_sampling(self, samples: list) -> list:
        """
        按 third-level subtopic 分组，轮询各组各取一个，
        确保覆盖多个子话题。
        若全部无 subtopic 字段，直接返回原列表。

        subtopic 字段候选（按优先级）：
            'subtopic'  /  'sub_category'  /  'category_3'
        """
        if not samples:
            return samples

        def _subtopic(s):
            for k in ("subtopic", "sub_category", "category_3"):
                v = s.get(k)
                if v:
                    return str(v)
            return None

        topic_dict: Dict[str, list] = {}
        no_topic: list = []
        for s in samples:
            t = _subtopic(s)
            if t is None:
                no_topic.append(s)
            else:
                topic_dict.setdefault(t, []).append(s)

        if not topic_dict:
            return samples   # 无 subtopic 信息，不做过滤

        groups = list(topic_dict.values())
        for g in groups:
            np.random.shuffle(g)

        selected: list = []
        ptrs = [0] * len(groups)
        while len(selected) < self.num_shots:
            progress = False
            for gi, grp in enumerate(groups):
                if len(selected) >= self.num_shots:
                    break
                if ptrs[gi] < len(grp):
                    selected.append(grp[ptrs[gi]])
                    ptrs[gi] += 1
                    progress = True
            if not progress:
                break

        # 补充无 subtopic 样本
        if len(selected) < self.num_shots and no_topic:
            np.random.shuffle(no_topic)
            needed = self.num_shots - len(selected)
            selected.extend(no_topic[:needed])

        return selected

    # ---------------------------------------------------------------
    # Stage 3: User Diversity
    # ---------------------------------------------------------------

    def _user_diverse_sampling(self, samples: list) -> list:
        """
        按 user_id 分组，轮询各用户队列各取一个，
        避免单一用户风格/内容偏好主导原型。
        """
        if not samples:
            return samples

        user_dict: Dict[str, list] = {}
        for s in samples:
            uid = str(s.get("user_id", "unknown"))
            user_dict.setdefault(uid, []).append(s)

        for uid in user_dict:
            np.random.shuffle(user_dict[uid])

        selected: list = []
        queue = deque(list(user_dict.keys()))

        while len(selected) < self.num_shots and queue:
            uid = queue.popleft()
            if user_dict[uid]:
                selected.append(user_dict[uid].pop(0))
                if user_dict[uid]:     # 该用户还有剩余，放回尾部
                    queue.append(uid)

        return selected

    # ---------------------------------------------------------------
    # 持久化
    # ---------------------------------------------------------------

    def save_prototypes(
        self,
        visual_prototypes: torch.Tensor,
        textual_prototypes: torch.Tensor,
        save_path: str,
    ) -> None:
        torch.save(
            {
                "visual_prototypes":  visual_prototypes,
                "textual_prototypes": textual_prototypes,
                "num_classes": self.num_classes,
                "num_shots":   self.num_shots,
            },
            save_path,
        )
        print(f"[PrototypeBuilder] Saved → {save_path}  "
              f"visual={tuple(visual_prototypes.shape)}, "
              f"textual={tuple(textual_prototypes.shape)}")

    def load_prototypes(
        self, load_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ckpt = torch.load(load_path, map_location=self.device)
        return ckpt["visual_prototypes"], ckpt["textual_prototypes"]