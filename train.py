"""
Training Script  —  论文 Section 4.1.3

超参数（对齐论文值）：
    lr          = 1e-4   （论文）
    weight_decay = 1e-5  （论文）
    batch_size  = 128    （论文）
    epochs      = 4      （论文）

数据划分策略（无独立 val 文件时）：
    test_new 按 popularity 分层对半切分：
        前 50%  → val  （分类 early stopping + GBDT 权重选择）
        后 50%  → test （最终 SRC / MAE 报告，唯一一次使用）
    val 和 test 在训练阶段和 GBDT 阶段完全使用同一批索引。

GBDT 三段式评估：
    train（分层 77% 筛选）-> LightGBM + CatBoost fit（长尾加权）
    val（不筛选）         -> 网格搜索最优融合权重
    test（不筛选）        -> 最终 SRC / MAE 报告
"""

import os
import random
from collections import Counter
from pathlib import Path

import clip
import lightgbm as lgb
import numpy as np
import swanlab
import torch
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset           import JSONDataset
from prototype_builder import PrototypeBuilder
from smpp_loss         import SMPPLoss
from smpp_model        import SMPPModel, SMPPTrainer


# ======================================================================
# 工具函数
# ======================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


def split_files_exist(metadata_dir: str, split: str) -> bool:
    base = Path(metadata_dir)
    required = [
        base / f"{split}_img_filepath.txt",
        base / f"{split}_text.json",
        base / f"{split}_category.json",
    ]
    return all(p.exists() for p in required)


def resolve_split(metadata_dir: str, candidates, role: str, required: bool = True):
    for split in candidates:
        if split_files_exist(metadata_dir, split):
            return split
    if required:
        raise FileNotFoundError(
            f"No valid {role} split found in {metadata_dir}. Tried: {list(candidates)}"
        )
    return None


# ── 修改二：split_dataset_in_half 改为按 popularity 分层切分 ──────────
# 原来是按顺序对半切分（存在前后两半 popularity 分布不一致的问题）。
# 现在改为：按 popularity 分 n_bins 个桶，每桶内随机各取 50%，
# 确保 val 和 test 的 popularity 分布高度一致。
def split_dataset_in_half(dataset: JSONDataset, seed: int = 42, n_bins: int = 10):
    """
    按 popularity 分层对半切分。
    每个分位桶内随机各取 50%，保证 val/test 的 popularity 分布一致。
    奇数桶时 val 多一条。

    Returns:
        val_subset:  Subset（分类 early stopping + GBDT 权重选择）
        test_subset: Subset（最终指标，唯一一次使用）
    """
    n    = len(dataset)
    pops = np.array(dataset.popularities)
    rng  = np.random.default_rng(seed)

    bin_edges = np.percentile(pops, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6                          # 最大值归入最后一桶
    bin_ids = np.digitize(pops, bin_edges[1:-1])   # 每个样本所属桶 [0, n_bins-1]

    val_indices, test_indices = [], []
    for b in range(n_bins):
        bucket = np.where(bin_ids == b)[0].tolist()
        rng.shuffle(bucket)
        mid = (len(bucket) + 1) // 2              # 奇数时 val 多一条
        val_indices.extend(bucket[:mid])
        test_indices.extend(bucket[mid:])

    val_subset  = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # 打印分布统计，确认两半分布一致
    val_pops  = pops[val_indices]
    test_pops = pops[test_indices]
    print(
        f"[Split] {dataset.data_prefix} total={n} "
        f"→ val={len(val_indices)}  test={len(test_indices)}"
    )
    print(f"  Val  pop: mean={val_pops.mean():.3f}  std={val_pops.std():.3f}  "
          f"min={val_pops.min():.3f}  max={val_pops.max():.3f}")
    print(f"  Test pop: mean={test_pops.mean():.3f}  std={test_pops.std():.3f}  "
          f"min={test_pops.min():.3f}  max={test_pops.max():.3f}")
    return val_subset, test_subset
# ── 修改二结束 ───────────────────────────────────────────────────────


def prototype_file_is_usable(path: str, expected_num_classes: int) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        ckpt = torch.load(p, map_location="cpu")
        V    = ckpt.get("visual_prototypes")
        T    = ckpt.get("textual_prototypes")
    except Exception as e:
        print(f"[Prototype] Load failed: {e}")
        return False
    if not (isinstance(V, torch.Tensor) and isinstance(T, torch.Tensor)):
        return False
    if V.ndim != 2 or T.ndim != 2:
        return False
    if V.shape[0] != expected_num_classes or T.shape[0] != expected_num_classes:
        print(f"[Prototype] Class mismatch: expected {expected_num_classes}, "
              f"got visual={V.shape[0]}, textual={T.shape[0]}")
        return False
    if V.shape[1] != T.shape[1] or V.shape[1] == 0:
        return False
    return True


def create_model(
    num_classes, class_names, prototype_path, device,
    freeze_image=True, freeze_text=True, freeze_text_embedding=True,
):
    model = SMPPModel(
        num_classes=num_classes,
        clip_model_name="ViT-B/32",
        global_ctx_length=16,
        local_ctx_length=8,
        fusion_hidden_dim=512,
        fusion_num_heads=8,
        fusion_num_layers=2,
        class_names=class_names,
        device=device,
        freeze_image=freeze_image,
        freeze_text=freeze_text,
        freeze_text_embedding=freeze_text_embedding,
    )
    if prototype_path and Path(prototype_path).exists():
        ckpt = torch.load(prototype_path, map_location=device)
        model.load_prototypes(ckpt["visual_prototypes"], ckpt["textual_prototypes"])
        print(f"[Model] Prototypes loaded from {prototype_path}")
    else:
        print("[Model] WARNING: No prototypes loaded.")
    return model


def build_prototypes(dataset, num_classes, num_shots, save_path, device):
    print(f"[Prototype] Building {num_classes}-class {num_shots}-shot prototypes ...")
    builder = PrototypeBuilder(num_classes=num_classes, num_shots=num_shots, device=device)
    image_samples, text_samples = {}, {}
    for class_id in range(num_classes):
        print(f"  Sampling class {class_id}/{num_classes}", end="\r")
        imgs, txts = builder.sample_diverse_data(
            dataset, class_id, sampling_strategy="diverse"
        )
        image_samples[class_id] = imgs
        text_samples[class_id]  = txts
    V = builder.build_visual_prototypes(image_samples)
    T = builder.build_textual_prototypes(text_samples)
    builder.save_prototypes(V, T, save_path)
    return V, T


# ======================================================================
# 特征提取
# ======================================================================

def extract_features_for_gbdt(model, dataloader, save_path, device):
    print(f"[GBDT] Extracting features -> {save_path}")
    model.eval()
    all_features, all_labels, all_popularities = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 5:
                images, text_tokens, labels, popularities, user_feats = batch
                user_feats = user_feats.to(device)
            else:
                images, text_tokens, labels, popularities = batch
                user_feats = None

            images      = images.to(device)
            text_tokens = text_tokens.to(device)

            if isinstance(model, torch.nn.DataParallel):
                feats = model.module.extract_features_for_regression(
                    images, text_tokens, user_feats)
            else:
                feats = model.extract_features_for_regression(
                    images, text_tokens, user_feats)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_popularities.append(popularities.numpy())

    X = np.concatenate(all_features,     axis=0)
    y = np.concatenate(all_popularities, axis=0)
    l = np.concatenate(all_labels,       axis=0)
    np.savez(save_path, features=X, labels=l, popularities=y)
    print(f"[GBDT] Feature shape: {X.shape}")
    return X, l, y


# ======================================================================
# Sample Selection（论文 Section 4.2，改为分层筛选）
# ======================================================================

# ── 修改三：filter_samples_by_loss 改为按 popularity 分层筛选 ────────
# 原来是全局按 loss 排序取前 77%，会系统性地把长尾高 popularity
# 样本过滤掉（因为它们 loss 普遍偏高），导致 GBDT 无法学习长尾。
# 现在改为：按 popularity 分桶，每桶内独立保留低 loss 的 77%，
# 确保每个 popularity 区间都有代表性样本保留。
def filter_samples_by_loss(
    model, dataloader, keep_ratio=0.77, device="cuda",
    stratify_by_popularity=True, n_bins=10,
):
    """
    按分类损失筛选训练样本。

    stratify_by_popularity=True（推荐）：
        按 popularity 分 n_bins 个桶，每桶内独立保留低 loss 的 keep_ratio。
        长尾样本（高 popularity）不会被整体过滤。
    stratify_by_popularity=False：
        全局按 loss 排序保留前 keep_ratio（原始论文方案）。
    """
    print(f"[Sample Selection] Filtering {keep_ratio*100:.0f}% low-loss samples "
          f"(stratify={stratify_by_popularity}) ...")
    model.eval()
    losses, popularities, indices = [], [], []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if len(batch) >= 4:
                images, text_tokens, labels, pops, *_ = batch
            else:
                images, text_tokens, labels = batch
                pops = torch.zeros(images.shape[0])

            actual_bs   = images.shape[0]
            images      = images.to(device)
            text_tokens = text_tokens.to(device)
            labels      = labels.to(device)

            out    = model(images, text_tokens, labels)
            logits = (out["logits_global"] + out["logits_local"]
                      + out["logits_visual"]) / 3

            losses.append(criterion(logits, labels).cpu().numpy())
            popularities.append(pops.numpy())
            start = batch_idx * dataloader.batch_size
            indices.extend(range(start, start + actual_bs))

    all_losses = np.concatenate(losses)
    all_pops   = np.concatenate(popularities)
    assert len(all_losses) == len(indices), (
        f"Loss count {len(all_losses)} != index count {len(indices)}"
    )

    if not stratify_by_popularity:
        # 原始论文方案：全局排序
        keep_n       = int(len(all_losses) * keep_ratio)
        keep_indices = np.argsort(all_losses)[:keep_n].tolist()
    else:
        # 分层方案
        bin_edges = np.percentile(all_pops, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-6
        bin_ids = np.digitize(all_pops, bin_edges[1:-1])

        keep_indices = []
        for b in range(n_bins):
            bucket_mask   = np.where(bin_ids == b)[0]
            if len(bucket_mask) == 0:
                continue
            bucket_losses = all_losses[bucket_mask]
            keep_n        = max(1, int(len(bucket_mask) * keep_ratio))
            local_sorted  = np.argsort(bucket_losses)[:keep_n]
            keep_indices.extend(bucket_mask[local_sorted].tolist())
            pop_lo = bin_edges[b]
            pop_hi = bin_edges[b + 1]
            print(f"  Bucket {b:2d} pop=[{pop_lo:.1f},{pop_hi:.1f}): "
                  f"{keep_n}/{len(bucket_mask)} kept")

    print(f"[Sample Selection] Kept {len(keep_indices)}/{len(all_losses)} samples")
    return keep_indices
# ── 修改三结束 ───────────────────────────────────────────────────────


# ======================================================================
# GBDT 训练与评估
# ======================================================================

def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    src = spearmanr(y_true, y_pred).correlation
    if src is None or np.isnan(src):
        src = 0.0
    return {"mse": float(mse), "mae": float(mae), "spearman": float(src)}


def fusion_model(lgb_preds, cat_preds, w_lgb=0.5, w_cat=0.5):
    total = w_lgb + w_cat
    return (w_lgb / total) * lgb_preds + (w_cat / total) * cat_preds


def train_gbdt_and_evaluate(
    model, train_dataset, val_loader, test_loader,
    device="cuda", random_state=42,
):
    """
    三段式评估：
        train（分层 77% 筛选）-> GBDT fit（长尾加权）
        val（不筛选）         -> 选最优融合权重
        test（不筛选）        -> 最终指标（唯一一次）
    """
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    keep_indices = filter_samples_by_loss(
        model, train_loader, keep_ratio=0.77,
        device=device, stratify_by_popularity=True,   # ← 分层筛选
    )
    train_subset = Subset(train_dataset, keep_indices)

    X_train, _, y_train = extract_features_for_gbdt(
        model,
        DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0),
        "features_train_filtered.npz", device,
    )
    X_val,  _, y_val  = extract_features_for_gbdt(
        model, val_loader,  "features_val.npz",  device)
    X_test, _, y_test = extract_features_for_gbdt(
        model, test_loader, "features_test.npz", device)

    # ── 修改四：GBDT 训练时对长尾样本（popularity > 10）加权 ─────────
    # 原来所有样本权重相同，模型在 popularity > 10 的长尾上预测偏差大。
    # 现在对长尾样本设置 3 倍权重，引导 GBDT 更关注该区间的拟合。
    sample_weight = np.where(y_train > 10, 3.0, 1.0)
    print(f"[GBDT] Tail samples (pop>10): {(y_train > 10).sum()} / {len(y_train)}, "
          f"weight=3.0")
    # ── 修改四结束 ───────────────────────────────────────────────────

    # LightGBM
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05,
        max_depth=-1, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state,
        min_child_samples=20,    # 防止长尾样本过拟合
    )
    lgb_reg.fit(X_train, y_train, sample_weight=sample_weight)  # ← 修改四：加 sample_weight

    # CatBoost
    cat_reg = CatBoostRegressor(
        iterations=500, learning_rate=0.05,
        depth=6, random_seed=random_state, verbose=0,
    )
    cat_reg.fit(X_train, y_train, sample_weight=sample_weight)  # ← 修改四：加 sample_weight

    # 验证集搜索最优融合权重
    lgb_val = lgb_reg.predict(X_val)
    cat_val = cat_reg.predict(X_val)
    best_w, best_mse = 0.5, float("inf")
    best_val_metrics = None
    for w in np.linspace(0.0, 1.0, 11):
        fused = fusion_model(lgb_val, cat_val, w_lgb=float(w), w_cat=float(1 - w))
        m     = _metrics(y_val, fused)
        if m["mse"] < best_mse:
            best_mse         = m["mse"]
            best_w           = float(w)
            best_val_metrics = m

    # 测试集最终评估（唯一一次）
    lgb_test   = lgb_reg.predict(X_test)
    cat_test   = cat_reg.predict(X_test)
    fused_test = fusion_model(lgb_test, cat_test, w_lgb=best_w, w_cat=1 - best_w)
    test_metrics = _metrics(y_test, fused_test)

    return {
        "best_weight_lightgbm": best_w,
        "val":  best_val_metrics,
        "test": test_metrics,
    }


# ======================================================================
# main
# ======================================================================

def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    seed          = int(os.getenv("SEED", "42"))
    batch_size    = 128
    num_epochs    = 10
    learning_rate = 1e-4
    weight_decay  = 1e-5
    early_stop_patience = 3

    hashtag_min_freq   = int(os.getenv("HASHTAG_MIN_FREQ", "5"))
    hashtag_top_k      = int(os.getenv("HASHTAG_TOP_K", "300"))
    hashtag_other_name = os.getenv("HASHTAG_OTHER_NAME", "other")

    train_metadata_dir = "/mnt/sda/data/fame_split_37"
    test_metadata_dir  = "/mnt/sda/data/fame_split_37"
    image_dir          = "/mnt/sda/data/fame_split_37"

    train_split = resolve_split(
        train_metadata_dir, candidates=["train_new", "train"],
        role="train", required=True,
    )
    test_split = resolve_split(
        test_metadata_dir, candidates=["test_new", "test"],
        role="test", required=True,
    )
    print(f"[Data] train={train_split}  test(raw)={test_split}")

    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    swanlab_id = os.getenv("SWANLAB_ID", "").strip()
    swanlab_init_kwargs = {
        "project": os.getenv("SWANLAB_PROJECT", "SMPP"),
        "resume":  os.getenv("SWANLAB_RESUME",  "allow"),
        "config": {
            "seed": seed, "batch_size": batch_size,
            "num_epochs": num_epochs, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "early_stop_patience": early_stop_patience, "device": device,
        },
    }
    if len(swanlab_id) == 21 and swanlab_id.isalnum() and swanlab_id == swanlab_id.lower():
        swanlab_init_kwargs["id"] = swanlab_id
    elif swanlab_id:
        print("[SwanLab] Ignore invalid SWANLAB_ID.")
    swanlab.init(**swanlab_init_kwargs)

    _, preprocess = clip.load("ViT-B/32", device=device)

    common_kw = dict(
        image_dir=image_dir,
        clip_preprocess=preprocess,
        hashtag_min_freq=hashtag_min_freq,
        hashtag_top_k=hashtag_top_k,
        hashtag_other_name=hashtag_other_name,
        include_user_features=False,
    )

    train_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        split=train_split, is_training=True, **common_kw,
    )

    full_test_dataset = JSONDataset(
        metadata_dir=test_metadata_dir,
        split=test_split, is_training=False,
        class_mapping=train_dataset.class_mapping, **common_kw,
    )
    # 使用分层切分（修改二）
    val_subset, test_subset = split_dataset_in_half(full_test_dataset, seed=seed)

    try:
        train_dataset.print_statistics()
        train_dataset.plot_distribution()
    except Exception as e:
        print(f"[Warning] Statistics failed: {e}")

    class_names = [
        k for k, v in
        sorted(train_dataset.class_mapping.items(), key=lambda x: x[1])
    ]
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    proto_path = f"prototypes_{train_split}_{num_classes}cls.pth"
    if not prototype_file_is_usable(proto_path, num_classes):
        build_prototypes(train_dataset, num_classes, 256, proto_path, device)

    model = create_model(
        num_classes=num_classes, class_names=class_names,
        prototype_path=proto_path, device=device,
        freeze_image=True, freeze_text=True, freeze_text_embedding=True,
    )
    total_p     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_p:.2f}M  Trainable: {trainable_p:.2f}M")

    # ── 修改五：计算类别权重并传入 SMPPLoss（解决类别不均衡问题）──────
    # 原来三个损失的 CrossEntropyLoss 未设置 weight，小类（如 class1/3）
    # 的学习信号被大类（class8，53k 条）淹没。
    # 现在用反频率权重：类越少权重越大，让模型对小类给予更多关注。
    label_counts = Counter(
        train_dataset.class_mapping[cat["Category"]]
        for cat in train_dataset.categories
        if isinstance(cat, dict)
        and cat.get("Category") in train_dataset.class_mapping
    )
    total_samples = sum(label_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * label_counts.get(i, 1))
         for i in range(num_classes)],
        dtype=torch.float32, device=device,
    )
    print("[Class Weights]", {i: f"{w:.2f}" for i, w in enumerate(class_weights.tolist())})

    model.criterion = SMPPLoss(
        temperature=0.07,
        spatial_temperature=0.1,
        temperature_visual=0.07,
        temperature_textual=0.07,
        class_weights=class_weights,   # ← 新增，传入三个子损失的 CE
    )
    # ── 修改五结束 ───────────────────────────────────────────────────

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )
    trainer = SMPPTrainer(model, optimizer, device=device)

    ckpt_dir   = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_files = list(ckpt_dir.glob("epoch_*.pth"))
    start_epoch = 0
    if ckpt_files:
        def _epoch_n(p):
            try:
                return int(p.stem.split("_")[-1])
            except Exception:
                return -1
        latest = max(ckpt_files, key=_epoch_n)
        if _epoch_n(latest) >= 0:
            last_epoch, _ = trainer.load_checkpoint(str(latest))
            start_epoch   = last_epoch + 1
            print(f"Resuming at epoch {start_epoch}")

    best_val_loss = float("inf")
    no_improve    = 0
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc     = trainer.validate(val_loader)
        scheduler.step()

        swanlab.log({
            "epoch": epoch,
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss":   val_loss,   "val/acc":   val_acc,
        })

        trainer.save_checkpoint(f"checkpoints/epoch_{epoch}.pth", epoch, val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("\nRunning GBDT regression ...")
    gbdt_train = JSONDataset(
        metadata_dir=train_metadata_dir,
        split=train_split, is_training=False,
        class_mapping=train_dataset.class_mapping, **common_kw,
    )

    result = train_gbdt_and_evaluate(
        model=model,
        train_dataset=gbdt_train,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    vm = result["val"]
    tm = result["test"]
    print(
        f"GBDT Val  — MSE={vm['mse']:.4f}  MAE={vm['mae']:.4f}  "
        f"SRC={vm['spearman']:.4f}  "
        f"best_lgb_weight={result['best_weight_lightgbm']:.2f}"
    )
    print(
        f"GBDT Test — MSE={tm['mse']:.4f}  MAE={tm['mae']:.4f}  "
        f"SRC={tm['spearman']:.4f}"
    )

    swanlab.log({
        "gbdt/best_weight_lightgbm": result["best_weight_lightgbm"],
        "gbdt/val_mse":  vm["mse"],  "gbdt/val_mae":  vm["mae"],
        "gbdt/val_src":  vm["spearman"],
        "gbdt/test_mse": tm["mse"],  "gbdt/test_mae": tm["mae"],
        "gbdt/test_src": tm["spearman"],
    })
    print("\nAll done!")


if __name__ == "__main__":
    main()