"""
Training Script  —  论文 Section 4.1.3
ssh_prompt(cc)结果为GBDT Test — MSE=4.6540  MAE=1.6973  SRC=0.5230
超参数（对齐论文值）：
    lr          = 1e-4   （论文）
    weight_decay = 1e-5  （论文）
    batch_size  = 128    （论文）
    epochs      = 4      （论文）

数据划分策略（无独立 val 文件时）：
    test_new 按顺序对半切分：
        前 50%  → val  （分类 early stopping + GBDT 权重选择）
        后 50%  → test （最终 SRC / MAE 报告，唯一一次使用）
    val 和 test 在训练阶段和 GBDT 阶段完全使用同一批索引。

GBDT 三段式评估：
    train（77% 筛选）-> LightGBM + CatBoost fit
    val（不筛选）    -> 网格搜索最优融合权重
    test（不筛选）   -> 最终 SRC / MAE 报告
"""

import os
import random
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
    """检查某个 split 的核心元数据文件是否存在。"""
    base = Path(metadata_dir)
    required = [
        base / f"{split}_img_filepath.txt",
        base / f"{split}_text.json",
        base / f"{split}_category.json",
    ]
    return all(p.exists() for p in required)


def resolve_split(metadata_dir: str, candidates, role: str, required: bool = True):
    """从候选 split 名中选第一个实际存在的。"""
    for split in candidates:
        if split_files_exist(metadata_dir, split):
            return split
    if required:
        raise FileNotFoundError(
            f"No valid {role} split found in {metadata_dir}. Tried: {list(candidates)}"
        )
    return None


def split_dataset_in_half(dataset: JSONDataset, seed: int = 42):
    """
    将 dataset 按原始顺序对半切分，返回两个规模相等（或相差 1）的 Subset。

    切分规则：
        奇数时前半多一条，|val| >= |test|。
        不打乱顺序，保证可复现，且与数据的时序关系保持一致。

    Returns:
        val_subset:  前 50% 的 Subset（分类 early stopping + GBDT 权重选择）
        test_subset: 后 50% 的 Subset（最终指标报告，唯一一次使用）
    """
    n   = len(dataset)
    mid = (n + 1) // 2                          # 奇数时前半多一条
    val_indices  = list(range(0,   mid))
    test_indices = list(range(mid, n  ))
    val_subset   = Subset(dataset, val_indices)
    test_subset  = Subset(dataset, test_indices)
    print(
        f"[Split] {dataset.data_prefix} total={n} "
        f"→ val={len(val_subset)}  test={len(test_subset)}"
    )
    return val_subset, test_subset


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
    num_classes,
    class_names,
    prototype_path,
    device,
    freeze_image=True,
    freeze_text=True,
    freeze_text_embedding=True,
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
        model.load_prototypes(
            ckpt["visual_prototypes"], ckpt["textual_prototypes"]
        )
        print(f"[Model] Prototypes loaded from {prototype_path}")
    else:
        print("[Model] WARNING: No prototypes loaded.")
    return model


def build_prototypes(dataset, num_classes, num_shots, save_path, device):
    print(f"[Prototype] Building {num_classes}-class {num_shots}-shot prototypes ...")
    builder = PrototypeBuilder(
        num_classes=num_classes, num_shots=num_shots, device=device
    )
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
                    images, text_tokens, user_feats
                )
            else:
                feats = model.extract_features_for_regression(
                    images, text_tokens, user_feats
                )

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
# Sample Selection（论文 Section 4.2）
# ======================================================================

def filter_samples_by_loss(model, dataloader, keep_ratio=0.77, device="cuda"):
    """
    按分类损失从小到大排序，保留前 keep_ratio 的样本。
    索引计算使用 images.shape[0] 避免末尾 batch 越界。
    """
    print(f"[Sample Selection] Filtering {keep_ratio*100:.0f}% low-loss samples ...")
    model.eval()
    losses  = []
    indices = []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if len(batch) >= 4:
                images, text_tokens, labels, *_ = batch
            else:
                images, text_tokens, labels = batch

            actual_bs   = images.shape[0]          # 真实 batch 大小
            images      = images.to(device)
            text_tokens = text_tokens.to(device)
            labels      = labels.to(device)

            out    = model(images, text_tokens, labels)
            logits = (
                out["logits_global"] + out["logits_local"] + out["logits_visual"]
            ) / 3

            batch_loss = criterion(logits, labels).cpu().numpy()
            losses.append(batch_loss)

            start = batch_idx * dataloader.batch_size
            indices.extend(range(start, start + actual_bs))

    all_losses = np.concatenate(losses)
    assert len(all_losses) == len(indices), (
        f"Loss count {len(all_losses)} != index count {len(indices)}"
    )

    threshold_n  = int(len(all_losses) * keep_ratio)
    sorted_idx   = np.argsort(all_losses)
    keep_indices = sorted_idx[:threshold_n].tolist()
    print(f"[Sample Selection] Kept {len(keep_indices)}/{len(all_losses)} samples")
    return keep_indices


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
    model,
    train_dataset,
    val_loader,        # 直接传 DataLoader，与训练阶段共享同一个对象
    test_loader,       # 直接传 DataLoader，与训练阶段共享同一个对象
    device="cuda",
    random_state=42,
):
    """
    三段式评估：
        train（77% 筛选）-> GBDT fit
        val（不筛选）    -> 选最优融合权重
        test（不筛选）   -> 最终指标（唯一一次）

    val_loader / test_loader 直接复用 main() 中已构建的 DataLoader，
    确保两个阶段看到的样本集合完全相同。
    """
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    # 筛选训练样本
    keep_indices = filter_samples_by_loss(
        model, train_loader, keep_ratio=0.77, device=device
    )
    train_subset = Subset(train_dataset, keep_indices)

    X_train, _, y_train = extract_features_for_gbdt(
        model,
        DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0),
        "features_train_filtered.npz",
        device,
    )
    X_val,  _, y_val  = extract_features_for_gbdt(
        model, val_loader,  "features_val.npz",  device
    )
    X_test, _, y_test = extract_features_for_gbdt(
        model, test_loader, "features_test.npz", device
    )

    # LightGBM
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=-1,     num_leaves=63,
        subsample=0.8,    colsample_bytree=0.8,
        random_state=random_state,
    )
    lgb_reg.fit(X_train, y_train)

    # CatBoost
    cat_reg = CatBoostRegressor(
        iterations=300, learning_rate=0.05,
        depth=6, random_seed=random_state, verbose=0,
    )
    cat_reg.fit(X_train, y_train)

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

    # ---- 超参数（对齐论文 Section 4.1.3）----
    seed          = int(os.getenv("SEED", "42"))
    batch_size    = 64
    num_epochs    = 10
    learning_rate = 1e-4
    weight_decay  = 1e-5
    early_stop_patience = 4

    hashtag_min_freq   = int(os.getenv("HASHTAG_MIN_FREQ", "5"))
    hashtag_top_k      = int(os.getenv("HASHTAG_TOP_K", "300"))
    hashtag_other_name = os.getenv("HASHTAG_OTHER_NAME", "other")

    # ---- 数据路径 ----
    train_metadata_dir = "/mnt/sda/data/fame_split_37"
    test_metadata_dir  = "/mnt/sda/data/fame_split_37"
    image_dir          = "/mnt/sda/data/fame_split_37"

    # 自动匹配 split 前缀
    train_split = resolve_split(
        train_metadata_dir,
        candidates=["train_new", "train"],
        role="train", required=True,
    )
    # test_split 只用于加载原始完整 test 数据，后续再对半切分
    test_split = resolve_split(
        test_metadata_dir,
        candidates=["test_new", "test"],
        role="test", required=True,
    )
    print(f"[Data] train={train_split}  test(raw)={test_split}")

    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # SwanLab
    swanlab_id = os.getenv("SWANLAB_ID", "jas27jjeblqkjdpwlohgh").strip()
    swanlab_init_kwargs = {
        "project": os.getenv("SWANLAB_PROJECT", "SMPP"),
        "resume":  os.getenv("SWANLAB_RESUME",  "allow"),
        "config": {
            "seed": seed, "batch_size": batch_size,
            "num_epochs": num_epochs, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "device": device,
        },
    }
    if len(swanlab_id) == 21 and swanlab_id.isalnum() and swanlab_id == swanlab_id.lower():
        swanlab_init_kwargs["id"] = swanlab_id
    elif swanlab_id:
        print("[SwanLab] Ignore invalid SWANLAB_ID.")
    swanlab.init(**swanlab_init_kwargs)

    _, preprocess = clip.load("ViT-B/32", device=device)

    # ---- Step 0: 加载数据集 ----
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
        split=train_split, is_training=True,
        **common_kw,
    )

    # ── 核心改动：加载完整 test 数据集，然后对半切分 ─────────────────
    # 1. 加载完整 test_new（关闭增广，class_mapping 与 train 一致）
    full_test_dataset = JSONDataset(
        metadata_dir=test_metadata_dir,
        split=test_split, is_training=False,
        class_mapping=train_dataset.class_mapping,
        **common_kw,
    )
    # 2. 按原始顺序对半切分，产生固定的 val_subset / test_subset
    #    此后所有阶段（分类训练 early stopping、GBDT 特征提取）
    #    均复用这两个 Subset，保证样本集合完全一致
    val_subset, test_subset = split_dataset_in_half(full_test_dataset, seed=seed)
    # ─────────────────────────────────────────────────────────────────

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
    # val_loader / test_loader 在此处唯一构建，后续所有阶段均复用
    val_loader = DataLoader(
        val_subset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # ---- Step 1: 构建原型（仅使用 train_dataset，完全不接触 val/test）----
    proto_path = f"prototypes_{train_split}_{num_classes}cls.pth"
    if not prototype_file_is_usable(proto_path, num_classes):
        build_prototypes(train_dataset, num_classes, 256, proto_path, device)

    # ---- Step 2: 创建模型 ----
    model = create_model(
        num_classes=num_classes,
        class_names=class_names,
        prototype_path=proto_path,
        device=device,
        freeze_image=True,
        freeze_text=True,
        freeze_text_embedding=True,
    )
    total_p     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_p:.2f}M  Trainable: {trainable_p:.2f}M")

    # ---- Step 3: 分类训练（early stopping 使用 val_loader）----
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
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
        val_loss, val_acc     = trainer.validate(val_loader)   # ← val 前 50%

        swanlab.log({
            "epoch": epoch,
            "train/loss": train_loss, "train/acc": train_acc,
            "val/loss":   val_loss,   "val/acc":   val_acc,
        })

        trainer.save_checkpoint(
            f"checkpoints/epoch_{epoch}.pth", epoch, val_loss
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ---- Step 4: GBDT 三段式评估 ----
    # gbdt_train 需关闭数据增广（is_training=False）
    print("\nRunning GBDT regression ...")
    gbdt_train = JSONDataset(
        metadata_dir=train_metadata_dir,
        split=train_split, is_training=False,
        class_mapping=train_dataset.class_mapping,
        **common_kw,
    )

    result = train_gbdt_and_evaluate(
        model=model,
        train_dataset=gbdt_train,
        val_loader=val_loader,     # ← 复用同一个 val_loader（test_new 前 50%）
        test_loader=test_loader,   # ← 复用同一个 test_loader（test_new 后 50%）
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
