"""
Training Script for SMPP Model
完整的训练流程示例
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import clip
from tqdm import tqdm
import numpy as np
from pathlib import Path
import swanlab
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import random

from smpp_model import SMPPModel, SMPPTrainer
from prototype_builder import PrototypeBuilder
from dataset import JSONDataset

def create_model(
    num_classes: int = 77,
    class_names: list = None,
    prototype_path: str = None,
    device: str = "cuda",
    freeze_image: bool = True,
    freeze_text: bool = True,
    freeze_text_embedding: bool = True
):
    """
    创建并初始化 SMPP 模型
    
    Args:
        num_classes: 类别数量
        class_names: 类别名称列表
        prototype_path: 预先计算的原型文件路径
        device: 计算设备
        freeze_image: 是否冻结图像编码器
        freeze_text: 是否冻结文本编码器
        freeze_text_embedding: 是否冻结文本token embedding
    
    Returns:
        model: 初始化好的 SMPP 模型
    """
    # 创建模型
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
        freeze_text_embedding=freeze_text_embedding
    )
    
    # 加载原型 (如果有)
    if prototype_path and Path(prototype_path).exists():
        print(f"Loading prototypes from {prototype_path}")
        checkpoint = torch.load(prototype_path, map_location=device)
        visual_prototypes = checkpoint['visual_prototypes']
        textual_prototypes = checkpoint['textual_prototypes']
        model.load_prototypes(visual_prototypes, textual_prototypes)
    else:
        print("Warning: No prototypes loaded. Please build prototypes first!")
    
    return model


def build_prototypes(
    dataset,
    num_classes: int = 77,
    num_shots: int = 256,
    save_path: str = "prototypes.pth",
    device: str = "cuda"
):
    """
    构建原型并保存，使用多样性采样策略。

    Args:
        dataset: 包含所有训练数据的数据集
        num_classes: 类别数量
        num_shots: 每类采样数量
        save_path: 保存路径
        device: 计算设备
    """
    print("Building prototypes with random sampling...")

    builder = PrototypeBuilder(
        num_classes=num_classes,
        num_shots=num_shots,
        device=device
    )

    image_samples = {}
    text_samples = {}

    for class_id in range(num_classes):
        print(f"Sampling class {class_id}/{num_classes}")

        # 使用 sample_diverse_data 方法采样数据
        images, texts = builder.sample_diverse_data(dataset, class_id)
        image_samples[class_id] = images
        text_samples[class_id] = texts

    visual_prototypes = builder.build_visual_prototypes(image_samples)
    textual_prototypes = builder.build_textual_prototypes(text_samples)

    builder.save_prototypes(visual_prototypes, textual_prototypes, save_path)
    print(f"Prototypes saved to {save_path}")

    return visual_prototypes, textual_prototypes


def train(
    model: SMPPModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    save_dir: str = "./checkpoints",
    device: str = "cuda"
):
    """
    训练模型
    
    Args:
        model: SMPP 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        save_dir: 模型保存目录
        device: 计算设备
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # 创建训练器
    trainer = SMPPTrainer(model, optimizer, device)
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        
        # 验证
        val_loss, val_acc = trainer.validate(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(save_dir) / f"best_model_epoch{epoch}.pth"
            trainer.save_checkpoint(save_path, epoch, val_loss)
            print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
        
        # 定期保存检查点
        if epoch % 1 == 0:
            save_path = Path(save_dir) / f"checkpoint_epoch{epoch}.pth"
            trainer.save_checkpoint(save_path, epoch, val_loss)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


def extract_features_for_gbdt(
    model: SMPPModel,
    dataloader: DataLoader,
    save_path: str,
    device: str = "cuda"
):
    """
    提取特征用于 GBDT 回归
    
    Args:
        model: 训练好的 SMPP 模型
        dataloader: 数据加载器
        save_path: 特征保存路径
        device: 计算设备
    """
    print("Extracting features for GBDT regression...")
    
    model.eval()
    all_features = []
    all_labels = []
    all_popularities = []  # 真实流行度数值
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, text_tokens, labels, popularities = batch
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            
            # 提取特征
            # 如果有用户特征，也可以传入
            if isinstance(model, torch.nn.DataParallel):
                features = model.module.extract_features_for_regression(images, text_tokens)
            else:
                features = model.extract_features_for_regression(images, text_tokens)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_popularities.append(popularities.cpu().numpy())
    
    # 合并所有特征
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_popularities = np.concatenate(all_popularities, axis=0)
    
    # 保存
    np.savez(
        save_path,
        features=all_features,
        labels=all_labels,
        popularities=all_popularities
    )
    
    print(f"Features saved to {save_path}")
    print(f"Feature shape: {all_features.shape}")
    
    return all_features, all_labels, all_popularities


def train_gbdt_and_evaluate(
    model: SMPPModel,
    train_dataset,
    val_dataset,
    test_dataset,
    device: str = "cuda",
    random_state: int = 42
):
    """
    使用提取特征训练 GBDT，并在 train/val/test 上进行严格评估。

    说明:
    - train: 训练 GBDT 参数
    - val: 从训练集划分得到，用于模型选择/调参
    - test: 独立测试集，仅用于最终报告
    """
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # 提取特征
    X_train, _, y_train = extract_features_for_gbdt(
        model=model,
        dataloader=train_loader,
        save_path="features_train.npz",
        device=device
    )
    X_val, _, y_val = extract_features_for_gbdt(
        model=model,
        dataloader=val_loader,
        save_path="features_val.npz",
        device=device
    )
    X_test, _, y_test = extract_features_for_gbdt(
        model=model,
        dataloader=test_loader,
        save_path="features_test.npz",
        device=device
    )

    # 训练 LightGBM
    reg = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    reg.fit(X_train, y_train)

    # 在验证集评估（用于模型选择）
    preds_val = reg.predict(X_val)
    val_mse = mean_squared_error(y_val, preds_val)
    val_mae = mean_absolute_error(y_val, preds_val)
    val_spearman = spearmanr(y_val, preds_val).correlation

    # 在测试集评估（最终结果）
    preds_test = reg.predict(X_test)
    test_mse = mean_squared_error(y_test, preds_test)
    test_mae = mean_absolute_error(y_test, preds_test)
    test_spearman = spearmanr(y_test, preds_test).correlation

    # Build a compact test evaluation curve (sorted by true popularity).
    order = np.argsort(y_test)
    y_test_sorted = y_test[order]
    preds_test_sorted = preds_test[order]
    max_curve_points = int(os.getenv("GBDT_TEST_CURVE_POINTS", "200"))
    max_curve_points = max(2, min(max_curve_points, len(y_test_sorted)))
    sampled_idx = np.linspace(0, len(y_test_sorted) - 1, num=max_curve_points, dtype=int)
    curve_points = []
    for rank, idx in enumerate(sampled_idx):
        y_true_point = float(y_test_sorted[idx])
        y_pred_point = float(preds_test_sorted[idx])
        curve_points.append({
            "rank": rank,
            "sample_index": int(idx),
            "y_true": y_true_point,
            "y_pred": y_pred_point,
            "abs_error": float(abs(y_pred_point - y_true_point)),
        })

    return {
        "val": {
            "mse": val_mse,
            "mae": val_mae,
            "spearman": val_spearman,
        },
        "test": {
            "mse": test_mse,
            "mae": test_mae,
            "spearman": test_spearman,
        },
        "test_curve": curve_points,
    }


def main():
    """
    主函数: 完整的训练流程
    """
    # 配置
    mp_start_method = os.getenv("MP_START_METHOD", "").strip().lower()
    if mp_start_method:
        try:
            torch.multiprocessing.set_start_method(mp_start_method, force=True)
            print(f"Using multiprocessing start method from MP_START_METHOD: {mp_start_method}")
        except RuntimeError:
            pass
    elif os.name == "nt":
        # Windows requires spawn; on Linux/macOS keep default to avoid resource_tracker warnings.
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
            print("Using multiprocessing start method: spawn (Windows default)")
        except RuntimeError:
            pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = int(os.getenv("SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_classes = 77
    batch_size = 64
    num_epochs = 8
    learning_rate = 1e-5
    weight_decay = 5e-4
    early_stop_patience = 4
    train_split = "train_new"
    test_split = "test_new"
    val_ratio = float(os.getenv("VAL_RATIO", "0.1"))
    
    # Paths
    # Update dataset paths to the specified locations
    train_metadata_dir = "/mnt/sda/data/fame_split_37"
    test_metadata_dir = "/mnt/sda/data/fame_split_37"
    image_dir = "/mnt/sda/data/fame_split_37"
    
    print(f"Using device: {device}")

    # Initialize SwanLab
    swanlab_resume = os.getenv("SWANLAB_RESUME", "allow")
    swanlab_id = os.getenv("SWANLAB_ID", "ak74w5nde5rfmrh7vtaca")
    swanlab.init(
        project="SMPP",
        resume=swanlab_resume,
        id=swanlab_id if swanlab_id else None,
        config={
            "seed": seed,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stop_patience": early_stop_patience,
            "val_ratio": val_ratio,
            "device": device
        }
    )
    
    # Prepare Preprocessing
    _, preprocess = clip.load("ViT-B/32", device=device)
    
    # Step 0: Load Datasets
    print("Loading datasets...")
    # Training set (full train split)
    train_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split=train_split,
        is_training=True
    )

    # Validation view over the same split (without augmentation), then perform index split.
    val_view_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split=train_split,
        is_training=False,
        class_mapping=train_dataset.class_mapping,
    )
    
    # 类别名称
    class_names = [k for k, v in sorted(train_dataset.class_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes.")

    all_indices = np.arange(len(train_dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    val_count = max(1, int(len(all_indices) * val_ratio))
    if val_count >= len(all_indices):
        val_count = max(1, len(all_indices) - 1)
    val_indices = all_indices[:val_count]
    train_indices = all_indices[val_count:]

    print(f"Train/Val split (from {train_split}): train={len(train_indices)}, val={len(val_indices)}")

    # Enforce: training stage must only use train split with an internal val split.
    overlap = np.intersect1d(train_indices, val_indices)
    if overlap.size != 0:
        raise RuntimeError("Train/Val split overlap detected in training stage.")
    print(
        f"Training protocol: train and val are both from {train_split}; "
        f"test split {test_split} is reserved for GBDT final evaluation."
    )

    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(val_view_dataset, val_indices.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Step 1: 构建原型 (如果还没有)
    prototype_path = f"prototypes_{train_split}_{num_classes}cls.pth"
    if not Path(prototype_path).exists():
        # Build prototypes using train-only indices to avoid validation leakage.
        original_class_indices = train_dataset.class_indices
        train_index_set = set(train_indices.tolist())
        train_dataset.class_indices = {
            class_id: [idx for idx in idxs if idx in train_index_set]
            for class_id, idxs in original_class_indices.items()
        }
        build_prototypes(
            dataset=train_dataset,
            num_classes=num_classes,
            num_shots=256,
            save_path=prototype_path,
            device=device
        )
        train_dataset.class_indices = original_class_indices
    
    # Step 2: 创建模型
    print("\nCreating model...")
    model = create_model(
        num_classes=num_classes,
        class_names=class_names,
        prototype_path=prototype_path,
        device=device,
        freeze_image=True,
        freeze_text=True,
        freeze_text_embedding=True
    )

    # Use two GPUs if available
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("Using DataParallel on 2 GPUs")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Step 3: Train
    print("\nStarting training...")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    trainer = SMPPTrainer(model, optimizer, device=device)

    # Resume from latest checkpoint if available
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_files = list(ckpt_dir.glob("epoch_*.pth"))
    start_epoch = 0
    if ckpt_files:
        def _epoch_num(p):
            try:
                return int(p.stem.split("_")[-1])
            except Exception:
                return -1
        latest_ckpt = max(ckpt_files, key=_epoch_num)
        if _epoch_num(latest_ckpt) >= 0:
            print(f"Resuming from checkpoint: {latest_ckpt}")
            last_epoch, _ = trainer.load_checkpoint(str(latest_ckpt))
            start_epoch = last_epoch + 1
            print(f"Resuming at epoch {start_epoch}")

    best_val_loss = float("inf")
    no_improve = 0
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate(val_loader)

        swanlab.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc
        })

        trainer.save_checkpoint(f"checkpoints/epoch_{epoch}.pth", epoch, val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (val_loss did not improve).")
                break
    
    # Step 4: GBDT Regression Evaluation with strict train/val/test protocol
    print("\nRunning GBDT regression evaluation with strict train/val/test splits...")
    reg_dataset_view = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split=train_split,
        class_mapping=train_dataset.class_mapping,
        is_training=False
    )

    # Reuse the same train/val indices as the classification training stage.
    reg_train_dataset = Subset(reg_dataset_view, train_indices.tolist())
    reg_val_dataset = Subset(reg_dataset_view, val_indices.tolist())

    if len(reg_train_dataset) == 0 or len(reg_val_dataset) == 0:
        raise RuntimeError("GBDT train/val dataset is empty. Check val_ratio and train split size.")

    # Independent test split for final evaluation.
    reg_test_dataset = JSONDataset(
        metadata_dir=test_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split=test_split,
        class_mapping=train_dataset.class_mapping,
        is_training=False
    )

    if len(reg_test_dataset) == 0:
        raise RuntimeError("GBDT test dataset is empty. Check test split metadata.")

    gbdt_metrics = train_gbdt_and_evaluate(
        model=model,
        train_dataset=reg_train_dataset,
        val_dataset=reg_val_dataset,
        test_dataset=reg_test_dataset,
        device=device
    )

    print(
        "GBDT Val Metrics - "
        f"MSE: {gbdt_metrics['val']['mse']:.4f}, "
        f"MAE: {gbdt_metrics['val']['mae']:.4f}, "
        f"Spearman: {gbdt_metrics['val']['spearman']:.4f}"
    )
    print(
        "GBDT Test Metrics - "
        f"MSE: {gbdt_metrics['test']['mse']:.4f}, "
        f"MAE: {gbdt_metrics['test']['mae']:.4f}, "
        f"Spearman: {gbdt_metrics['test']['spearman']:.4f}"
    )

    swanlab.log({
        "gbdt/val_mse": gbdt_metrics["val"]["mse"],
        "gbdt/val_mae": gbdt_metrics["val"]["mae"],
        "gbdt/val_spearman": gbdt_metrics["val"]["spearman"],
        "gbdt/test_mse": gbdt_metrics["test"]["mse"],
        "gbdt/test_mae": gbdt_metrics["test"]["mae"],
        "gbdt/test_spearman": gbdt_metrics["test"]["spearman"]
    })

    # Log test evaluation curve points for SwanLab visualization.
    for point in gbdt_metrics.get("test_curve", []):
        swanlab.log({
            "gbdt/test_curve_rank": point["rank"],
            "gbdt/test_curve_sample_index": point["sample_index"],
            "gbdt/test_curve_y_true": point["y_true"],
            "gbdt/test_curve_y_pred": point["y_pred"],
            "gbdt/test_curve_abs_error": point["abs_error"],
        })
    swanlab.finish()

    print("\nAll done!")


if __name__ == "__main__":
    main()
