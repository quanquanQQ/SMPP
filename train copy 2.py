"""
Training Script for SMPP Model
完整的训练流程示例
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import numpy as np
from pathlib import Path
import swanlab
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor

from smpp_model import SMPPModel, SMPPTrainer
from prototype_builder import PrototypeBuilder
from dataset import JSONDataset, DatasetStatistics

def create_model(
    num_classes: int = 77,
    class_names: list = None,
    prototype_path: str = None,
    device: str = "cuda",
    freeze_image: bool = False,
    freeze_text: bool = True,
    freeze_text_embedding: bool = False
):
    """
    创建并初始化 SMPP 模型
    
    Args:
        num_classes: 类别数量
        class_names: 类别名称列表
        prototype_path: 预先计算的原型文件路径
        device: 计算设备
    
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

        images, texts = builder.sample_diverse_data(dataset, class_id)
        image_samples[class_id] = images
        text_samples[class_id] = texts

    visual_prototypes = builder.build_visual_prototypes(image_samples)
    textual_prototypes = builder.build_textual_prototypes(text_samples)

    builder.save_prototypes(visual_prototypes, textual_prototypes, save_path)
    print(f"Prototypes saved to {save_path}")

    return visual_prototypes, textual_prototypes


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


def train_gbdt_and_evaluate_with_loss_filter(
    model: SMPPModel,
    dataset: JSONDataset,
    device: str = "cuda",
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    使用提取特征训练 LightGBM 和 CatBoost，并融合它们的预测结果，计算 MSE/MAE/Spearman。
    """
    # 划分训练/验证
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=0)

    # 筛选样本
    keep_indices = filter_samples_by_loss(model, train_loader, keep_ratio=0.77, device=device)
    train_subset = torch.utils.data.Subset(train_subset, keep_indices)

    # 提取特征
    X_train, _, y_train = extract_features_for_gbdt(
        model=model,
        dataloader=DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0),
        save_path="features_train_filtered.npz",
        device=device
    )
    X_val, _, y_val = extract_features_for_gbdt(
        model=model,
        dataloader=val_loader,
        save_path="features_val.npz",
        device=device
    )

    # 训练 LightGBM
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    lgb_reg.fit(X_train, y_train)
    lgb_preds = lgb_reg.predict(X_val)

    # 训练 CatBoost
    catboost_reg = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=0
    )
    catboost_reg.fit(X_train, y_train)
    catboost_preds = catboost_reg.predict(X_val)

    # 融合 LightGBM 和 CatBoost 的预测结果
    fused_preds = fusion_model(lgb_preds, catboost_preds, weight_lightgbm=0.5, weight_catboost=0.5)

    # 评估
    mse = mean_squared_error(y_val, fused_preds)
    mae = mean_absolute_error(y_val, fused_preds)
    spearman = spearmanr(y_val, fused_preds).correlation

    return mse, mae, spearman


def fusion_model(lightgbm_preds, catboost_preds, weight_lightgbm=0.5, weight_catboost=0.5):
    """
    融合 LightGBM 和 CatBoost 的预测结果。

    Args:
        lightgbm_preds: LightGBM 的预测结果 (numpy array)
        catboost_preds: CatBoost 的预测结果 (numpy array)
        weight_lightgbm: LightGBM 的权重
        weight_catboost: CatBoost 的权重

    Returns:
        融合后的预测结果 (numpy array)
    """
    # 确保权重和为 1
    total_weight = weight_lightgbm + weight_catboost
    weight_lightgbm /= total_weight
    weight_catboost /= total_weight

    # 加权平均融合
    fused_preds = (weight_lightgbm * lightgbm_preds) + (weight_catboost * catboost_preds)
    return fused_preds


def filter_samples_by_loss(model, dataloader, keep_ratio=0.77, device="cuda"):
    """
    基于分类 Loss 筛选样本。

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        keep_ratio: 保留的样本比例
        device: 计算设备

    Returns:
        筛选后的样本索引
    """
    print("Filtering samples based on classification loss...")
    model.eval()
    losses = []
    indices = []

    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # 不求平均，保留每个样本的 loss

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images, text_tokens, labels, _ = batch
            images, text_tokens, labels = images.to(device), text_tokens.to(device), labels.to(device)

            output = model(images, text_tokens, labels)
            # Use averaged logits like training/validation for consistent filtering.
            logits = (
                output["logits_global"] +
                output["logits_local"] +
                output["logits_visual"]
            ) / 3

            batch_loss = criterion(logits, labels)
            losses.append(batch_loss.cpu().numpy())
            indices.extend(range(batch_idx * dataloader.batch_size, (batch_idx + 1) * dataloader.batch_size))

    all_losses = np.concatenate(losses)

    # 找到阈值
    threshold_idx = int(len(all_losses) * keep_ratio)
    sorted_indices = np.argsort(all_losses)
    keep_indices = sorted_indices[:threshold_idx]

    return keep_indices


def main():
    """
    主函数: 完整的训练流程
    """
    # 配置
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_classes = 77
    batch_size = 64
    num_epochs = 4
    learning_rate = 1e-5
    weight_decay = 5e-4
    early_stop_patience = 2
    
    # Paths
    # Use fame_split (same level as prompt): train_new / test_new
    train_metadata_dir = "/mnt/sda/data/fame_split_37"
    test_metadata_dir = "/mnt/sda/data/fame_split_37"
    image_dir = "/mnt/sda/data/fame_split_37"  # Parent of train_new/ and test_new/
    
    print(f"Using device: {device}")

    # Initialize SwanLab
    swanlab_project = os.getenv("SWANLAB_PROJECT", "SMPP")
    swanlab_resume = os.getenv("SWANLAB_RESUME", "allow")
    swanlab_id = os.getenv("SWANLAB_ID", "bg8rk1xguli8ni0atmbs6")
    swanlab.init(
        project=swanlab_project,
        resume=swanlab_resume,
        id=swanlab_id if swanlab_id else None,
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stop_patience": early_stop_patience,
            "device": device
        }
    )
    
    # Prepare Preprocessing
    _, preprocess = clip.load("ViT-B/32", device=device)
    
    # Step 0: Load Datasets
    print("Loading datasets...")
    # Training set
    train_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split="train_new",
        is_training=True
    )

    val_dataset = JSONDataset(
        metadata_dir=test_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split="test_new",
        is_training=False,
        class_mapping=train_dataset.class_mapping
    )

    reg_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split="train_new",
        is_training=False,
        class_mapping=train_dataset.class_mapping
    )

    # Dataset statistics & visualization
    try:
        stats = DatasetStatistics(train_dataset)
        stats.print_statistics()
        stats.plot_distribution()
    except Exception as e:
        print(f"Warning: Failed to plot dataset statistics: {e}")
    
    # 类别名称
    class_names = [k for k, v in sorted(train_dataset.class_mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Step 1: 构建原型 (如果还没有)
    prototype_path = "prototypes.pth"
    if not Path(prototype_path).exists():
        build_prototypes(
            dataset=train_dataset,
            num_classes=num_classes,
            num_shots=256,
            save_path=prototype_path,
            device=device
        )
    
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

    # Force single-GPU/CPU to avoid DataParallel issues.
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("DataParallel disabled; using a single device.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Step 3: Train
    print("\nStarting training...")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay
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
    
    # Step 4: GBDT Regression Evaluation (Original scheme)
    print("\nRunning GBDT regression evaluation...")
    reg_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split="train_new",
        is_training=False,
        class_mapping=train_dataset.class_mapping,
    )
    mse, mae, spearman = train_gbdt_and_evaluate_with_loss_filter(
        model=model,
        dataset=reg_dataset,
        device=device
    )
    print(f"GBDT Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, Spearman: {spearman:.4f}")
    swanlab.log({
        "gbdt/mse": mse,
        "gbdt/mae": mae,
        "gbdt/spearman": spearman
    })

    print("\nAll done!")


if __name__ == "__main__":
    main()
