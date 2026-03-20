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

from smpp_model import SMPPModel, SMPPTrainer
from prototype_builder import PrototypeBuilder
from dataset import JSONDataset

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
    dataset: JSONDataset,
    device: str = "cuda",
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    使用提取特征训练 GBDT，并计算 MSE/MAE/Spearman
    """
    # 划分训练/验证
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=0)

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

    # 评估
    preds = reg.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    spearman = spearmanr(y_val, preds).correlation

    return mse, mae, spearman


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
    num_epochs = 100
    learning_rate = 1e-5
    weight_decay = 5e-4
    early_stop_patience = 4
    train_split = "train_new"
    val_split = "test_new"
    
    # Paths
    # Update dataset paths to the specified locations
    train_metadata_dir = "/mnt/sda/data/fame_split_37"
    test_metadata_dir = "/mnt/sda/data/fame_split_37"
    image_dir = "/mnt/sda/data/fame_split_37"
    
    print(f"Using device: {device}")

    # Initialize SwanLab
    swanlab_resume = os.getenv("SWANLAB_RESUME", "allow")
    swanlab_id = os.getenv("SWANLAB_ID", "3ppl5s3lbwa09fs8nrzsm")
    swanlab.init(
        project="SMPP",
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
        split=train_split,
        is_training=True
    )
    
    # Validation set
    val_dataset = JSONDataset(
        metadata_dir=test_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split=val_split,
        is_training=False,
        # Use training class mapping to ensure consistency
        class_mapping=train_dataset.class_mapping 
    )
    
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
    prototype_path = f"prototypes_{train_split}_{num_classes}cls.pth"
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
    
    # Step 4: GBDT Regression Evaluation (Original scheme)
    print("\nRunning GBDT regression evaluation...")
    reg_dataset = JSONDataset(
        metadata_dir=train_metadata_dir,
        image_dir=image_dir,
        clip_preprocess=preprocess,
        split=train_split,
        class_mapping=train_dataset.class_mapping,
        is_training=True
    )
    mse, mae, spearman = train_gbdt_and_evaluate(
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
