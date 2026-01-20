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

from smpp_model import SMPPModel, SMPPTrainer
from prototype_builder import PrototypeBuilder


def create_model(
    num_classes: int = 77,
    class_names: list = None,
    prototype_path: str = None,
    device: str = "cuda"
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
        device=device
    )
    
    # 加载原型 (如果有)
    if prototype_path and Path(prototype_path).exists():
        print(f"Loading prototypes from {prototype_path}")
        checkpoint = torch.load(prototype_path)
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
    构建原型并保存
    
    Args:
        dataset: 包含所有训练数据的数据集
        num_classes: 类别数量
        num_shots: 每类采样数量
        save_path: 保存路径
        device: 计算设备
    """
    print("Building prototypes...")
    
    # 创建原型构建器
    builder = PrototypeBuilder(
        num_classes=num_classes,
        num_shots=num_shots,
        device=device
    )
    
    # 为每个类别采样数据
    image_samples = {}
    text_samples = {}
    
    for class_id in range(num_classes):
        print(f"Sampling class {class_id}/{num_classes}")
        
        # 从数据集中采样
        # 这里需要根据实际数据集实现采样逻辑
        # images, texts = builder.sample_diverse_data(dataset, class_id)
        
        # 示例: 假设已经有采样好的数据
        # image_samples[class_id] = images
        # text_samples[class_id] = texts
        pass
    
    # 构建视觉和文本原型
    visual_prototypes = builder.build_visual_prototypes(image_samples)
    textual_prototypes = builder.build_textual_prototypes(text_samples)
    
    # 保存原型
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


def main():
    """
    主函数: 完整的训练流程
    """
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 77
    batch_size = 128
    num_epochs = 4
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # 类别名称 (需要根据实际数据集定义)
    class_names = [f"class_{i}" for i in range(num_classes)]
    
    print(f"Using device: {device}")
    
    # Step 1: 构建原型 (如果还没有)
    # build_prototypes(
    #     dataset=your_dataset,
    #     num_classes=num_classes,
    #     num_shots=256,
    #     save_path="prototypes.pth",
    #     device=device
    # )
    
    # Step 2: 创建模型
    print("\nCreating model...")
    model = create_model(
        num_classes=num_classes,
        class_names=class_names,
        prototype_path="prototypes.pth",
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Step 3: 准备数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Step 4: 训练模型
    # print("\nStarting training...")
    # train(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=num_epochs,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     save_dir="./checkpoints",
    #     device=device
    # )
    
    # Step 5: 提取特征用于 GBDT
    # print("\nExtracting features for GBDT...")
    # extract_features_for_gbdt(
    #     model=model,
    #     dataloader=test_loader,
    #     save_path="features_for_gbdt.npz",
    #     device=device
    # )
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
