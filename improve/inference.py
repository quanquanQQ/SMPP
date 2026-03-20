"""
Inference and Feature Extraction Script
推理和特征提取脚本
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import JSONDataset
from smpp_model import SMPPModel


class FeatureExtractor:
    """
    特征提取器，用于提取用于 GBDT 回归的特征
    """
    
    def __init__(
        self,
        model: SMPPModel,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
    
    @torch.no_grad()
    def extract_batch(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        user_features: torch.Tensor = None
    ) -> np.ndarray:
        """
        提取一个 batch 的特征
        
        Args:
            images: [batch_size, 3, H, W]
            text_tokens: [batch_size, seq_len]
            user_features: [batch_size, user_feature_dim] (可选)
        
        Returns:
            features: [batch_size, feature_dim]
        """
        images = images.to(self.device)
        text_tokens = text_tokens.to(self.device)
        
        if user_features is not None:
            user_features = user_features.to(self.device)
        
        # 提取特征
        features = self.model.extract_features_for_regression(
            images, text_tokens, user_features
        )
        
        return features.cpu().numpy()
    
    def extract_dataset(
        self,
        dataloader: DataLoader,
        save_path: str = None
    ) -> tuple:
        """
        提取整个数据集的特征
        
        Args:
            dataloader: 数据加载器
            save_path: 保存路径 (可选)
        
        Returns:
            features, labels, popularities, losses
        """
        all_features = []
        all_labels = []
        all_popularities = []
        all_losses = []
        
        print("Extracting features...")
        for batch in tqdm(dataloader):
            # 根据数据集格式解包
            if len(batch) == 4:
                images, text_tokens, labels, popularities = batch
            elif len(batch) == 5:
                images, text_tokens, labels, popularities, user_features = batch
            else:
                images, text_tokens = batch[:2]
                labels = torch.zeros(images.shape[0])
                popularities = torch.zeros(images.shape[0])
                user_features = None
            
            # 提取特征并计算分类损失
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            labels = labels.to(self.device)
            if len(batch) == 5:
                user_features = user_features.to(self.device)
            else:
                user_features = None

            output = self.model(images, text_tokens, labels, return_features=True)
            features = output['all_features'].detach().cpu().numpy()

            logits = (
                output['logits_global'] +
                output['logits_local'] +
                output['logits_visual']
            ) / 3
            batch_loss = F.cross_entropy(logits, labels, reduction='none')
            all_losses.append(batch_loss.detach().cpu().numpy())
            
            all_features.append(features)
            all_labels.append(labels.detach().cpu().numpy())
            all_popularities.append(popularities.detach().cpu().numpy())
        
        # 合并
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        popularities = np.concatenate(all_popularities, axis=0)
        losses = np.concatenate(all_losses, axis=0)
        
        print(f"Extracted {features.shape[0]} samples, feature dim: {features.shape[1]}")
        
        # 保存
        if save_path:
            np.savez(
                save_path,
                features=features,
                labels=labels,
                popularities=popularities,
                losses=losses
            )
            print(f"Features saved to {save_path}")
        
        return features, labels, popularities, losses
    
    def filter_by_classification_loss(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        popularities: np.ndarray,
        losses: np.ndarray,
        loss_threshold_percentile: float = 77.0
    ) -> tuple:
        """
        Sample Selection Trick: 仅使用分类损失最低的前 77% 样本
        
        Args:
            features: 特征矩阵
            labels: 分类标签
            popularities: 流行度数值
            losses: 每个样本的分类损失
            loss_threshold_percentile: 保留的百分位数
        
        Returns:
            filtered_features, filtered_popularities, filtered_labels
        """
        print(f"\nApplying sample selection (keeping top {loss_threshold_percentile}% samples)...")
        
        num_samples = features.shape[0]
        num_keep = int(num_samples * loss_threshold_percentile / 100)

        # 根据分类损失从小到大排序，保留前 77%
        sorted_indices = np.argsort(losses)
        indices = sorted_indices[:num_keep]
        indices = np.sort(indices)
        
        filtered_features = features[indices]
        filtered_popularities = popularities[indices]
        filtered_labels = labels[indices]
        
        print(f"Kept {num_keep}/{num_samples} samples ({loss_threshold_percentile}%)")
        
        return filtered_features, filtered_popularities, filtered_labels


class GBDTRegressor:
    """
    GBDT 回归器 (LightGBM 或 CatBoost)
    """
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        Args:
            model_type: "lightgbm" or "catboost"
        """
        self.model_type = model_type
        
        if model_type == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=64,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == "catboost":
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """训练 GBDT 模型"""
        print(f"\nTraining {self.model_type} regressor...")
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=100
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("Training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """评估模型"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        import joblib
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")


def inference_pipeline(
    smpp_model_path: str,
    gbdt_model_path: str,
    test_dataloader: DataLoader,
    prototype_path: str = None,
    num_classes: int = 77,
    device: str = "cuda"
):
    """
    完整的推理流程
    
    Args:
        smpp_model_path: SMPP 模型路径
        gbdt_model_path: GBDT 模型路径
        test_dataloader: 测试数据加载器
        device: 计算设备
    
    Returns:
        predictions: 预测的流行度数值
    """
    print("Loading models...")
    
    # 1. 加载 SMPP 模型
    smpp_model = _load_smpp_model(
        checkpoint_path=smpp_model_path,
        prototype_path=prototype_path,
        num_classes=num_classes,
        device=device
    )
    
    # 2. 加载 GBDT 模型
    gbdt_model = GBDTRegressor()
    gbdt_model.load(gbdt_model_path)
    
    # 3. 提取特征
    extractor = FeatureExtractor(smpp_model, device)
    features, _, _, _ = extractor.extract_dataset(test_dataloader)
    
    # 4. GBDT 预测
    print("\nPredicting with GBDT...")
    predictions = gbdt_model.predict(features)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions


def main():
    """
    主函数：推理和特征提取示例
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 示例 1: 提取特征
    # feature_extractor = FeatureExtractor(model, device)
    # features, labels, popularities, losses = feature_extractor.extract_dataset(
    #     dataloader, save_path="extracted_features.npz"
    # )
    
    # 示例 2: 训练 GBDT
    # gbdt = GBDTRegressor(model_type="lightgbm")
    # gbdt.train(X_train, y_train, X_val, y_val)
    # gbdt.save("gbdt_model.pkl")
    
    # 示例 3: 完整推理
    # predictions = inference_pipeline(
    #     smpp_model_path="checkpoints/best_model.pth",
    #     gbdt_model_path="gbdt_model.pkl",
    #     test_dataloader=test_loader,
    #     device=device
    # )
    
    print("Inference script ready!")


def _build_dataloader(
    metadata_dir: str,
    image_dir: str,
    split: str,
    clip_preprocess,
    batch_size: int,
    num_workers: int,
    text_field: str,
    include_user_features: bool
) -> DataLoader:
    dataset = JSONDataset(
        metadata_dir=metadata_dir,
        image_dir=image_dir,
        clip_preprocess=clip_preprocess,
        split=split,
        text_field=text_field,
        include_user_features=include_user_features
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def _load_smpp_model(
    checkpoint_path: str,
    prototype_path: str,
    num_classes: int,
    device: str
) -> SMPPModel:
    model = SMPPModel(num_classes=num_classes, device=device)

    if prototype_path and Path(prototype_path).exists():
        proto_ckpt = torch.load(prototype_path, map_location=device)
        visual_prototypes = proto_ckpt.get('visual_prototypes')
        textual_prototypes = proto_ckpt.get('textual_prototypes')
        if visual_prototypes is not None and textual_prototypes is not None:
            model.load_prototypes(visual_prototypes, textual_prototypes)
        else:
            print("Warning: Prototypes file missing required keys.")
    else:
        print("Warning: No prototypes loaded. Please provide --prototype-path.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _load_features_npz(path: str) -> tuple:
    data = np.load(path)
    features = data['features']
    labels = data.get('labels')
    popularities = data.get('popularities')
    losses = data.get('losses')
    return features, labels, popularities, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPP inference utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract-features
    extract_parser = subparsers.add_parser("extract-features", help="Extract features for GBDT")
    extract_parser.add_argument("--metadata-dir", required=True)
    extract_parser.add_argument("--image-dir", required=True)
    extract_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    extract_parser.add_argument("--text-field", default="Title")
    extract_parser.add_argument("--checkpoint-path", required=True)
    extract_parser.add_argument("--prototype-path", default=None)
    extract_parser.add_argument("--num-classes", type=int, default=77)
    extract_parser.add_argument("--batch-size", type=int, default=128)
    extract_parser.add_argument("--num-workers", type=int, default=0)
    extract_parser.add_argument("--include-user-features", action="store_true")
    extract_parser.add_argument("--save-path", default="extracted_features.npz")
    extract_parser.add_argument("--filter-percentile", type=float, default=77.0)
    extract_parser.add_argument("--filtered-save-path", default=None)

    # train-gbdt
    gbdt_parser = subparsers.add_parser("train-gbdt", help="Train GBDT regressor")
    gbdt_parser.add_argument("--features-path", required=True)
    gbdt_parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "catboost"])
    gbdt_parser.add_argument("--save-path", default="gbdt_model.pkl")
    gbdt_parser.add_argument("--val-size", type=float, default=0.1)
    gbdt_parser.add_argument("--test-size", type=float, default=0.1)

    # inference
    infer_parser = subparsers.add_parser("infer", help="Run full inference")
    infer_parser.add_argument("--metadata-dir", required=True)
    infer_parser.add_argument("--image-dir", required=True)
    infer_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    infer_parser.add_argument("--text-field", default="Title")
    infer_parser.add_argument("--checkpoint-path", required=True)
    infer_parser.add_argument("--prototype-path", default=None)
    infer_parser.add_argument("--gbdt-model-path", required=True)
    infer_parser.add_argument("--num-classes", type=int, default=77)
    infer_parser.add_argument("--batch-size", type=int, default=128)
    infer_parser.add_argument("--num-workers", type=int, default=0)
    infer_parser.add_argument("--include-user-features", action="store_true")
    infer_parser.add_argument("--save-path", default="predictions.npy")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.command == "extract-features":
        model = _load_smpp_model(
            checkpoint_path=args.checkpoint_path,
            prototype_path=args.prototype_path,
            num_classes=args.num_classes,
            device=device
        )
        dataloader = _build_dataloader(
            metadata_dir=args.metadata_dir,
            image_dir=args.image_dir,
            split=args.split,
            clip_preprocess=model.preprocess,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_field=args.text_field,
            include_user_features=args.include_user_features
        )
        extractor = FeatureExtractor(model, device)
        features, labels, popularities, losses = extractor.extract_dataset(
            dataloader, save_path=args.save_path
        )

        if args.filtered_save_path:
            filtered_features, filtered_popularities, filtered_labels = (
                extractor.filter_by_classification_loss(
                    features,
                    labels,
                    popularities,
                    losses,
                    loss_threshold_percentile=args.filter_percentile
                )
            )
            np.savez(
                args.filtered_save_path,
                features=filtered_features,
                labels=filtered_labels,
                popularities=filtered_popularities
            )
            print(f"Filtered features saved to {args.filtered_save_path}")

    elif args.command == "train-gbdt":
        from sklearn.model_selection import train_test_split

        features, _, popularities, _ = _load_features_npz(args.features_path)
        X_train, X_temp, y_train, y_temp = train_test_split(
            features,
            popularities,
            test_size=(args.val_size + args.test_size),
            random_state=42
        )
        relative_test_size = args.test_size / (args.val_size + args.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=42
        )

        gbdt = GBDTRegressor(model_type=args.model_type)
        gbdt.train(X_train, y_train, X_val, y_val)
        gbdt.evaluate(X_test, y_test)
        gbdt.save(args.save_path)

    elif args.command == "infer":
        model = _load_smpp_model(
            checkpoint_path=args.checkpoint_path,
            prototype_path=args.prototype_path,
            num_classes=args.num_classes,
            device=device
        )
        dataloader = _build_dataloader(
            metadata_dir=args.metadata_dir,
            image_dir=args.image_dir,
            split=args.split,
            clip_preprocess=model.preprocess,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text_field=args.text_field,
            include_user_features=args.include_user_features
        )
        predictions = inference_pipeline(
            smpp_model_path=args.checkpoint_path,
            gbdt_model_path=args.gbdt_model_path,
            test_dataloader=dataloader,
            prototype_path=args.prototype_path,
            num_classes=args.num_classes,
            device=device
        )
        np.save(args.save_path, predictions)
        print(f"Predictions saved to {args.save_path}")
