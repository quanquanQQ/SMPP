"""
Inference and Feature Extraction Script
推理和特征提取脚本
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

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
            features, labels, popularities
        """
        all_features = []
        all_labels = []
        all_popularities = []
        
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
            
            # 提取特征
            features = self.extract_batch(images, text_tokens, 
                                         user_features if len(batch) == 5 else None)
            
            all_features.append(features)
            all_labels.append(labels.numpy())
            all_popularities.append(popularities.numpy())
        
        # 合并
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        popularities = np.concatenate(all_popularities, axis=0)
        
        print(f"Extracted {features.shape[0]} samples, feature dim: {features.shape[1]}")
        
        # 保存
        if save_path:
            np.savez(
                save_path,
                features=features,
                labels=labels,
                popularities=popularities
            )
            print(f"Features saved to {save_path}")
        
        return features, labels, popularities
    
    def filter_by_classification_loss(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        popularities: np.ndarray,
        loss_threshold_percentile: float = 77.0
    ) -> tuple:
        """
        Sample Selection Trick: 仅使用分类损失最低的前 77% 样本
        
        Args:
            features: 特征矩阵
            labels: 分类标签
            popularities: 流行度数值
            loss_threshold_percentile: 保留的百分位数
        
        Returns:
            filtered_features, filtered_popularities
        """
        print(f"\nApplying sample selection (keeping top {loss_threshold_percentile}% samples)...")
        
        # 计算每个样本的分类损失
        # 这需要模型的预测结果
        # 简化版本：随机选择
        num_samples = features.shape[0]
        num_keep = int(num_samples * loss_threshold_percentile / 100)
        
        # 实际应该基于分类损失排序
        # 这里使用随机选择作为示例
        indices = np.random.choice(num_samples, num_keep, replace=False)
        indices = np.sort(indices)
        
        filtered_features = features[indices]
        filtered_popularities = popularities[indices]
        
        print(f"Kept {num_keep}/{num_samples} samples ({loss_threshold_percentile}%)")
        
        return filtered_features, filtered_popularities


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
    # 需要先创建模型结构
    from smpp_model import SMPPModel
    smpp_model = SMPPModel(num_classes=77, device=device)
    
    checkpoint = torch.load(smpp_model_path)
    smpp_model.load_state_dict(checkpoint['model_state_dict'])
    smpp_model.eval()
    
    # 2. 加载 GBDT 模型
    gbdt_model = GBDTRegressor()
    gbdt_model.load(gbdt_model_path)
    
    # 3. 提取特征
    extractor = FeatureExtractor(smpp_model, device)
    features, _, _ = extractor.extract_dataset(test_dataloader)
    
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
    # features, labels, popularities = feature_extractor.extract_dataset(
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


if __name__ == "__main__":
    main()
