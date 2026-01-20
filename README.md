# Social Media Popularity Prediction (SMPP) Model

完整的 PyTorch 实现，基于跨模态原型增强和双粒度 Prompt Learning。
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
## 项目结构

```
prompt_code/
├── prototype_builder.py       # 原型构建模块 (256-shot 采样)
├── custom_clip_encoder.py     # 自定义 CLIP 编码器 (返回序列特征)
├── dual_prompt_learner.py     # 双粒度 Prompt Learning 模块
├── cross_modal_fusion.py      # 跨模态融合模块 (Transformer)
├── smpp_loss.py              # 损失函数 (L_g + L_o + L_c)
├── smpp_model.py             # 主模型类
├── dataset.py                # 数据集和数据加载器
├── train.py                  # 训练脚本
├── inference.py              # 推理和特征提取脚本
└── README.md                 # 本文件
```
核心模块
### prototype_builder.py - 原型构建模块

实现 256-shot 视觉和文本原型构建
支持时间、用户、语义多样性采样策略
### custom_clip_encoder.py - 自定义 CLIP 编码器

返回全局特征 h 和序列特征 H
支持 Prompt 编码
### dual_prompt_learner.py - 双粒度 Prompt Learning

Global Prompt: w_i^G = [θ_{1:k}^G, e_i]
Local Prompt: w_i^L = [θ_{1:s}^L, e_i]
可学习的 context 向量
### cross_modal_fusion.py - 跨模态融合

Transformer 层进行特征交互
投影层统一不同模态维度
支持自适应融合和原型选择
### smpp_loss.py - 损失函数 ⭐

L_g: 全局文本对齐损失 (h vs G)
L_o: 局部文本对齐损失 (H vs L, 空间聚合)
L_c: 视觉与原型对齐损失 (x̃ vs Ṽ, T̃)
总损失: L = L_g + L_o + L_c
### smpp_model.py - 主模型类

整合所有模块
完整的前向传播
训练器类
辅助模块
### dataset.py - 数据集和数据加载器

SMPDataset: 标准数据集
PrototypeDataset: 原型构建专用
数据统计和可视化工具
### train.py - 训练脚本

完整训练流程
模型创建和配置
检查点保存
### inference.py - 推理脚本

特征提取器
GBDT 回归器 (LightGBM/CatBoost)
Sample Selection (77% 样本筛选)
完整推理流程
README.md - 详细文档

requirements.txt - 依赖项
## 核心模块说明

### 1. Prototype Builder (`prototype_builder.py`)

构建视觉和文本原型：
- 每个类别采样 256 个代表性样本
- 考虑时间多样性、用户多样性、语义多样性
- 使用 CLIP 编码并计算平均作为原型

```python
from prototype_builder import PrototypeBuilder

builder = PrototypeBuilder(num_classes=77, num_shots=256)
visual_prototypes = builder.build_visual_prototypes(image_samples)
textual_prototypes = builder.build_textual_prototypes(text_samples)
```

### 2. Custom CLIP Encoder (`custom_clip_encoder.py`)

修改 CLIP Text Encoder，返回：
- `h`: 全局文本特征 (EOS token)
- `H`: 完整序列 Token 特征

```python
from custom_clip_encoder import create_custom_text_encoder

text_encoder, clip_model, preprocess = create_custom_text_encoder("ViT-B/32")
h, H = text_encoder(text_tokens)  # h: [B, d], H: [B, L, d]
```

### 3. Dual-Grained Prompt Learner (`dual_prompt_learner.py`)

实现可学习的 Global 和 Local Prompts：
- Global Prompt: `w_i^G = [θ_{1:k}^G, e_i]`
- Local Prompt: `w_i^L = [θ_{1:s}^L, e_i]`

```python
from dual_prompt_learner import create_prompt_learner

prompt_learner = create_prompt_learner(
    clip_model, num_classes=77,
    global_ctx_length=16, local_ctx_length=8
)
G, L = prompt_learner()  # G, L: [num_classes, feature_dim]
```

### 4. Cross-Modal Fusion (`cross_modal_fusion.py`)

使用 Transformer 进行跨模态特征交互：
- 投影: `P_I(x)`, `P_I(V)`, `P_I(T)`
- 自注意力: `[x̃, Ṽ, T̃] = SelfAttn(Concat(...))`

```python
from cross_modal_fusion import CrossModalFusion

fusion = CrossModalFusion(
    image_dim=512, text_dim=512, hidden_dim=512,
    num_heads=8, num_layers=2
)
x_tilde, V_tilde, T_tilde = fusion(image_features, V, T)
```

### 5. SMPP Loss (`smpp_loss.py`)

三个损失函数的组合：
- **L_g**: 全局文本对齐损失 (h vs G)
- **L_o**: 局部文本对齐损失 (H vs L, 带空间聚合)
- **L_c**: 视觉与原型对齐损失 (x̃ vs Ṽ, T̃)

```python
from smpp_loss import SMPPLoss

criterion = SMPPLoss(
    temperature=0.07,
    spatial_temperature=0.1,
    temperature_visual=0.07,
    temperature_textual=0.07
)
loss_dict = criterion(h, H, G, L, x_tilde, V_tilde, T_tilde, labels)
total_loss = loss_dict['loss']  # L_g + L_o + L_c
```

### 6. Main Model (`smpp_model.py`)

整合所有模块的完整模型：

```python
from smpp_model import SMPPModel

model = SMPPModel(
    num_classes=77,
    clip_model_name="ViT-B/32",
    global_ctx_length=16,
    local_ctx_length=8,
    fusion_hidden_dim=512,
    device="cuda"
)

# 加载原型
model.load_prototypes(visual_prototypes, textual_prototypes)

# 前向传播
output = model(images, text_tokens, labels)
loss = output['loss']
```

## 使用流程

### Step 1: 准备数据

数据格式 (CSV):
```
image_path,title,category,popularity,timestamp,user_id,...
img1.jpg,"Sample Title",5,1234,2024-01-01,user001,...
```

### Step 2: 构建原型

```python
from prototype_builder import PrototypeBuilder
from dataset import PrototypeDataset

# 创建原型数据集
proto_dataset = PrototypeDataset(
    data_csv="train.csv",
    image_dir="./images",
    clip_preprocess=preprocess,
    num_classes=77,
    num_shots=256
)

# 构建原型
builder = PrototypeBuilder(num_classes=77, num_shots=256)
visual_prototypes = builder.build_visual_prototypes(image_samples)
textual_prototypes = builder.build_textual_prototypes(text_samples)
builder.save_prototypes(visual_prototypes, textual_prototypes, "prototypes.pth")
```

### Step 3: 训练模型

```python
from train import train, create_model
from dataset import create_dataloaders
import clip

# 加载 CLIP 预处理
_, preprocess = clip.load("ViT-B/32", device="cuda")

# 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    train_csv="train.csv",
    val_csv="val.csv",
    test_csv="test.csv",
    image_dir="./images",
    clip_preprocess=preprocess,
    batch_size=128
)

# 创建模型
model = create_model(
    num_classes=77,
    class_names=class_names,
    prototype_path="prototypes.pth",
    device="cuda"
)

# 训练
train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    save_dir="./checkpoints"
)
```

### Step 4: 提取特征用于 GBDT

```python
from inference import FeatureExtractor, GBDTRegressor
import numpy as np

# 提取特征
extractor = FeatureExtractor(model, device="cuda")
features, labels, popularities = extractor.extract_dataset(
    test_loader, save_path="features.npz"
)

# Sample Selection: 保留分类损失最低的 77% 样本
features_filtered, popularities_filtered = extractor.filter_by_classification_loss(
    features, labels, popularities, loss_threshold_percentile=77.0
)

# 训练 GBDT
gbdt = GBDTRegressor(model_type="lightgbm")
gbdt.train(X_train, y_train, X_val, y_val)
gbdt.evaluate(X_test, y_test)
gbdt.save("gbdt_model.pkl")
```

### Step 5: 推理

```python
from inference import inference_pipeline

predictions = inference_pipeline(
    smpp_model_path="checkpoints/best_model.pth",
    gbdt_model_path="gbdt_model.pkl",
    test_dataloader=test_loader,
    device="cuda"
)
```

## 训练配置

根据论文实验设置：

- **Framework**: PyTorch
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-5
- **Batch Size**: 128
- **Epochs**: 4
- **Dataset**: SMPD-Image (486k posts, 70k users)

## 特征向量构成

用于 GBDT 回归的特征向量 `F`:

```
F = [f_I(x), f_T(t), w_i^G, w_i^L, x̃, Ṽ, T̃, s]
```

- `f_I(x)`: CLIP 图像特征
- `f_T(t)`: CLIP 文本特征
- `w_i^G`: Global Prompt 特征
- `w_i^L`: Local Prompt 特征
- `x̃`: 增强图像特征
- `Ṽ`: 增强视觉原型特征
- `T̃`: 增强文本原型特征
- `s`: 用户行为特征

## 依赖项

```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install lightgbm catboost
pip install pandas numpy scikit-learn
pip install matplotlib pillow
```

## 注意事项

1. **内存管理**: 77 个类别的原型在每次前向传播时都需要融合，建议使用 batch 处理或缓存策略
2. **数据预处理**: 确保图像和文本质量，处理缺失值
3. **超参数调整**: 根据具体数据集调整 context 长度、温度参数等
4. **采样策略**: 原型采样时需要平衡多样性和代表性

## 引用

如果使用本代码，请引用相关论文。

## License

MIT License
