Technical Spec: Cross-Modal Prototype Augmentation & Dual-Grained Prompt Learning

1. 任务概述 (Task Overview)

本文提出一种用于社交媒体流行度预测 (Social Media Popularity Prediction, SMPP) 的多模态框架。

核心逻辑：



Pre-training/Fine-tuning: 将回归任务转化为 77类 的分类任务，微调 CLIP 模型 1111。

+1



Inference/Feature Extraction: 使用微调后的模型提取特征，输入到 GBDT (LightGBM/CatBoost) 进行最终的流行度数值预测 22。

+1

2. 核心模块详解 (Core Modules)

2.1 原型构建 (Prototype Construction)

为了增强特征的判别力，构建 256-shot 的多模态原型 3。



类别设定: 使用 77个 细粒度子类别 (Fine-grained subcategories) 代替原本的 11 个大类 4。

视觉原型 ($V_i$): 对每个类别选取 256 张代表性图片，通过 CLIP Image Encoder $f_I$ 编码后取平均：

$$V_i = \frac{1}{256} \sum_{n=1}^{256} f_I(x_n)$$

其中 $x_n$ 是第 $i$ 类的第 $n$ 张图片 5。

文本原型 ($T_i$): 对每个类别选取 256 个具有描述性的标题 (Titles)，通过 CLIP Text Encoder $f_T$ 编码后取平均：

$$T_i = \frac{1}{256} \sum_{n=1}^{256} f_T(t_n)$$

其中 $t_n$ 是对应的标题文本 6。



采样策略: 需考虑时间多样性、语义多样性 (子话题过滤) 和用户多样性 7。

2.2 双粒度提示学习 (Dual-Grained Prompt Learning)

为了解决文本输入不完整（如标题缺失）的问题，设计可学习的 Global 和 Local Prompts 8888。

+1

Prompt 定义:

Global Prompt ($w_i^G$): 用于提取全局类别 Embedding。

$$w_i^G = [\theta_{1:k}^G, e_i]$$

Local Prompt ($w_i^L$): 用于提取局部细粒度特征。

$$w_i^L = [\theta_{1:s}^L, e_i]$$



注: $\theta$ 为可学习向量，$e_i$ 为类别标签 Embedding 9。

编码过程:

将 Prompts 输入 CLIP Text Encoder $f_T$ 得到类别 Embeddings：

$$\{G_i, L_i\} = f_T(w_i^G, w_i^L)$$

其中 $G_i$ 是全局类别特征，$L_i$ 是局部类别特征 10。

输入文本处理:

保留输入文本 $r$ (Title 或 All tags) 的完整 Token 序列特征，而不仅仅是 EOS Token：

$$\{h, H\} = f_T(r)$$

$h \in \mathbb{R}^d$: 全局文本 Embedding (EOS token)。



$H \in \mathbb{R}^{l \times d}$: 序列 Token Embeddings (长度为 $l$) 11。

2.3 跨模态投影与交互 (Cross-Modal Projection)

引入 Transformer Layer 进行特征融合 12121212。

+1



投影 (Projection): 使用投影层 $P_I$ 和 $P_T$ 将图像特征 $x$、视觉原型 $V$、文本原型 $T$ 映射到统一维度 $d$ 13。

拼接 (Concatenation): 将三者拼接：

$$\text{Input} = \text{Concat}(P_I(x), P_I(V), P_I(T))$$

自注意力交互 (Self-Attention):

$$[\tilde{x}, \tilde{V}, \tilde{T}] = \text{SelfAttn}(\text{Input})$$

此操作捕捉模态间的上下文关系 14。

3. 损失函数设计 (Loss Functions)

总 Loss 定义为：

$$\mathcal{L} = \mathcal{L}_g + \mathcal{L}_o + \mathcal{L}_c$$

15

3.1 全局文本对齐损失 ($\mathcal{L}_g$)

计算输入文本全局特征 $h$ 与全局 Prompt 特征 $G_i$ 的余弦相似度 $p_i$，并计算 Cross-Entropy Loss 16161616。

+1

$$p_i = \cos \langle h, G_i \rangle$$

3.2 局部文本对齐损失 ($\mathcal{L}_o$)

对文本序列特征 $H$ 和局部 Prompt 特征 $L_i$ 进行空间聚合 (Spatial Aggregation) 17:

计算 Token 级相似度: $P_{ij} = \cos \langle H_j, L_i \rangle$

加权聚合:

$$p_i' = \frac{\sum_{j=1}^l \exp(P_{ij}/\tau_s)}{\sum_{j=1}^l \exp(P_{ij}/\tau_s)} \cdot P_{ij}$$

对 $p_i'$ 计算 Cross-Entropy Loss。

3.3 视觉与原型对齐损失 ($\mathcal{L}_c$)

基于交互后的特征 $\tilde{x}$ 进行分类预测18181818:

+1

视觉原型预测概率:

$$p_V(y_i=1) = \frac{\exp(\cos \langle \tilde{x}, \tilde{V}_i \rangle / \tau_v)}{\sum_{j=1}^K \exp(\cos \langle \tilde{x}, \tilde{V}_j \rangle / \tau_v)}$$

文本原型预测概率:

$$p_T(y_i=1) = \frac{\exp(\cos \langle \tilde{x}, \tilde{T}_i \rangle / \tau_t)}{\sum_{j=1}^K \exp(\cos \langle \tilde{x}, \tilde{T}_j \rangle / \tau_t)}$$

最终 $\mathcal{L}_c$: 对 $p_V$ 和 $p_T$ 的 logits 取平均后计算 Cross-Entropy。

4. 实验与实现细节 (Implementation Details)

4.1 训练设置 (Training Setup)



Framework: PyTorch19.



Optimizer: AdamW20.



Learning Rate: $1 \times 10^{-4}$21.



Weight Decay: $1 \times 10^{-5}$22.



Batch Size: 12823.



Epochs: 424.



Dataset: SMPD-Image (486k posts, 70k users)25.

4.2 下游回归预测 (Regression Stage)

微调完成后，冻结模型参数，提取以下特征向量 $F$26:

$$F = [f_I(x), f_I(t), w_i^G, w_i^L, \tilde{x}, \tilde{V}, \tilde{T}, s]$$

$f_I(x), f_I(t)$: 原始 CLIP 视觉/文本特征。

$w_i^G, w_i^L$: 学习到的 Prompt 特征。

$\tilde{x}, \tilde{V}, \tilde{T}$: 交互后的增强特征。



$s$: 用户行为特征 (User behavior features, 来源于引用文献 [33]) 27。



预测模型: LightGBM 或 CatBoost 28。

Sample Selection Trick: 仅使用 Classification Loss 最低的前 77% 的样本进行回归模型训练，以提高效率并保持性能 29。+1

5. 待编写代码模块 (Code Modules Required)

PrototypeBuilder: 负责从数据集中采样并计算 Visual/Textual Prototypes。

CustomCLIPTextEncoder: 修改 CLIP 的 Text Encoder，使其能返回 Token 序列特征 $H$ (不仅仅是 pooler output)。

DualGrainedPromptLearner: nn.Module，包含可学习的 global_ctx 和 local_ctx 参数。

CrossModalFusion: 实现 Multi-Head Self-Attention 的特征融合层。

SMPPLoss: 实现包含 $L_g, L_o, L_c$ 的组合 Loss。