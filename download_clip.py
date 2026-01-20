"""
下载和测试 CLIP 模型
"""

import torch
import clip

def download_clip_model():
    """下载 CLIP 模型"""
    print("开始下载 CLIP 模型...")
    print("模型名称: ViT-B/32")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 下载并加载模型
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ CLIP 模型下载并加载成功！")
        
        # 测试模型
        print("\n模型信息:")
        print(f"  - 视觉编码器输出维度: {model.visual.output_dim}")
        print(f"  - Token embedding 维度: {model.token_embedding.weight.shape[1]}")
        print(f"  - 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # 测试简单推理
        import numpy as np
        from PIL import Image
        
        print("\n测试模型推理...")
        # 创建一个测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_input = preprocess(test_image).unsqueeze(0).to(device)
        
        # 测试文本
        text_input = clip.tokenize(["a photo of a cat"]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
        
        print(f"  - 图像特征形状: {image_features.shape}")
        print(f"  - 文本特征形状: {text_features.shape}")
        print("✓ 模型推理测试成功！")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


if __name__ == "__main__":
    success = download_clip_model()
    
    if success:
        print("\n" + "="*50)
        print("CLIP 模型准备完成！")
        print("="*50)
        print("\n下一步:")
        print("1. 准备你的数据集 (图像 + CSV 文件)")
        print("2. 运行 prototype_builder.py 构建原型")
        print("3. 运行 train.py 开始训练")
    else:
        print("\n请检查错误并重试")
