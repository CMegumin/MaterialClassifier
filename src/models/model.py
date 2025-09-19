import torch
import torch.nn as nn
from torchvision import models
import os
from tqdm import tqdm

def download_model(model_name):
    """
    下载预训练模型
    
    Args:
        model_name (str): 模型名称
    """
    # 模型权重文件的默认保存位置
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"正在检查预训练模型 {model_name} ...")
    
    try:
        # 使用 tqdm 显示进度条
        for _ in tqdm(range(1), desc=f"下载 {model_name}"):
            if model_name == "resnet18":
                models.resnet18(weights="IMAGENET1K_V1")
            elif model_name == "resnet34":
                models.resnet34(weights="IMAGENET1K_V1")
            elif model_name == "resnet50":
                models.resnet50(weights="IMAGENET1K_V2")
            elif model_name == "efficientnet_b0":
                models.efficientnet_b0(weights="IMAGENET1K_V1")
            elif model_name == "mobilenet_v2":
                models.mobilenet_v2(weights="IMAGENET1K_V1")
            
        print(f"预训练模型 {model_name} 已准备就绪！")
        print(f"模型文件保存在: {cache_dir}")
        
    except Exception as e:
        print(f"下载模型时出错: {str(e)}")
        print("请检查网络连接，或手动下载模型文件。")
        raise

def get_model(model_name, num_classes):
    """
    获取预训练模型
    
    Args:
        model_name (str): 模型名称，支持的模型：
            - resnet18: 轻量级模型，适合快速实验
            - resnet34: 轻量级模型，比resnet18深
            - resnet50: 中等规模，性能和效率较好的平衡
            - efficientnet_b0: 轻量级但效果好
            - mobilenet_v2: 移动端友好的轻量级模型
        num_classes (int): 类别数量
    
    Returns:
        model (nn.Module): PyTorch模型
    """
    model_name = model_name.lower()
    
    # 确保模型已下载
    download_model(model_name)
    
    # ResNet系列
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = models.resnet18(weights="IMAGENET1K_V1")
        elif model_name == 'resnet34':
            model = models.resnet34(weights="IMAGENET1K_V1")
        elif model_name == 'resnet50':
            model = models.resnet50(weights="IMAGENET1K_V2")
        else:
            raise ValueError(f"不支持的ResNet模型: {model_name}")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    # EfficientNet
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    # MobileNet
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        supported_models = [
            'resnet18', 'resnet34', 'resnet50',
            'efficientnet_b0', 'mobilenet_v2'
        ]
        raise ValueError(
            f"不支持的模型类型: {model_name}\n"
            f"支持的模型类型有: {', '.join(supported_models)}"
        )
    
    return model

def load_model(model_path):
    """
    加载保存的模型
    
    Args:
        model_path (str): 模型文件路径
    
    Returns:
        model (nn.Module): 加载的模型
    """
    model = torch.load(model_path)
    model.eval()
    return model

def save_model(model, save_path):
    """
    保存模型
    
    Args:
        model (nn.Module): 要保存的模型
        save_path (str): 保存路径
    """
    torch.save(model, save_path) 