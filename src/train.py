import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    MODEL_NAME, DEVICE, NUM_EPOCHS,
    LEARNING_RATE, RESULT_DIR, LABEL_NAME_PATH,
    TrainingConfig
)
from data.dataset import create_data_loaders
from models.model import get_model
from utils.visualization import plot_training_history, plot_confusion_matrix
from utils.trainer import Trainer

def load_label_names():
    """加载类别名称"""
    label_to_name = {}
    with open(LABEL_NAME_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r"'(\d+)':\s*(.*)", line.strip())
            if match:
                label_id = int(match.group(1))
                label_name = match.group(2).strip()
                if label_name.startswith("'") and label_name.endswith("'"):
                    label_name = label_name[1:-1]
                label_to_name[label_id] = label_name
    return label_to_name

def train(cfg=None):
    """
    训练函数
    
    Args:
        cfg (TrainingConfig, optional): 训练配置。如果为None，使用默认配置
    """
    # 使用默认配置或合并配置
    if cfg is None:
        cfg = TrainingConfig()
    
    # 检查CUDA是否可用并打印设备信息
    print("\n=== 设备信息 ===")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"使用设备: {cfg.device}")
    print("==============\n")
    
    # 创建实验目录
    exp_dir = cfg.create_exp_dir()
    
    # 加载类别名称
    label_to_name = load_label_names()
    num_classes = len(label_to_name)
    print(f"总类别数: {num_classes}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    model = get_model(cfg.model_name, num_classes)
    model = model.to(cfg.device)  # 确保模型在正确的设备上
    
    # 打印模型所在设备
    print(f"\n模型所在设备: {next(model.parameters()).device}")
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(cfg.device)  # 确保损失函数在正确的设备上
    
    # 选择优化器
    if cfg.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器类型: {cfg.optimizer}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=cfg.device,
        exp_dir=exp_dir,
        label_to_name=label_to_name
    )
    
    # 训练模型
    print("\n开始训练...")
    # 打印一次GPU使用情况
    if torch.cuda.is_available():
        print(f"训练开始时GPU显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    history = trainer.train(train_loader, val_loader, cfg.epochs)
    
    # 训练后再次打印GPU使用情况
    if torch.cuda.is_available():
        print(f"\n训练结束时GPU显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    # 保存模型
    trainer.save_model()
    
    # 绘制训练历史
    plot_training_history(history, exp_dir)
    
    # 在测试集上评估
    all_preds, all_labels, acc, confusion_matrix = trainer.evaluate(test_loader)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(confusion_matrix, label_to_name, exp_dir)
    
    # 从测试集中随机选择5张图片进行预测展示
    print("\n随机选择5张图片进行预测展示：")
    import random
    import numpy as np
    from torchvision.utils import save_image
    
    test_dataset = test_loader.dataset
    indices = random.sample(range(len(test_dataset)), 5)
    
    results_summary = {
        "训练配置": cfg.__dict__,
        "训练历史": history,
        "测试集准确率": float(acc),
        "示例预测": []
    }
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            img, label, img_path = test_dataset[idx]  # 修复这里，正确解包三个返回值
            img = img.unsqueeze(0).to(cfg.device)
            output = model(img)
            pred = output.argmax(dim=1).item()
            
            # 保存图片
            img_save_path = os.path.join(exp_dir, f'pred_example_{idx}.jpg')
            save_image(test_dataset[idx][0], img_save_path)
            
            # 获取真实标签和预测标签的名称
            true_label_name = label_to_name[label]
            pred_label_name = label_to_name[pred]
            
            # 打印预测结果
            result = {
                "图片索引": idx,
                "原始图片路径": img_path,  # 添加原始图片路径
                "真实类别": true_label_name,
                "预测类别": pred_label_name,
                "预测正确": pred == label,
                "预测图片路径": img_save_path
            }
            results_summary["示例预测"].append(result)
            
            print(f"图片 {idx}:")
            print(f"原始路径: {img_path}")
            print(f"真实类别: {true_label_name}")
            print(f"预测类别: {pred_label_name}")
            print(f"预测{'正确' if pred == label else '错误'}\n")
    
    # 保存训练和测试结果摘要
    summary_path = os.path.join(exp_dir, 'results_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 写入训练配置
        f.write("=== 训练配置 ===\n")
        for key, value in cfg.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 写入训练历史
        f.write("=== 训练历史 ===\n")
        f.write("轮次  训练损失  验证损失  训练精度  验证精度\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"{epoch+1:3d}   {history['train_loss'][epoch]:.4f}    {history['val_loss'][epoch]:.4f}    "
                   f"{history['train_acc'][epoch]:.4f}    {history['val_acc'][epoch]:.4f}\n")
        f.write("\n")
        
        # 写入测试集结果
        f.write("=== 测试集结果 ===\n")
        f.write(f"测试集准确率: {acc:.4f}\n\n")
        
        # 写入示例预测结果
        f.write("=== 预测示例 ===\n")
        for i, pred in enumerate(results_summary["示例预测"], 1):
            f.write(f"示例 {i}:\n")
            f.write(f"原始图片路径: {pred['原始图片路径']}\n")
            f.write(f"真实类别: {pred['真实类别']}\n")
            f.write(f"预测类别: {pred['预测类别']}\n")
            f.write(f"预测结果: {'正确' if pred['预测正确'] else '错误'}\n")
            f.write(f"预测图片保存于: {pred['预测图片路径']}\n")
            f.write("\n")
    
    print(f"\n训练和测试结果摘要已保存至: {summary_path}")
    
    print("训练和评估完成！")
    return trainer  # 返回训练器以便进一步使用

if __name__ == "__main__":
    # 示例：使用自定义配置，展示所有可调参数
    cfg = TrainingConfig(
        # 数据路径相关
        data_root="F:/competition/sort_data",  # 数据根目录
        
        # 训练参数
        batch_size=32,  # 增大批次大小
        epochs=20,  # 训练轮数
        learning_rate=0.001,  # 学习率
        num_workers=4,  # 增加数据加载线程数
        pin_memory=True,  # 启用内存固定，可加速GPU训练
        
        # 模型参数
        model_name="resnet50",  # 模型名称
        image_size=(224, 224),  # 输入图像大小
        optimizer='SGD',  # 优化器类型：'Adam' 或 'SGD'
        
        # 数据预处理参数
        normalize_mean=[0.485, 0.456, 0.406],  # 归一化均值
        normalize_std=[0.229, 0.224, 0.225],  # 归一化标准差
        
        # 结果保存相关
        project='runs/train',  # 项目保存目录
        #name='exp2',  # 实验名称（可选，默认自动生成）
        resume=False,  # 是否从断点继续训练
        
        # 设备配置
        device='cuda:0' if torch.cuda.is_available() else 'cpu'  # 使用GPU或CPU
    )
    
    # 开始训练
    trainer = train(cfg) 