import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import re
import matplotlib
from matplotlib.ticker import LinearLocator
import seaborn as sns
import platform

# 根据操作系统设置合适的中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
elif platform.system() == 'Linux':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置 Seaborn 样式
sns.set_style("whitegrid")  # 使用seaborn的whitegrid样式
sns.set_context("notebook", font_scale=1.2)  # 设置字体大小

def plot_training_history(history, result_dir):
    """
    绘制训练历史图表
    
    Args:
        history (dict): 训练历史数据
        result_dir (str): 结果保存目录
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)

    epochs = range(1, len(history['train_loss']) + 1)  # 从1开始的轮次

    # 绘制损失曲线
    ax1.plot(epochs, history['train_loss'], 'o-', label='训练损失', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'o-', label='验证损失', linewidth=2, markersize=6)
    ax1.set_title('损失曲线', fontsize=14, pad=15, fontweight='bold')
    ax1.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(epochs)
    
    # 设置y轴范围
    max_loss = max(max(history['train_loss']), max(history['val_loss']))
    ax1.set_ylim(0, max_loss * 1.1)  # 设置y轴范围，留出一些空间

    # 绘制精度曲线
    ax2.plot(epochs, history['train_acc'], 'o-', label='训练精度', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'], 'o-', label='验证精度', linewidth=2, markersize=6)
    ax2.set_title('精度曲线', fontsize=14, pad=15, fontweight='bold')
    ax2.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('精度', fontsize=12)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(epochs)
    
    # 设置精度y轴范围
    ax2.set_ylim(-0.05, 1.05)

    # 调整布局
    plt.tight_layout(pad=2.0)

    # 保存高质量图表
    history_file = os.path.join(result_dir, 'training_history.png')
    plt.savefig(history_file, dpi=300, bbox_inches='tight')
    print(f"已保存训练历史图表: {history_file}")
    plt.close()


def plot_confusion_matrix(confusion_matrix, label_to_name, result_dir, top_k=10):
    """
    绘制混淆矩阵
    
    Args:
        confusion_matrix (np.ndarray): 混淆矩阵
        label_to_name (dict): 标签到类别名称的映射
        result_dir (str): 结果保存目录
        top_k (int): 显示前k个类别
    """
    # 选择混淆矩阵中最常见的top_k类别
    total_per_class = np.sum(confusion_matrix, axis=1)
    top_classes = np.argsort(total_per_class)[-top_k:]

    cm_subset = confusion_matrix[top_classes, :][:, top_classes]
    labels = [label_to_name[i] for i in top_classes]

    plt.figure(figsize=(12, 10), dpi=100)
    plt.imshow(cm_subset, interpolation='nearest', cmap='Blues')
    plt.title(f'前{top_k}类的混淆矩阵', fontsize=16, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))

    # 由于中文标签可能很长，调整旋转角度和字体大小
    plt.xticks(tick_marks, labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)

    # 添加数值
    thresh = cm_subset.max() / 2.
    for i in range(cm_subset.shape[0]):
        for j in range(cm_subset.shape[1]):
            plt.text(j, i, format(cm_subset[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_subset[i, j] > thresh else "black",
                     fontsize=8)

    plt.tight_layout()
    plt.ylabel('真实类别', fontsize=14)
    plt.xlabel('预测类别', fontsize=14)

    # 保存图表
    confusion_file = os.path.join(result_dir, 'confusion_matrix.png')
    plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
    print(f"已保存混淆矩阵图表: {confusion_file}")
    plt.close()


def predict_and_display_image(model, image_path, transform, save_path, label_to_name):
    """
    预测单张图片并显示结果
    
    Args:
        model (nn.Module): 模型
        image_path (str): 图片路径
        transform (callable): 数据转换
        save_path (str): 结果保存路径
        label_to_name (dict): 标签到类别名称的映射
    
    Returns:
        pred_idx (int): 预测的类别索引
        true_label (int): 真实标签
        pred_class (str): 预测的类别名称
        prob (float): 预测概率
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_display = img.copy()

    # 转换图像
    img_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    pred_idx = preds.item()
    pred_class = label_to_name[pred_idx]
    prob = probabilities[0, pred_idx].item() * 100

    # 获取真实标签
    label_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    label_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'labels', label_file)

    true_label = None
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            match = re.search(r'\d+', content)
            if match:
                true_label = int(match.group())
    except:
        pass

    # 显示图像和预测结果
    plt.figure(figsize=(8, 6))
    plt.imshow(img_display)
    plt.axis('off')

    result_title = f"预测类别: {pred_class} ({prob:.2f}%)"
    if true_label is not None:
        true_class = label_to_name[true_label]
        result_title += f"\n真实类别: {true_class}"
        if pred_idx == true_label:
            result_title += " ✓"
        else:
            result_title += " ✗"

    plt.title(result_title, fontsize=12)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return pred_idx, true_label, pred_class, prob
