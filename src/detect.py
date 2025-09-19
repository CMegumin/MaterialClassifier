import os
import re
import argparse
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import DEVICE, RESULT_DIR, LABEL_NAME_PATH
from src.data.dataset import get_data_transforms
from src.models.model import load_model
from src.utils.visualization import predict_and_display_image
from src.train import load_label_names

def predict_single_image(model, image_path, transform, label_to_name, save_dir):
    """
    预测单张图片
    
    Args:
        model (nn.Module): 模型
        image_path (str): 图片路径
        transform (callable): 数据转换
        label_to_name (dict): 标签到类别名称的映射
        save_dir (str): 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成保存路径
    save_name = os.path.splitext(os.path.basename(image_path))[0] + '_pred.png'
    save_path = os.path.join(save_dir, save_name)
    
    # 预测并保存结果
    pred_idx, true_label, pred_class, prob = predict_and_display_image(
        model, image_path, transform, save_path, label_to_name
    )
    
    print(f"\n预测结果:")
    print(f"图片: {image_path}")
    print(f"预测类别: {pred_class}")
    print(f"预测概率: {prob:.2f}%")
    if true_label is not None:
        print(f"真实类别: {label_to_name[true_label]}")
        print(f"预测{'正确' if pred_idx == true_label else '错误'}")
    print(f"结果已保存到: {save_path}")

def predict_batch(model, image_dir, transform, label_to_name, save_dir):
    """
    批量预测图片
    
    Args:
        model (nn.Module): 模型
        image_dir (str): 图片目录
        transform (callable): 数据转换
        label_to_name (dict): 标签到类别名称的映射
        save_dir (str): 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查目录是否为空
    if not os.path.isdir(image_dir):
        print(f"错误: {image_dir} 不是一个有效的目录")
        return
        
    if not os.listdir(image_dir):
        print(f"错误: 目录 {image_dir} 是空的")
        return
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"警告: 在 {image_dir} 中没有找到支持的图片文件")
        print("支持的格式: .png, .jpg, .jpeg")
        return
    
    print(f"\n开始批量预测 {len(image_files)} 张图片...")
    
    # 统计结果
    correct = 0
    total = 0
    results = []
    
    # 批量预测
    for img_file in tqdm(image_files, desc="预测进度"):
        image_path = os.path.join(image_dir, img_file)
        save_name = os.path.splitext(img_file)[0] + '_pred.png'
        save_path = os.path.join(save_dir, save_name)
        
        try:
            pred_idx, true_label, pred_class, prob = predict_and_display_image(
                model, image_path, transform, save_path, label_to_name
            )
            
            results.append({
                'image': img_file,
                'prediction': pred_class,
                'probability': prob,
                'true_label': label_to_name[true_label] if true_label is not None else 'unknown'
            })
            
            if true_label is not None:
                total += 1
                if pred_idx == true_label:
                    correct += 1
                    
        except Exception as e:
            print(f"\n处理图片 {img_file} 时出错: {str(e)}")
            continue
    
    # 打印统计结果
    print("\n预测完成!")
    print(f"处理图片总数: {len(image_files)}")
    if total > 0:
        accuracy = correct / total * 100
        print(f"准确率: {accuracy:.2f}% ({correct}/{total})")
    
    # 保存预测结果到文本文件
    result_file = os.path.join(save_dir, 'prediction_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("预测结果汇总:\n")
        f.write("-" * 50 + "\n")
        for result in results:
            f.write(f"图片: {result['image']}\n")
            f.write(f"预测类别: {result['prediction']}\n")
            f.write(f"预测概率: {result['probability']:.2f}%\n")
            f.write(f"真实类别: {result['true_label']}\n")
            f.write("-" * 50 + "\n")
        
        if total > 0:
            f.write(f"\n准确率: {accuracy:.2f}% ({correct}/{total})")
    
    print(f"详细结果已保存到: {result_file}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图像分类预测')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图片路径或目录')
    parser.add_argument('--output', type=str, default='predictions', help='输出目录')
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return
    
    # 检查输入路径是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入路径 {args.input} 不存在")
        return
    
    # 加载模型
    try:
        print("加载模型...")
        model = load_model(args.model)
        model = model.to(DEVICE)
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 获取数据转换
    transform = get_data_transforms()['test']
    
    # 加载类别名称
    label_to_name = load_label_names()
    
    # 根据输入类型选择预测模式
    if os.path.isfile(args.input):
        # 单张图片预测
        predict_single_image(
            model, args.input, transform,
            label_to_name, args.output
        )
    else:
        # 批量预测
        predict_batch(
            model, args.input, transform,
            label_to_name, args.output
        )

if __name__ == '__main__':
    main() 