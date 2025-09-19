import torch
import numpy as np
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    """
    在测试集上评估模型
    
    Args:
        model (nn.Module): 要评估的模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 计算设备
    
    Returns:
        all_preds (list): 所有预测结果
        all_labels (list): 所有真实标签
        acc (float): 准确率
        confusion_matrix (np.ndarray): 混淆矩阵
    """
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    num_classes = model.fc.out_features
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    print("在测试集上评估模型...")
    with torch.no_grad():
        # 改进进度条
        total_batches = len(dataloader)
        data_iter = tqdm(
            dataloader, 
            total=total_batches,
            desc="测试评估", 
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="yellow",
            ncols=100
        )
        
        for i, (inputs, labels, _) in enumerate(data_iter):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            batch_corrects = torch.sum(preds == labels.data).item()
            running_corrects += batch_corrects
            
            # 收集预测和标签
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            
            # 更新混淆矩阵
            for t, p in zip(labels_np, preds_np):
                confusion_matrix[t, p] += 1
                            
            # 计算当前准确率并更新进度条
            processed_samples = (i + 1) * dataloader.batch_size
            processed_samples = min(processed_samples, len(dataloader.dataset))
            current_acc = running_corrects / processed_samples
            
            data_iter.set_postfix(
                acc=f"{current_acc:.4f}",
                correct=f"{running_corrects}/{processed_samples}",
                batch=f"{i+1}/{total_batches}"
            )
            
    acc = running_corrects / len(dataloader.dataset)
    print(f'测试集精度: {acc:.4f}')
    
    return all_preds, all_labels, acc, confusion_matrix 