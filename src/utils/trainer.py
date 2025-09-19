import os
import time
import copy
import torch
from tqdm import tqdm


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=5):
    """
    训练模型
    
    Args:
        model (nn.Module): 要训练的模型
        dataloaders (dict): 数据加载器字典
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        num_epochs (int): 训练轮数
    
    Returns:
        model (nn.Module): 训练好的模型
        history (dict): 训练历史
        best_acc (float): 最佳验证精度
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            processed_samples = 0

            # 迭代数据
            total_batches = len(dataloaders[phase])
            data_iter = tqdm(
                dataloaders[phase],
                total=total_batches,
                desc=f"[{epoch + 1}/{num_epochs}] {phase.capitalize()}",
                unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="green" if phase == "train" else "blue",
                ncols=100,
                leave=False
            )

            try:
                for i, (inputs, labels, _) in enumerate(data_iter):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 清零参数梯度
                    optimizer.zero_grad()

                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 如果是训练阶段，则反向传播+优化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计
                    current_loss = loss.item()
                    current_corrects = torch.sum(preds == labels.data).item()
                    batch_size = inputs.size(0)

                    running_loss += current_loss * batch_size
                    running_corrects += current_corrects
                    processed_samples += batch_size

                    # 更新进度条信息
                    current_avg_loss = running_loss / processed_samples
                    current_avg_acc = running_corrects / processed_samples

                    data_iter.set_postfix(
                        loss=f"{current_avg_loss:.4f}",
                        acc=f"{current_avg_acc:.4f}",
                        batch=f"{i + 1}/{total_batches}"
                    )

            except Exception as e:
                print(f"训练过程中发生错误: {str(e)}")
                continue

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

                # 如果是最佳模型，保存模型状态
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证精度: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history, best_acc


class Trainer:
    """模型训练器类"""

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            device,
            exp_dir,
            label_to_name
    ):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 模型
            criterion: 损失函数
            optimizer: 优化器
            device: 计算设备
            exp_dir (str): 实验结果保存目录
            label_to_name (dict): 标签到类别名称的映射
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.exp_dir = exp_dir
        self.label_to_name = label_to_name

        # 记录最佳模型和精度
        self.best_model_path = os.path.join(exp_dir, 'best_model.pth')
        self.best_acc = 0.0

    def train(self, train_loader, val_loader, num_epochs):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs (int): 训练轮数
        
        Returns:
            history (dict): 训练历史
        """
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        # 训练模型
        model, history, best_acc = train_model(
            self.model,
            dataloaders,
            self.criterion,
            self.optimizer,
            self.device,
            num_epochs
        )

        # 更新最佳精度
        self.best_acc = best_acc

        return history

    def save_model(self):
        """保存当前模型"""
        torch.save(self.model, self.best_model_path)
        print(f"模型已保存到: {self.best_model_path}")

    def evaluate(self, test_loader):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            all_preds: 所有预测结果
            all_labels: 所有真实标签
            acc: 准确率
            confusion_matrix: 混淆矩阵
        """
        from utils.metrics import evaluate_model
        return evaluate_model(self.model, test_loader, self.device)
