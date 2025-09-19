import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
    VAL_IMG_DIR, VAL_LABEL_DIR,
    TEST_IMG_DIR, TEST_LABEL_DIR,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
)


class ImageClassificationDataset(Dataset):
    """图像分类数据集类"""
    
    def __init__(self, img_dir, label_dir, transform=None):
        """
        初始化数据集
        
        Args:
            img_dir (str): 图像目录路径
            label_dir (str): 标签目录路径
            transform (callable, optional): 数据转换
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 获取所有图片文件名
        self.img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.img_files.sort()  # 确保顺序一致
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            # 读取图像
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            # 读取标签
            label_file = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_file)
            
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 尝试匹配 label: 数字 格式
                match = re.search(r'label:?\s*(\d+)', content)
                if match:
                    label = int(match.group(1))
                else:
                    # 如果匹配失败，尝试提取任何数字
                    match = re.search(r'\d+', content)
                    if match:
                        label = int(match.group())
                    else:
                        print(f"警告: 无法解析标签文件 {label_path}, 内容: {content}")
                        label = 0  # 默认值
            
            # 应用转换
            if self.transform:
                image = self.transform(image)
                
            return image, label, img_path
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            # 返回一个空图像和默认标签
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, 0, ""

def get_data_transforms():
    """获取数据转换"""
    return {
        'train': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    }

def create_data_loaders():
    """创建数据加载器"""
    # 获取数据转换
    data_transforms = get_data_transforms()
    
    # 创建数据集
    train_dataset = ImageClassificationDataset(
        TRAIN_IMG_DIR, TRAIN_LABEL_DIR,
        transform=data_transforms['train']
    )
    val_dataset = ImageClassificationDataset(
        VAL_IMG_DIR, VAL_LABEL_DIR,
        transform=data_transforms['val']
    )
    test_dataset = ImageClassificationDataset(
        TEST_IMG_DIR, TEST_LABEL_DIR,
        transform=data_transforms['test']
    )
    
    # 创建数据加载器，启用pin_memory以加速GPU训练
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True  # 强制启用pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True  # 强制启用pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True  # 强制启用pin_memory
    )
    
    return train_loader, val_loader, test_loader 