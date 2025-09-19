import os
import torch

class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, **kwargs):
        # 数据路径
        self.data_root = kwargs.get('data_root', "F:/competition/sort_data")
        self.train_img_dir = os.path.join(self.data_root, "train/images")
        self.train_label_dir = os.path.join(self.data_root, "train/labels")
        self.val_img_dir = os.path.join(self.data_root, "validation/images")
        self.val_label_dir = os.path.join(self.data_root, "validation/labels")
        self.test_img_dir = os.path.join(self.data_root, "test/images")
        self.test_label_dir = os.path.join(self.data_root, "test/labels")
        self.label_name_path = os.path.join(self.data_root, "label_name.txt")
        
        # 训练参数
        self.batch_size = kwargs.get('batch_size', 8)
        self.epochs = kwargs.get('epochs', 30)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory', False)
        
        # 模型参数
        self.model_name = kwargs.get('model_name', "resnet18")
        self.image_size = kwargs.get('image_size', (224, 224))
        self.optimizer = kwargs.get('optimizer', 'Adam')
        
        # 数据预处理参数
        self.normalize_mean = kwargs.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = kwargs.get('normalize_std', [0.229, 0.224, 0.225])
        
        # 结果保存路径
        self.project = kwargs.get('project', 'result')
        self.name = kwargs.get('name', None)
        self.resume = kwargs.get('resume', False)
        
        # 设备配置
        self.device = kwargs.get('device', "cuda:0" if torch.cuda.is_available() else "cpu")
    
    def create_exp_dir(self):
        """创建实验目录"""
        os.makedirs(self.project, exist_ok=True)
        
        if self.name is None:
            exp_folders = [f for f in os.listdir(self.project) if f.startswith("exp")]
            next_exp_number = len(exp_folders)
            self.name = f"exp{next_exp_number}"
        
        exp_dir = os.path.join(self.project, self.name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

# 默认配置
MODEL_NAME = "resnet18"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
RESULT_DIR = "result"
LABEL_NAME_PATH = "F:/competition/sort_data/label_name.txt"

# 数据路径
DATA_ROOT = "F:/competition/sort_data"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train/images")
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "train/labels")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "validation/images")
VAL_LABEL_DIR = os.path.join(DATA_ROOT, "validation/labels")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test/images")
TEST_LABEL_DIR = os.path.join(DATA_ROOT, "test/labels")

# 训练参数
BATCH_SIZE = 8
NUM_WORKERS = 0
PIN_MEMORY = False

# 模型参数
IMAGE_SIZE = (224, 224)

# 数据预处理参数
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225] 