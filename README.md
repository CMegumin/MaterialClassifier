# 文物材质分类系统

这是一个基于深度学习的文物材质分类系统，包含模型训练和图形界面应用两个主要部分。系统可以对文物图像进行材质分类，支持单张图片预测和批量预测功能。

## 项目结构

```
project/
│
├── src/                    # 源代码目录
│   ├── train.py           # 模型训练脚本
│   ├── detect.py          # 模型预测脚本
│   ├── config.py          # 配置文件
│   ├── requirements.txt    # 项目依赖
│   │
│   ├── models/            # 模型定义
│   ├── data/              # 数据加载和处理
│   ├── utils/             # 工具函数
│   └── runs/              # 训练结果和日志
│
├── 部分数据集/
|   ├── test/
│   |   ├── images/
│   |   └── labels/
|   ├── train/
│   |   ├── images/
│   |   └── labels/
|   └── validation/
|       ├── images/
|       └── labels/
|
└── 文物材质分类软件/        # GUI应用程序
    ├── material_classifier_app.py  # GUI应用主程序
    ├── run_classifier.bat         # 启动脚本
    ├── README_GUI.md             # GUI使用说明
    └── predictions/              # 预测结果保存目录
```

## 环境配置

### 系统要求

- Python 3.8+ (推荐使用Anaconda或Miniconda)
- CUDA支持的NVIDIA显卡（用于GPU训练）
- Windows/Linux/MacOS

### 创建虚拟环境

```bash
# 创建虚拟环境
conda create -n material_classifier python=3.8

# 激活环境
conda activate material_classifier

# 安装依赖
pip install -r requirements.txt
```

### 安装CUDA（可选，用于GPU训练）

如果您想使用GPU加速训练，请确保：

1. 安装了NVIDIA显卡驱动
2. 安装了与PyTorch兼容的CUDA版本
3. 使用兼容的PyTorch-CUDA版本

## 快速开始

### 1. 训练模型

```bash
# 进入源代码目录
cd src

# 使用默认配置开始训练
python train.py

# 或使用自定义配置
python train.py --batch_size 32 --epochs 20 --learning_rate 0.001
```

训练过程会自动：

- 保存训练日志和模型
- 生成训练过程可视化图表
- 在测试集上评估模型
- 保存详细的训练报告

### 2. 使用GUI应用

1. 确保已安装所有依赖
2. 进入GUI应用目录
3. 运行 `run_classifier.bat` 或执行：
   ```bash
   python material_classifier_app.py
   ```

详细的GUI使用说明请参考 `文物材质分类软件/README_GUI.md`。
