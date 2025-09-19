import os
import sys
import glob
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# 设置matplotlib支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 可能无法正确显示中文")

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文物材质分类软件")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        # 检查CUDA可用性
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        self.output_dir = "predictions"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 初始化变量
        self.model = None
        self.label_to_name = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 创建界面
        self.create_widgets()
        
        # 自动查找最新的模型
        self.find_latest_model()

    def create_widgets(self):
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建顶部控制区域
        control_frame = tk.LabelFrame(main_frame, text="模型和图像选择", padx=10, pady=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 第一行 - 模型选择
        model_frame = tk.Frame(control_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(model_frame, text="模型文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.model_path_var = tk.StringVar()
        self.model_path_entry = tk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        self.model_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_model_btn = tk.Button(model_frame, text="浏览...", command=self.browse_model)
        self.browse_model_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.load_model_btn = tk.Button(model_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # 第二行 - 标签文件选择
        label_frame = tk.Frame(control_frame)
        label_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(label_frame, text="标签文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.label_path_var = tk.StringVar()
        self.label_path_var.set("F:/competition/sort_data/label_name.txt")
        self.label_path_entry = tk.Entry(label_frame, textvariable=self.label_path_var, width=50)
        self.label_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_label_btn = tk.Button(label_frame, text="浏览...", command=self.browse_label)
        self.browse_label_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 第三行 - 图像选择
        image_frame = tk.Frame(control_frame)
        image_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(image_frame, text="图像文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.image_path_var = tk.StringVar()
        self.image_path_entry = tk.Entry(image_frame, textvariable=self.image_path_var, width=50)
        self.image_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_image_btn = tk.Button(image_frame, text="浏览...", command=self.browse_image)
        self.browse_image_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.predict_btn = tk.Button(image_frame, text="预测", command=self.predict_image, state=tk.DISABLED)
        self.predict_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # 第四行 - 批量预测
        batch_frame = tk.Frame(control_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(batch_frame, text="批量处理:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.dir_path_var = tk.StringVar()
        self.dir_path_entry = tk.Entry(batch_frame, textvariable=self.dir_path_var, width=50)
        self.dir_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        self.browse_dir_btn = tk.Button(batch_frame, text="浏览...", command=self.browse_directory)
        self.browse_dir_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.batch_predict_btn = tk.Button(batch_frame, text="批量预测", command=self.batch_predict, state=tk.DISABLED)
        self.batch_predict_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = tk.Label(control_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建内容区域
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧 - 原始图像
        self.left_frame = tk.LabelFrame(content_frame, text="原始图像", padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_label = tk.Label(self.left_frame, text="请选择图像")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧 - 预测结果
        self.right_frame = tk.LabelFrame(content_frame, text="预测结果", padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(self.right_frame, height=20, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, 
                                      length=100, mode='determinate', 
                                      variable=self.progress_var)
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # 确保列宽度可调整
        model_frame.columnconfigure(1, weight=1)
        label_frame.columnconfigure(1, weight=1)
        image_frame.columnconfigure(1, weight=1)
        batch_frame.columnconfigure(1, weight=1)
        
    def find_latest_model(self):
        try:
            # 查找最新的实验目录
            exp_dirs = glob.glob("result/exp*")
            if not exp_dirs:
                self.status_var.set("未找到实验目录，请手动选择模型文件")
                return
            
            # 获取目录数字编号
            exp_numbers = []
            for dir_name in exp_dirs:
                try:
                    exp_num = int(os.path.basename(dir_name)[3:])
                    exp_numbers.append((exp_num, dir_name))
                except ValueError:
                    continue
            
            if not exp_numbers:
                self.status_var.set("未找到有效的实验目录，请手动选择模型文件")
                return
            
            # 选择最新的实验目录
            latest_exp = max(exp_numbers, key=lambda x: x[0])[1]
            model_path = os.path.join(latest_exp, "image_classification_model.pth")
            
            if os.path.exists(model_path):
                self.model_path_var.set(model_path)
                self.status_var.set(f"找到最新模型：{model_path}")
            else:
                self.status_var.set(f"在最新实验目录中未找到模型文件")
        except Exception as e:
            self.status_var.set(f"查找模型时出错：{str(e)}")
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch 模型", "*.pth"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def browse_label(self):
        filename = filedialog.askopenfilename(
            title="选择标签文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if filename:
            self.label_path_var.set(filename)
    
    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp"), 
                ("JPEG 文件", "*.jpg *.jpeg"),
                ("PNG 文件", "*.png"),
                ("BMP 文件", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.image_path_var.set(filename)
            self.display_image(filename)
    
    def browse_directory(self):
        dirname = filedialog.askdirectory(title="选择包含图像的文件夹")
        if dirname:
            self.dir_path_var.set(dirname)
    
    def load_model(self):
        model_path = self.model_path_var.get().strip()
        label_path = self.label_path_var.get().strip()
        
        if not model_path:
            messagebox.showerror("错误", "请选择模型文件")
            return
        
        if not label_path:
            messagebox.showerror("错误", "请选择标签文件")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"模型文件不存在：{model_path}")
            return
        
        if not os.path.exists(label_path):
            messagebox.showerror("错误", f"标签文件不存在：{label_path}")
            return
        
        # 禁用加载按钮
        self.load_model_btn.config(state=tk.DISABLED)
        self.status_var.set("正在加载模型...")
        self.progress_var.set(0)
        
        # 在线程中加载模型
        threading.Thread(target=self._load_model_thread, args=(model_path, label_path)).start()
    
    def _load_model_thread(self, model_path, label_path):
        try:
            # 加载标签文件
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            self.label_to_name = self.load_label_names(label_path)
            num_classes = len(self.label_to_name)
            
            if num_classes == 0:
                self.root.after(0, lambda: messagebox.showerror("错误", "标签文件中未找到有效的类别"))
                self.root.after(0, lambda: self.status_var.set("加载失败：标签文件无效"))
                self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL))
                return
            
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            # 加载模型
            try:
                # 尝试直接加载模型
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 检查加载的是模型还是状态字典
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # 加载的是带有state_dict的检查点
                    model = models.resnet18(weights=None)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict):
                    # 加载的是状态字典
                    model = models.resnet18(weights=None)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                    model.load_state_dict(checkpoint)
                else:
                    # 加载的是整个模型对象
                    model = checkpoint
                
                model = model.to(self.device)
                model.eval()
                
                self.model = model
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"加载模型时出错：{str(e)}"))
                self.root.after(0, lambda: self.status_var.set(f"加载失败：{str(e)}"))
                self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL))
                return
            
            self.progress_var.set(100)
            self.root.update_idletasks()
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set(f"模型加载成功，共 {num_classes} 个类别"))
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.batch_predict_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"加载过程中出错：{str(e)}"))
            self.root.after(0, lambda: self.status_var.set(f"加载失败：{str(e)}"))
            self.root.after(0, lambda: self.load_model_btn.config(state=tk.NORMAL))
    
    def load_label_names(self, label_file_path):
        label_to_name = {}
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r"'(\d+)':\s*(.*)", line.strip())
                if match:
                    label_id = int(match.group(1))
                    label_name = match.group(2).strip()
                    if label_name.startswith("'") and label_name.endswith("'"):
                        label_name = label_name[1:-1]
                    label_to_name[label_id] = label_name
        
        return label_to_name
    
    def display_image(self, image_path):
        try:
            # 打开图像
            img = Image.open(image_path)
            
            # 计算调整大小的比例，以适应显示区域
            display_width = self.left_frame.winfo_width() - 20
            display_height = self.left_frame.winfo_height() - 20
            
            if display_width <= 1 or display_height <= 1:
                # 框架可能还没有正确的大小
                display_width = 300
                display_height = 300
            
            # 保持宽高比例
            img_width, img_height = img.size
            ratio = min(display_width/img_width, display_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # 调整图像大小
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter可用的格式
            photo = ImageTk.PhotoImage(img)
            
            # 更新标签
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用以防止被垃圾回收
            
        except Exception as e:
            messagebox.showerror("错误", f"无法显示图像：{str(e)}")
    
    def predict_image(self):
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        image_path = self.image_path_var.get().strip()
        if not image_path:
            messagebox.showerror("错误", "请选择图像文件")
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("错误", f"图像文件不存在：{image_path}")
            return
        
        # 禁用预测按钮
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set("正在预测...")
        self.progress_var.set(0)
        
        # 清空结果
        self.results_text.delete(1.0, tk.END)
        
        # 在线程中进行预测
        threading.Thread(target=self._predict_image_thread, args=(image_path,)).start()
    
    def _predict_image_thread(self, image_path):
        try:
            # 加载图像
            self.progress_var.set(10)
            img = Image.open(image_path).convert('RGB')
            
            # 预处理图像
            self.progress_var.set(30)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # 预测
            self.progress_var.set(50)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                values, indices = torch.topk(probabilities, 5)  # 获取前5个预测结果
            
            self.progress_var.set(70)
            
            # 获取预测结果
            results = []
            for i in range(min(5, len(indices[0]))):
                idx = indices[0, i].item()
                prob = values[0, i].item() * 100
                class_name = self.label_to_name.get(idx, f"未知类别({idx})")
                results.append((idx, class_name, prob))
            
            # 创建预测结果可视化
            base_name = os.path.basename(image_path)
            output_path = os.path.join(self.output_dir, base_name)
            self.create_prediction_visualization(img, results, output_path)
            
            self.progress_var.set(100)
            
            # 更新结果文本
            result_text = "预测结果：\n\n"
            for i, (idx, class_name, prob) in enumerate(results):
                result_text += f"{i+1}. {class_name}: {prob:.2f}%\n"
            
            result_text += f"\n预测图表已保存到：\n{output_path}"
            
            self.root.after(0, lambda: self.results_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.results_text.insert(tk.END, result_text))
            self.root.after(0, lambda: self.status_var.set(f"预测完成，最可能的类别: {results[0][1]} ({results[0][2]:.2f}%)"))
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"预测过程中出错：{str(e)}"))
            self.root.after(0, lambda: self.status_var.set(f"预测失败：{str(e)}"))
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
    
    def create_prediction_visualization(self, img, results, output_path):
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原图
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title("输入图像", fontsize=14)
        
        # 显示预测结果
        class_names = [name for _, name, _ in results]
        probabilities = [prob for _, _, prob in results]
        
        y_pos = np.arange(len(class_names))
        ax2.barh(y_pos, probabilities, align='center', color='skyblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(class_names)
        ax2.invert_yaxis()  # 最高值在顶部
        ax2.set_xlabel('置信度 (%)')
        ax2.set_title('预测结果')
        
        # 添加百分比标签
        for i, prob in enumerate(probabilities):
            ax2.text(prob + 1, i, f"{prob:.2f}%", va='center')
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    def batch_predict(self):
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        dir_path = self.dir_path_var.get().strip()
        if not dir_path:
            messagebox.showerror("错误", "请选择图像文件夹")
            return
        
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            messagebox.showerror("错误", f"文件夹不存在：{dir_path}")
            return
        
        # 查找所有图像文件
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(glob.glob(os.path.join(dir_path, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(dir_path, f"*.{ext.upper()}")))
        
        if not image_files:
            messagebox.showerror("错误", f"在文件夹中未找到图像文件：{dir_path}")
            return
        
        # 禁用按钮
        self.batch_predict_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set(f"正在批量预测 {len(image_files)} 个文件...")
        self.progress_var.set(0)
        
        # 清空结果
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"开始处理 {len(image_files)} 个图像文件...\n\n")
        
        # 在线程中进行批量预测
        threading.Thread(target=self._batch_predict_thread, args=(image_files,)).start()
    
    def _batch_predict_thread(self, image_files):
        try:
            # 批量预测
            total_files = len(image_files)
            success_count = 0
            results_summary = []
            
            for i, image_path in enumerate(image_files):
                try:
                    # 更新进度
                    progress = (i / total_files) * 100
                    self.progress_var.set(progress)
                    
                    # 更新状态
                    status_msg = f"正在处理 [{i+1}/{total_files}]: {os.path.basename(image_path)}"
                    self.root.after(0, lambda msg=status_msg: self.status_var.set(msg))
                    
                    # 加载和预测图像
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        values, indices = torch.topk(probabilities, 3)
                    
                    # 获取预测结果
                    top_idx = indices[0, 0].item()
                    top_prob = values[0, 0].item() * 100
                    top_class = self.label_to_name.get(top_idx, f"未知类别({top_idx})")
                    
                    # 创建结果可视化
                    base_name = os.path.basename(image_path)
                    output_path = os.path.join(self.output_dir, base_name)
                    
                    # 收集所有预测结果
                    results = []
                    for j in range(min(3, len(indices[0]))):
                        idx = indices[0, j].item()
                        prob = values[0, j].item() * 100
                        class_name = self.label_to_name.get(idx, f"未知类别({idx})")
                        results.append((idx, class_name, prob))
                    
                    # 创建预测结果可视化
                    self.create_prediction_visualization(img, results, output_path)
                    
                    # 添加到结果摘要
                    filename = os.path.basename(image_path)
                    results_summary.append((filename, top_class, top_prob, output_path))
                    success_count += 1
                    
                    # 更新结果文本
                    self.root.after(0, lambda msg=f"[{i+1}/{total_files}] {filename}: {top_class} ({top_prob:.2f}%)\n": 
                                   self.results_text.insert(tk.END, msg))
                    
                except Exception as e:
                    error_msg = f"[{i+1}/{total_files}] 处理 {os.path.basename(image_path)} 失败: {str(e)}\n"
                    self.root.after(0, lambda msg=error_msg: self.results_text.insert(tk.END, msg))
            
            # 完成处理
            self.progress_var.set(100)
            summary_msg = f"\n处理完成！成功: {success_count}/{total_files}\n\n"
            summary_msg += f"所有预测结果已保存到: {os.path.abspath(self.output_dir)}"
            
            self.root.after(0, lambda msg=summary_msg: self.results_text.insert(tk.END, msg))
            self.root.after(0, lambda: self.status_var.set(f"批量预测完成：成功 {success_count}/{total_files}"))
            self.root.after(0, lambda: self.batch_predict_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"批量预测过程中出错：{str(e)}"))
            self.root.after(0, lambda: self.status_var.set(f"批量预测失败：{str(e)}"))
            self.root.after(0, lambda: self.batch_predict_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop() 