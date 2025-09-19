@echo off
chcp 65001 > nul
title 材质图像分类软件

echo 正在启动材质图像分类软件...
echo 如果程序没有自动启动，请确保已安装Python及相关依赖包。

"D:\anaconda3\envs\MaterialClassifier\python.exe" material_classifier_app.py

if %errorlevel% neq 0 (
    echo.
    echo 启动失败！请确保已正确安装以下包：
    echo - Python 3.6+
    echo - PyTorch
    echo - torchvision
    echo - matplotlib
    echo - numpy
    echo - Pillow
    echo.
    echo 您可以使用以下命令安装依赖：
    echo pip install torch torchvision matplotlib numpy pillow
    echo.
    pause
) 