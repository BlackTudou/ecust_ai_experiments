# MNIST手写数字识别 - 基于改进LeNet-5 CNN模型

本实验使用PyTorch框架实现了一个改进的LeNet-5卷积神经网络（CNN）模型，用于MNIST手写数字识别任务。

## 实验内容

### 1. 数据集准备
- 使用MNIST手写数字数据集
- 60000张训练图像，10000张测试图像
- 每张图像为28×28灰度图
- 共10个类别（数字0-9）
- 通过PyTorch框架自带接口加载数据集

### 2. 图像数据预处理
- **数据格式转换**：将图像数据转换为PyTorch张量格式
- **归一化**：将像素值从[0,255]转换为[0,1]，加速模型训练
- **数据增强（可选）**：支持随机旋转、平移等，提高模型泛化能力

### 3. 构建CNN模型（基于LeNet-5改进）
网络结构：
- 输入层（28×28×1）
- 卷积层1（32个3×3卷积核，ReLU激活函数）
- 池化层1（2×2最大池化）
- 卷积层2（64个3×3卷积核，ReLU激活函数）
- 池化层2（2×2最大池化）
- 扁平化层
- 全连接层（128个神经元，ReLU激活函数）
- Dropout层（dropout rate=0.5，防止过拟合）
- 输出层（10个神经元，Softmax激活函数）

### 4. 模型训练与监控
- **优化器**：Adam（学习率0.001）
- **损失函数**：交叉熵损失（CrossEntropyLoss，适用于整数标签）
- **评估指标**：准确率
- **训练参数**：epochs=10，batch_size=32
- **训练监控**：记录训练准确率、验证准确率、训练损失、验证损失
- **可视化**：绘制训练曲线，分析过拟合情况

### 5. 模型评估与预测
- 在测试集上评估模型性能
- 计算测试准确率
- 生成混淆矩阵，分析不同类别数字的分类效果
- 随机选取测试集中的图像进行预测，直观观察模型预测效果

### 6. 模型优化（可选）
- 调整模型结构（如增加卷积层数量、改变卷积核大小、调整全连接层神经元数量）
- 调整训练参数（如学习率、batch_size、epochs、dropout rate）
- 应用数据增强技术，对比优化前后模型的性能变化

## 文件说明

- `model.py`: CNN模型定义文件
  - `ImprovedLeNet5`类：改进的LeNet-5模型实现
  - 包含模型信息打印功能

- `main.py`: 主程序文件
  - 数据加载与预处理
  - 模型训练与监控
  - 模型评估与预测
  - 训练过程可视化

- `optimization.py`: 模型优化模块（可选）
  - 学习率优化分析
  - 批次大小优化分析
  - Dropout比率优化分析
  - 数据增强效果对比
  - 综合优化分析

## 环境要求

```bash
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
pandas
```

安装依赖：
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas
```

## 使用方法

### 基本使用

直接运行主程序：
```bash
cd ex8_handwritten_digit_recognition
python main.py
```

### 运行优化分析

如果需要运行模型优化分析：
```bash
python optimization.py
```

或者在主程序中调用优化函数：
```python
from optimization import comprehensive_optimization
comprehensive_optimization(train_loader, test_loader, device=device)
```

## 输出结果

运行程序后会生成以下文件：

1. **训练过程可视化**：
   - `training_curves.png` - 训练准确率与验证准确率曲线、训练损失与验证损失曲线

2. **模型评估结果**：
   - `confusion_matrix.png` - 混淆矩阵，展示各类别的分类效果
   - `random_predictions.png` - 随机预测结果可视化

3. **优化分析结果**（如果运行优化模块）：
   - `learning_rate_analysis.png` - 学习率分析图
   - `learning_rate_results.csv` - 学习率实验结果
   - `batch_size_analysis.png` - 批次大小分析图
   - `dropout_analysis.png` - Dropout比率分析图
   - `data_augmentation_comparison.png` - 数据增强效果对比图

## 模型结构

### ImprovedLeNet5模型

```
输入层: 28×28×1
    ↓
卷积层1: 32个3×3卷积核 + ReLU激活
    ↓
池化层1: 2×2最大池化 → 14×14×32
    ↓
卷积层2: 64个3×3卷积核 + ReLU激活
    ↓
池化层2: 2×2最大池化 → 7×7×64
    ↓
扁平化: 3136
    ↓
全连接层: 128个神经元 + ReLU激活
    ↓
Dropout层: dropout rate=0.5
    ↓
输出层: 10个神经元（对应10个数字类别）
```

### 模型参数统计

模型包含可训练参数，具体数量会在运行时打印。可通过`model.print_model_info()`查看详细的模型结构和参数信息。

## 实验参数

### 默认训练参数
- **学习率**：0.001
- **优化器**：Adam
- **损失函数**：CrossEntropyLoss（交叉熵损失）
- **批次大小**：32
- **训练轮数**：10
- **Dropout比率**：0.5

### 数据增强参数（可选）
- 随机旋转：±10度
- 随机平移：10%范围

## 性能指标

- **测试准确率**：通常在98%以上
- **训练时间**：根据硬件配置不同，通常在几分钟到十几分钟
- **模型大小**：约几MB

## 注意事项

1. **首次运行**：程序会自动下载MNIST数据集到`./data`目录
2. **设备选择**：如果有GPU，程序会自动使用GPU加速训练
3. **随机种子**：使用固定随机种子（42）确保结果可复现
4. **Windows系统**：数据加载器的`num_workers`参数已设置为0，避免多进程问题

## 过拟合分析

程序会自动分析训练过程中的过拟合情况：
- 检查验证准确率是否在后期下降
- 检查验证损失是否在后期上升
- 提供过拟合警告信息

## 扩展功能

可以通过修改代码实现以下扩展：
1. 调整模型结构（增加或减少层数、改变神经元数量）
2. 尝试不同的优化器（SGD、RMSprop等）
3. 添加学习率调度器
4. 实现早停法（Early Stopping）
5. 添加Batch Normalization层
6. 尝试不同的数据增强方法

## 参考资料

- LeNet-5原始论文：Gradient-Based Learning Applied to Document Recognition
- PyTorch官方文档：https://pytorch.org/docs/stable/index.html
- MNIST数据集：http://yann.lecun.com/exdb/mnist/

## 作者

本实验代码根据课程需求文档实现。

