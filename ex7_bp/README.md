# BP神经网络实验

本实验实现了BP神经网络用于鸢尾花分类任务，包含手动实现和PyTorch框架实现两种方式。

## 文件说明

- `bp_manual.py`: 手动实现的BP神经网络类
  - 包含前向传播、反向传播、参数更新等核心功能
  - 隐藏层使用ReLU激活函数
  - 输出层使用Softmax激活函数（多分类）

- `bp_pytorch.py`: 使用PyTorch框架实现的BP神经网络
  - 使用Sequential模型构建网络
  - 支持Adam优化器
  - 包含早停法（Early Stopping）防止过拟合
  - 支持验证集监控

- `bp_analysis.py`: 实验分析代码
  - 分析学习率对模型性能的影响
  - 分析迭代次数对模型性能的影响
  - 分析隐藏层神经元数量对模型性能的影响

- `main.py`: 主程序
  - 整合所有功能
  - 数据加载与预处理
  - 模型训练与评估
  - 结果对比与可视化

## 环境要求

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
```

安装依赖：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

## 使用方法

直接运行主程序：

```bash
cd ex7_bp
python main.py
```

## 实验内容

### 1. 手动实现BP神经网络
- 网络结构：输入层(4) → 隐藏层(10, ReLU) → 输出层(3, Softmax)
- 实现前向传播和反向传播算法
- 使用梯度下降法更新参数
- 记录训练过程中的损失值变化

### 2. PyTorch框架实现
- 使用Sequential模型构建网络
- 配置Adam优化器
- 使用交叉熵损失函数
- 实现早停法防止过拟合
- 使用验证集监控模型性能

### 3. 实验分析
- **学习率分析**：测试不同学习率（0.001, 0.01, 0.1, 0.5）对模型性能的影响
- **迭代次数分析**：测试不同迭代次数（100, 500, 1000, 2000）对模型性能的影响
- **隐藏层神经元数量分析**：测试不同隐藏层神经元数量（5, 10, 20, 50）对模型性能的影响

## 输出结果

运行程序后会生成以下文件：

1. **模型训练结果**：
   - `manual_bp_loss_curve.png` - 手动实现损失曲线
   - `pytorch_bp_training_curves.png` - PyTorch实现训练曲线

2. **模型对比**：
   - `manual_vs_pytorch_comparison.png` - 手动实现与PyTorch实现对比
   - `confusion_matrices_comparison.png` - 混淆矩阵对比

3. **实验分析结果**：
   - `learning_rate_analysis.png` - 学习率分析图
   - `learning_rate_vs_accuracy.png` - 学习率-准确率关系图
   - `epochs_analysis.png` - 迭代次数分析图
   - `epochs_vs_accuracy.png` - 迭代次数-准确率关系图
   - `hidden_neurons_analysis.png` - 隐藏层神经元数量分析图
   - `hidden_neurons_vs_accuracy.png` - 隐藏层神经元数量-准确率关系图

4. **实验数据**：
   - `learning_rate_results.csv` - 学习率实验结果
   - `epochs_results.csv` - 迭代次数实验结果
   - `hidden_neurons_results.csv` - 隐藏层神经元数量实验结果

## 网络结构

- **输入层**：4个神经元（对应4个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度）
- **隐藏层**：10个神经元（默认，可调整），使用ReLU激活函数
- **输出层**：3个神经元（对应3个类别），使用Softmax激活函数

## 参数说明

### 手动实现参数
- `input_dim`: 输入层维度（4）
- `hidden_dim`: 隐藏层维度（10）
- `output_dim`: 输出层维度（3）
- `learning_rate`: 学习率（0.01）

### PyTorch实现参数
- `learning_rate`: 学习率（0.001）
- `optimizer_name`: 优化器名称（'Adam' 或 'SGD'）
- `batch_size`: 批次大小（16）
- `early_stopping`: 早停法参数（patience=50）

## 注意事项

1. 数据文件路径：程序会自动从 `../data/exp04_iris_data.txt` 加载数据
2. 随机种子：手动实现使用固定随机种子（42）确保结果可复现
3. 训练时间：完整实验可能需要几分钟时间，请耐心等待
4. GPU支持：PyTorch实现会自动使用GPU（如果可用），否则使用CPU

