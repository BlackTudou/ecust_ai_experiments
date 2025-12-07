# 实验5：回归分析 - 波士顿房价预测

## 实验目标

本实验使用波士顿房价数据集，实现多元线性回归、非线性回归（SVM）以及正则化回归（Lasso和Ridge），并对模型性能进行全面评估。

## 实验内容

1. **数据准备**：使用 Scikit-learn 库的波士顿房价数据集
2. **数据探索与预处理**：
   - 查看数据分布
   - 分析特征间相关性
   - 处理缺失值、异常值
   - 划分训练集和测试集
   - 对特征进行标准化
3. **多元线性回归实现**：
   - 使用所有特征构建多元线性回归模型
   - 训练模型
   - 分析各特征的权重系数
4. **多元非线性回归实现**：
   - 使用SVM构建回归模型
   - 训练模型
   - 与原始数据对比
5. **模型评估**：
   - 计算训练集和测试集的 MSE、MAE、R²
   - 分析模型拟合效果
   - 判断是否存在过拟合或欠拟合
6. **正则化应用**：
   - 分别构建 Lasso 回归和 Ridge 回归模型
   - 调整正则化参数 λ
   - 对比分析正则化前后模型的性能变化
   - 观察 Lasso 回归对特征的选择效果

## 文件说明

- `boston_regression.py`: 主程序文件，包含完整的回归分析实现
- `需求文档.md`: 实验需求文档

## 运行方法

```bash
cd ex5_regression
python boston_regression.py
```

## 依赖库

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

安装依赖：
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 输出文件

程序运行后会生成以下可视化图表：

1. `data_distribution.png` - 数据分布直方图
2. `feature_correlation.png` - 特征相关性热力图
3. `linear_regression_coefficients.png` - 线性回归特征权重系数图
4. `prediction_comparison.png` - 预测结果对比图（线性回归 vs SVM回归）
5. `model_performance_comparison.png` - 模型性能对比图
6. `lasso_feature_selection.png` - Lasso特征选择效果图
7. `regularization_analysis.png` - 正则化参数影响分析图

## 注意事项

1. **数据集兼容性**：如果您的 scikit-learn 版本 >= 1.2，波士顿房价数据集可能已被移除。程序会自动尝试使用替代数据集或生成模拟数据。

2. **运行时间**：SVM回归模型的训练可能需要一些时间，请耐心等待。

3. **正则化参数**：程序会自动测试多个正则化参数值，并选择最佳参数。

## 实验结果分析

程序会输出详细的实验结果，包括：
- 各模型的训练集和测试集性能指标（MSE、MAE、R²）
- 过拟合/欠拟合分析
- 特征重要性分析
- 正则化效果对比
- Lasso回归的特征选择结果

## 模型对比

程序会对比以下模型：
- **线性回归**（无正则化）
- **SVM回归**（非线性，RBF核）
- **Lasso回归**（L1正则化）
- **Ridge回归**（L2正则化）

每个模型都会在训练集和测试集上进行评估，以便全面分析模型性能。

