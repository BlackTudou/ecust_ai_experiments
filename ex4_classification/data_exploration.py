"""
鸢尾花数据集数据加载与探索
使用 Scikit-learn、Pandas 和 Matplotlib 进行数据分析和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 使用 Scikit-learn 加载鸢尾花数据集
iris = load_iris()

# 将数据转换为 DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data['target_name'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 通过 Pandas 查看数据基本信息
print("=" * 60)
print("数据基本信息")
print("=" * 60)
print(f"数据形状: {data.shape}")
print("\n数据集基本信息:")
print(data.info())
print("\n缺失值统计:")
print(data.isnull().sum())
print("\n统计描述:")
print(data.describe())

# 准备特征名称映射
feature_names = iris.feature_names
feature_names_cn = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
feature_names_short = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 创建简化的特征名称列用于分析
for i, (short_name, cn_name) in enumerate(zip(feature_names_short, feature_names_cn)):
    data[short_name] = data[feature_names[i]]

# 使用 Matplotlib 绘制特征分布直方图
print("\n" + "=" * 60)
print("绘制特征分布直方图")
print("=" * 60)
fig = plt.figure(figsize=(14, 10))

for i, (feature, feature_cn) in enumerate(zip(feature_names_short, feature_names_cn), 1):
    plt.subplot(2, 2, i)
    for target_name in iris.target_names:
        species_data = data[data['target_name'] == target_name][feature]
        plt.hist(species_data, alpha=0.6, label=target_name, bins=15)
    plt.xlabel(feature_cn)
    plt.ylabel('频数')
    plt.title(f'{feature_cn}分布直方图')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
output_path1 = os.path.join(script_dir, 'iris_histogram.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"直方图已保存为: {output_path1}")
plt.show()

# 使用 Matplotlib 绘制散点图
print("\n" + "=" * 60)
print("绘制特征散点图")
print("=" * 60)
fig2 = plt.figure(figsize=(14, 10))

# 花萼长度 vs 花萼宽度
plt.subplot(2, 2, 1)
for target_name in iris.target_names:
    species_data = data[data['target_name'] == target_name]
    plt.scatter(species_data['sepal_length'], species_data['sepal_width'],
                label=target_name, alpha=0.6, s=50)
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title('花萼长度 vs 花萼宽度')
plt.legend()
plt.grid(True, alpha=0.3)

# 花瓣长度 vs 花瓣宽度
plt.subplot(2, 2, 2)
for target_name in iris.target_names:
    species_data = data[data['target_name'] == target_name]
    plt.scatter(species_data['petal_length'], species_data['petal_width'],
                label=target_name, alpha=0.6, s=50)
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')
plt.title('花瓣长度 vs 花瓣宽度')
plt.legend()
plt.grid(True, alpha=0.3)

# 花萼长度 vs 花瓣长度
plt.subplot(2, 2, 3)
for target_name in iris.target_names:
    species_data = data[data['target_name'] == target_name]
    plt.scatter(species_data['sepal_length'], species_data['petal_length'],
                label=target_name, alpha=0.6, s=50)
plt.xlabel('花萼长度')
plt.ylabel('花瓣长度')
plt.title('花萼长度 vs 花瓣长度')
plt.legend()
plt.grid(True, alpha=0.3)

# 花萼宽度 vs 花瓣宽度
plt.subplot(2, 2, 4)
for target_name in iris.target_names:
    species_data = data[data['target_name'] == target_name]
    plt.scatter(species_data['sepal_width'], species_data['petal_width'],
                label=target_name, alpha=0.6, s=50)
plt.xlabel('花萼宽度')
plt.ylabel('花瓣宽度')
plt.title('花萼宽度 vs 花瓣宽度')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = os.path.join(script_dir, 'iris_scatter.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"散点图已保存为: {output_path2}")
plt.show()

# 分析特征与类别的关系
print("\n" + "=" * 60)
print("特征与类别关系分析")
print("=" * 60)
mean_by_species = data.groupby('target_name', observed=True)[feature_names_short].mean()
print("\n各类别特征均值:")
print(mean_by_species)

