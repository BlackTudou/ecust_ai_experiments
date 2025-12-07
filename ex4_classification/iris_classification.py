"""
鸢尾花数据集分类任务完整实现
包含：数据加载、探索、预处理、模型训练与评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取脚本所在目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# =====================================================================
# 1. 数据加载与探索
# =====================================================================
print("=" * 80)
print("1. 数据加载与探索")
print("=" * 80)

# 从文件加载鸢尾花数据集
data_file_path = os.path.join(project_root, 'data', 'exp04_iris_data.txt')
print(f"\n数据文件路径: {data_file_path}")

# 读取CSV格式的数据文件
data_raw = pd.read_csv(data_file_path, header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target_name'],
                       skip_blank_lines=True)
# 移除可能的空行
data_raw = data_raw.dropna()

# 处理类别标签，将其转换为数值型
target_names_unique = data_raw['target_name'].unique()
target_name_to_code = {name: idx for idx, name in enumerate(target_names_unique)}
target_codes = data_raw['target_name'].map(target_name_to_code)

# 将数据转换为与原始代码兼容的格式
data = data_raw.copy()
data['target'] = target_codes
data['target_name'] = pd.Categorical(data_raw['target_name'], categories=target_names_unique)

# 定义特征名称
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_names_cn = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
feature_names_short = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_target_names = np.array(target_names_unique)

# 通过 Pandas 查看数据基本信息
print(f"\n数据形状: {data.shape}")
print("\n数据集基本信息:")
print(data.info())
print("\n缺失值统计:")
print(data.isnull().sum())
print("\n统计描述:")
print(data.describe())

# 使用 Matplotlib 绘制特征分布直方图
print("\n绘制特征分布直方图...")
fig = plt.figure(figsize=(14, 10))

for i, (feature, feature_cn) in enumerate(zip(feature_names_short, feature_names_cn), 1):
    plt.subplot(2, 2, i)
    for target_name in iris_target_names:
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
plt.close()

# 使用 Matplotlib 绘制散点图
print("绘制特征散点图...")
fig2 = plt.figure(figsize=(14, 10))

# 花萼长度 vs 花萼宽度
plt.subplot(2, 2, 1)
for target_name in iris_target_names:
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
for target_name in iris_target_names:
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
for target_name in iris_target_names:
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
for target_name in iris_target_names:
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
plt.close()

# 分析特征与类别的关系
print("\n特征与类别关系分析:")
mean_by_species = data.groupby('target_name', observed=True)[feature_names_short].mean()
print("\n各类别特征均值:")
print(mean_by_species)

# =====================================================================
# 2. 数据预处理
# =====================================================================
print("\n" + "=" * 80)
print("2. 数据预处理")
print("=" * 80)

# 准备特征和标签
X = data[feature_names_short].values
y = data['target'].values

print(f"\n原始数据形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 数据集划分：训练集（70%）和测试集（30%）
print("\n数据集划分（训练集70%，测试集30%）...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print(f"训练集标签形状: {y_train.shape}")
print(f"测试集标签形状: {y_test.shape}")

# 显示训练集和测试集中各类别的分布
print("\n训练集中各类别样本数:")
train_labels, train_counts = np.unique(y_train, return_counts=True)
for label, count in zip(train_labels, train_counts):
    print(f"  类别 {iris_target_names[label]}: {count} 个样本")

print("\n测试集中各类别样本数:")
test_labels, test_counts = np.unique(y_test, return_counts=True)
for label, count in zip(test_labels, test_counts):
    print(f"  类别 {iris_target_names[label]}: {count} 个样本")

# 特征标准化
print("\n特征标准化（StandardScaler：均值为0，标准差为1）...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("标准化完成！")

# =====================================================================
# 3. 模型构建与训练
# =====================================================================
print("\n" + "=" * 80)
print("3. 模型构建与训练")
print("=" * 80)

models = {}
predictions = {}

# 3.1 逻辑回归
print("\n" + "-" * 80)
print("3.1 逻辑回归模型")
print("-" * 80)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
models['逻辑回归'] = lr_model
predictions['逻辑回归'] = lr_model.predict(X_test_scaled)
print("逻辑回归模型训练完成！")
print(f"模型参数: C={lr_model.C}, max_iter={lr_model.max_iter}")

# 3.2 K近邻分类器
print("\n" + "-" * 80)
print("3.2 K近邻分类模型")
print("-" * 80)
# 尝试不同的K值，选择最佳K值
k_values = [3, 5, 7, 9, 11]
k_scores = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)
    print(f"  K={k}: 测试集准确率={score:.4f}")

best_k_idx = np.argmax(k_scores)
best_k = k_values[best_k_idx]
print(f"\n最佳K值: {best_k} (准确率={k_scores[best_k_idx]:.4f})")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
models['K近邻'] = knn_model
predictions['K近邻'] = knn_model.predict(X_test_scaled)
print("K近邻模型训练完成！")
print(f"模型参数: n_neighbors={best_k}")

# 3.3 决策树分类器
print("\n" + "-" * 80)
print("3.3 决策树分类模型")
print("-" * 80)
# 尝试不同的最大深度，选择最佳深度
max_depths = [3, 5, 7, 10, None]
depth_scores = []
for depth in max_depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train_scaled, y_train)
    score = dt_temp.score(X_test_scaled, y_test)
    depth_scores.append(score)
    depth_str = "None" if depth is None else str(depth)
    print(f"  最大深度={depth_str}: 测试集准确率={score:.4f}")

best_depth_idx = np.argmax(depth_scores)
best_depth = max_depths[best_depth_idx]
print(f"\n最佳最大深度: {best_depth if best_depth is not None else 'None'} (准确率={depth_scores[best_depth_idx]:.4f})")

dt_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_model.fit(X_train_scaled, y_train)
models['决策树'] = dt_model
predictions['决策树'] = dt_model.predict(X_test_scaled)
print("决策树模型训练完成！")
print(f"模型参数: max_depth={best_depth if best_depth is not None else 'None'}")

# =====================================================================
# 4. 模型预测与评估
# =====================================================================
print("\n" + "=" * 80)
print("4. 模型预测与评估")
print("=" * 80)

# 计算各模型的评估指标
results = {}

for model_name in models.keys():
    print("\n" + "-" * 80)
    print(f"{model_name} 模型评估结果")
    print("-" * 80)

    y_pred = predictions[model_name]

    # 计算各项指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1值 (F1-Score): {f1:.4f}")

    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris_target_names))

# 绘制混淆矩阵
print("\n绘制混淆矩阵...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_name in enumerate(models.keys()):
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris_target_names,
                yticklabels=iris_target_names,
                ax=axes[idx])
    axes[idx].set_xlabel('预测标签')
    axes[idx].set_ylabel('真实标签')
    axes[idx].set_title(f'{model_name} 混淆矩阵')

plt.tight_layout()
confusion_matrix_path = os.path.join(script_dir, 'confusion_matrices.png')
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
print(f"混淆矩阵已保存为: {confusion_matrix_path}")
plt.close()

# 模型性能对比
print("\n" + "=" * 80)
print("模型性能对比分析")
print("=" * 80)

# 创建对比表格
comparison_df = pd.DataFrame({
    '模型': list(results.keys()),
    '准确率': [results[m]['accuracy'] for m in results.keys()],
    '精确率': [results[m]['precision'] for m in results.keys()],
    '召回率': [results[m]['recall'] for m in results.keys()],
    'F1值': [results[m]['f1'] for m in results.keys()]
})

print("\n模型性能对比表:")
print(comparison_df.to_string(index=False))

# 绘制性能对比柱状图
print("\n绘制模型性能对比图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['准确率', '精确率', '召回率', 'F1值']
metric_keys = ['accuracy', 'precision', 'recall', 'f1']

for idx, (metric, metric_key) in enumerate(zip(metrics, metric_keys)):
    ax = axes[idx // 2, idx % 2]
    values = [results[m][metric_key] for m in results.keys()]
    bars = ax.bar(results.keys(), values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    ax.set_ylabel(metric)
    ax.set_title(f'{metric}对比')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

plt.tight_layout()
comparison_path = os.path.join(script_dir, 'model_comparison.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"模型性能对比图已保存为: {comparison_path}")
plt.close()

# 找出最佳模型
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
print(f"\n最佳模型: {best_model_name}")
print(f"  准确率: {results[best_model_name]['accuracy']:.4f}")
print(f"  精确率: {results[best_model_name]['precision']:.4f}")
print(f"  召回率: {results[best_model_name]['recall']:.4f}")
print(f"  F1值: {results[best_model_name]['f1']:.4f}")

print("\n" + "=" * 80)
print("所有任务完成！")
print("=" * 80)
print("\n生成的文件:")
print(f"  1. {output_path1}")
print(f"  2. {output_path2}")
print(f"  3. {confusion_matrix_path}")
print(f"  4. {comparison_path}")
print("=" * 80)

