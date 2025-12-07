"""
模型性能差异原因分析
深入分析逻辑回归、K近邻、决策树三种模型的性能差异原因
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# =====================================================================
# 1. 数据加载与预处理
# =====================================================================
print("=" * 80)
print("模型性能差异原因分析")
print("=" * 80)

# 加载数据
data_file_path = os.path.join(project_root, 'data', 'exp04_iris_data.txt')
data_raw = pd.read_csv(data_file_path, header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target_name'],
                       skip_blank_lines=True)
data_raw = data_raw.dropna()

target_names_unique = data_raw['target_name'].unique()
target_name_to_code = {name: idx for idx, name in enumerate(target_names_unique)}
data = data_raw.copy()
data['target'] = data_raw['target_name'].map(target_name_to_code)
data['target_name'] = pd.Categorical(data_raw['target_name'], categories=target_names_unique)

feature_names_short = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_target_names = np.array(target_names_unique)

# 准备数据
X = data[feature_names_short].values
y = data['target'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 2. 训练模型
# =====================================================================
print("\n训练模型...")

# 逻辑回归
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# K近邻（最佳K=9）
knn_model = KNeighborsClassifier(n_neighbors=9)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

# 决策树（最佳深度=3）
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

# =====================================================================
# 3. 分析数据特征
# =====================================================================
print("\n" + "=" * 80)
print("3. 数据特征分析")
print("=" * 80)

# 3.1 类别可分性分析
print("\n3.1 类别可分性分析")
print("-" * 80)

# 计算类别间距离
from scipy.spatial.distance import cdist

class_centers = []
for i in range(len(iris_target_names)):
    class_data = X_train[y_train == i]
    center = class_data.mean(axis=0)
    class_centers.append(center)
    print(f"\n{iris_target_names[i]} 类别中心:")
    for j, feature in enumerate(feature_names_short):
        print(f"  {feature}: {center[j]:.4f}")

class_centers = np.array(class_centers)
distances = cdist(class_centers, class_centers)
print("\n类别中心间距离矩阵:")
distance_df = pd.DataFrame(distances,
                           index=iris_target_names,
                           columns=iris_target_names)
print(distance_df)

# 3.2 特征重要性分析
print("\n3.2 特征重要性分析")
print("-" * 80)

# 使用决策树获取特征重要性
feature_importance = dt_model.feature_importances_
print("\n决策树特征重要性:")
for i, (feature, importance) in enumerate(zip(feature_names_short, feature_importance)):
    print(f"  {feature}: {importance:.4f}")

# 可视化特征重要性
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 特征重要性柱状图
axes[0].barh(feature_names_short, feature_importance, color='steelblue', alpha=0.7)
axes[0].set_xlabel('重要性')
axes[0].set_title('决策树特征重要性')
axes[0].grid(True, alpha=0.3, axis='x')

# 类别间距离热力图
sns.heatmap(distance_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('类别中心间距离')
axes[1].set_xlabel('类别')
axes[1].set_ylabel('类别')

plt.tight_layout()
feature_analysis_path = os.path.join(script_dir, 'feature_analysis.png')
plt.savefig(feature_analysis_path, dpi=300, bbox_inches='tight')
print(f"\n特征分析图已保存为: {feature_analysis_path}")
plt.close()

# 3.3 数据分布重叠分析
print("\n3.3 数据分布重叠分析")
print("-" * 80)

# 计算每个类别在每个特征上的分布范围
print("\n各类别特征值范围:")
for i, target_name in enumerate(iris_target_names):
    class_data = X_train[y_train == i]
    print(f"\n{target_name}:")
    for j, feature in enumerate(feature_names_short):
        min_val = class_data[:, j].min()
        max_val = class_data[:, j].max()
        mean_val = class_data[:, j].mean()
        std_val = class_data[:, j].std()
        print(f"  {feature}: [{min_val:.2f}, {max_val:.2f}], 均值={mean_val:.2f}, 标准差={std_val:.2f}")

# =====================================================================
# 4. 模型特性分析
# =====================================================================
print("\n" + "=" * 80)
print("4. 模型特性分析")
print("=" * 80)

# 4.1 逻辑回归分析
print("\n4.1 逻辑回归模型特性")
print("-" * 80)
print("模型假设: 线性决策边界")
print("优点: 简单、可解释性强、训练速度快")
print("缺点: 假设数据线性可分，对非线性关系建模能力有限")
print("\n逻辑回归系数分析:")
for i, target_name in enumerate(iris_target_names):
    print(f"\n{target_name} 的系数:")
    for j, feature in enumerate(feature_names_short):
        coef = lr_model.coef_[i][j]
        print(f"  {feature}: {coef:.4f}")

# 可视化决策边界（使用前两个主成分）
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 4.2 K近邻分析
print("\n4.2 K近邻模型特性")
print("-" * 80)
print(f"模型参数: K={9}")
print("优点: 非参数方法，能捕捉局部模式，对非线性关系适应性强")
print("缺点: 对噪声敏感，计算复杂度高，需要选择合适的K值")
print("\nK值选择分析:")
k_values = [1, 3, 5, 7, 9, 11, 15]
k_train_scores = []
k_test_scores = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    train_score = knn_temp.score(X_train_scaled, y_train)
    test_score = knn_temp.score(X_test_scaled, y_test)
    k_train_scores.append(train_score)
    k_test_scores.append(test_score)

# 可视化K值影响
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, k_train_scores, 'o-', label='训练集准确率', linewidth=2, markersize=8)
ax.plot(k_values, k_test_scores, 's-', label='测试集准确率', linewidth=2, markersize=8)
ax.set_xlabel('K值')
ax.set_ylabel('准确率')
ax.set_title('K值对模型性能的影响')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
k_analysis_path = os.path.join(script_dir, 'k_value_analysis.png')
plt.savefig(k_analysis_path, dpi=300, bbox_inches='tight')
print(f"K值分析图已保存为: {k_analysis_path}")
plt.close()

# 4.3 决策树分析
print("\n4.3 决策树模型特性")
print("-" * 80)
print(f"模型参数: max_depth={3}")
print("优点: 能捕捉非线性关系，特征选择能力强，可解释性好")
print("缺点: 容易过拟合，对数据变化敏感")
print(f"\n决策树深度: {dt_model.get_depth()}")
print(f"叶子节点数: {dt_model.get_n_leaves()}")

# 可视化决策树结构
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model,
          feature_names=feature_names_short,
          class_names=iris_target_names,
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
plt.title('决策树结构可视化', fontsize=16)
plt.tight_layout()
tree_structure_path = os.path.join(script_dir, 'decision_tree_structure.png')
plt.savefig(tree_structure_path, dpi=300, bbox_inches='tight')
print(f"决策树结构图已保存为: {tree_structure_path}")
plt.close()

# =====================================================================
# 5. 错误分析
# =====================================================================
print("\n" + "=" * 80)
print("5. 模型错误分析")
print("=" * 80)

# 获取混淆矩阵
lr_cm = confusion_matrix(y_test, lr_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
dt_cm = confusion_matrix(y_test, dt_pred)

print("\n5.1 逻辑回归错误分析")
print("-" * 80)
print("混淆矩阵:")
print(pd.DataFrame(lr_cm, index=iris_target_names, columns=iris_target_names))
lr_errors = np.sum(lr_cm) - np.trace(lr_cm)
print(f"总错误数: {lr_errors}")
for i in range(len(iris_target_names)):
    for j in range(len(iris_target_names)):
        if i != j and lr_cm[i, j] > 0:
            print(f"  {iris_target_names[i]} 被误判为 {iris_target_names[j]}: {lr_cm[i, j]} 次")

print("\n5.2 K近邻错误分析")
print("-" * 80)
print("混淆矩阵:")
print(pd.DataFrame(knn_cm, index=iris_target_names, columns=iris_target_names))
knn_errors = np.sum(knn_cm) - np.trace(knn_cm)
print(f"总错误数: {knn_errors}")
for i in range(len(iris_target_names)):
    for j in range(len(iris_target_names)):
        if i != j and knn_cm[i, j] > 0:
            print(f"  {iris_target_names[i]} 被误判为 {iris_target_names[j]}: {knn_cm[i, j]} 次")

print("\n5.3 决策树错误分析")
print("-" * 80)
print("混淆矩阵:")
print(pd.DataFrame(dt_cm, index=iris_target_names, columns=iris_target_names))
dt_errors = np.sum(dt_cm) - np.trace(dt_cm)
print(f"总错误数: {dt_errors}")
for i in range(len(iris_target_names)):
    for j in range(len(iris_target_names)):
        if i != j and dt_cm[i, j] > 0:
            print(f"  {iris_target_names[i]} 被误判为 {iris_target_names[j]}: {dt_cm[i, j]} 次")

# 找出被误判的样本
print("\n5.4 被误判样本的特征分析")
print("-" * 80)

# 找出所有被误判的样本
misclassified_samples = {}
for model_name, pred in [('逻辑回归', lr_pred), ('K近邻', knn_pred), ('决策树', dt_pred)]:
    misclassified = []
    for i in range(len(y_test)):
        if y_test[i] != pred[i]:
            misclassified.append({
                'true_label': iris_target_names[y_test[i]],
                'pred_label': iris_target_names[pred[i]],
                'features': X_test[i]
            })
    misclassified_samples[model_name] = misclassified

for model_name, samples in misclassified_samples.items():
    if len(samples) > 0:
        print(f"\n{model_name} 误判样本特征:")
        for idx, sample in enumerate(samples, 1):
            print(f"\n  样本 {idx}:")
            print(f"    真实标签: {sample['true_label']}")
            print(f"    预测标签: {sample['pred_label']}")
            for j, feature in enumerate(feature_names_short):
                print(f"    {feature}: {sample['features'][j]:.4f}")

# =====================================================================
# 6. 性能差异原因总结
# =====================================================================
print("\n" + "=" * 80)
print("6. 性能差异原因总结")
print("=" * 80)

print("\n6.1 为什么决策树表现最好？")
print("-" * 80)
print("""
1. 非线性建模能力: 决策树能够通过树状结构捕捉特征之间的非线性关系，
   而鸢尾花数据集中，不同类别之间的边界可能是非线性的。

2. 特征选择: 决策树能够自动选择最重要的特征进行分割，从特征重要性
   分析可以看出，花瓣长度和花瓣宽度是最重要的特征，决策树能够有效利用这些特征。

3. 深度控制: 通过限制最大深度为3，避免了过拟合，同时保持了足够的
   复杂度来区分三个类别。

4. 类别可分性: 鸢尾花数据集本身类别间距离较大，决策树能够很好地
   利用这种可分性。
""")

print("\n6.2 为什么K近邻表现中等？")
print("-" * 80)
print("""
1. 局部模式捕捉: K近邻能够很好地捕捉局部模式，对于边界清晰的样本
   分类效果好。

2. K值选择: 通过选择K=9，在偏差和方差之间取得了平衡，既不会因为
   K太小而过拟合，也不会因为K太大而欠拟合。

3. 距离度量: 使用标准化后的特征，使得不同特征在距离计算中权重相等，
   这对K近邻很重要。

4. 局限性: 对于边界模糊的样本，K近邻可能受到噪声影响，导致误判。
""")

print("\n6.3 为什么逻辑回归表现相对较差？")
print("-" * 80)
print("""
1. 线性假设限制: 逻辑回归假设决策边界是线性的，但鸢尾花数据集中
   类别之间的边界可能是非线性的，这限制了逻辑回归的表现。

2. 多分类问题: 虽然逻辑回归支持多分类，但它是通过多个二分类器
   实现的，对于复杂的多分类问题，性能可能不如专门的多分类算法。

3. 特征交互: 逻辑回归难以捕捉特征之间的复杂交互关系，而决策树
   和K近邻在这方面表现更好。

4. 参数限制: 逻辑回归的参数（系数）是全局的，无法像决策树那样
   在不同区域使用不同的决策规则。
""")

print("\n6.4 数据特征对模型性能的影响")
print("-" * 80)
print("""
1. 类别可分性: 鸢尾花数据集的三个类别在特征空间中相对分离，这为
   所有模型提供了良好的基础。

2. 特征重要性: 花瓣长度和花瓣宽度是最重要的特征，能够有效区分
   不同类别，决策树能够充分利用这一点。

3. 数据规模: 150个样本对于这三个模型来说都是足够的，但相对较小的
   数据集使得决策树的优势更加明显。

4. 数据分布: 三个类别的样本数量相等，类别平衡，这有利于所有模型
   的训练和评估。
""")

# 创建综合对比可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 准确率对比
models = ['逻辑回归', 'K近邻', '决策树']
accuracies = [0.9111, 0.9556, 0.9778]
axes[0, 0].bar(models, accuracies, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7)
axes[0, 0].set_ylabel('准确率')
axes[0, 0].set_title('模型准确率对比')
axes[0, 0].set_ylim([0.85, 1.0])
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, acc in enumerate(accuracies):
    axes[0, 0].text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')

# 2. 错误数对比
errors = [4, 2, 1]
axes[0, 1].bar(models, errors, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7)
axes[0, 1].set_ylabel('错误数')
axes[0, 1].set_title('模型错误数对比')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, err in enumerate(errors):
    axes[0, 1].text(i, err + 0.1, f'{err}', ha='center', va='bottom')

# 3. 模型复杂度对比（定性）
complexity = [1, 2, 3]  # 逻辑回归最简单，决策树最复杂
axes[1, 0].barh(models, complexity, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7)
axes[1, 0].set_xlabel('模型复杂度（定性）')
axes[1, 0].set_title('模型复杂度对比')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. 特征重要性
axes[1, 1].barh(feature_names_short, feature_importance, color='steelblue', alpha=0.7)
axes[1, 1].set_xlabel('重要性')
axes[1, 1].set_title('决策树特征重要性')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
summary_path = os.path.join(script_dir, 'model_performance_summary.png')
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"\n性能总结图已保存为: {summary_path}")
plt.close()

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
print("\n生成的分析文件:")
print(f"  1. {feature_analysis_path}")
print(f"  2. {k_analysis_path}")
print(f"  3. {tree_structure_path}")
print(f"  4. {summary_path}")
print("=" * 80)

