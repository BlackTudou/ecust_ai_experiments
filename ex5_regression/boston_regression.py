"""
波士顿房价回归分析完整实现
包含：数据加载、探索、预处理、线性/非线性回归、模型评估、正则化应用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体支持和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII负号而不是Unicode负号
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 10
# 确保matplotlib使用ASCII字符而不是Unicode字符
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

def ensure_ascii_minus():
    """确保matplotlib使用ASCII负号而不是Unicode负号"""
    plt.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 数据准备：加载波士顿房价数据集
# =====================================================================
print("=" * 80)
print("1. 数据准备：加载波士顿房价数据集")
print("=" * 80)

# 尝试加载波士顿房价数据集（兼容不同版本的sklearn）
boston = None
try:
    # 方法1：尝试从sklearn直接加载
    from sklearn.datasets import load_boston
    boston = load_boston()
    print("\n成功从sklearn加载波士顿房价数据集")
except (ImportError, AttributeError):
    try:
        # 方法2：尝试使用fetch_california_housing作为替代
        from sklearn.datasets import fetch_california_housing
        print("\n警告：当前sklearn版本不支持波士顿房价数据集，使用加州房价数据集作为替代")
        california = fetch_california_housing()
        # 创建一个类似boston的对象
        class BostonData:
            def __init__(self, data, target, feature_names):
                self.data = data
                self.target = target
                self.feature_names = feature_names
        boston = BostonData(california.data, california.target, california.feature_names)
    except:
        # 方法3：如果都失败，创建一个模拟数据集
        print("\n警告：无法加载标准数据集，使用模拟数据")
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=506, n_features=13, noise=10, random_state=42)
        class BostonData:
            def __init__(self, data, target, feature_names):
                self.data = data
                self.target = target
                self.feature_names = feature_names
        boston = BostonData(X, y,
            ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

# 将数据转换为DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

print(f"\n数据形状: {df.shape}")
print(f"特征数量: {len(boston.feature_names)}")
print(f"样本数量: {len(df)}")
print(f"\n特征名称: {list(boston.feature_names)}")

# =====================================================================
# 2. 数据探索与预处理
# =====================================================================
print("\n" + "=" * 80)
print("2. 数据探索与预处理")
print("=" * 80)

# 2.1 查看数据基本信息
print("\n2.1 数据基本信息")
print("-" * 80)
print(df.head())
print("\n数据统计描述:")
print(df.describe())

# 2.2 检查缺失值
print("\n2.2 缺失值检查")
print("-" * 80)
missing_values = df.isnull().sum()
print("缺失值统计:")
print(missing_values)
if missing_values.sum() > 0:
    print("\n处理缺失值...")
    df = df.dropna()
    print(f"处理后数据形状: {df.shape}")
else:
    print("数据中没有缺失值")

# 2.3 检查异常值（使用IQR方法）
print("\n2.3 异常值检查与处理")
print("-" * 80)
outlier_count = 0
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if len(outliers) > 0:
        outlier_count += len(outliers)
        print(f"  {col}: 发现 {len(outliers)} 个异常值")

if outlier_count > 0:
    print(f"\n总共发现 {outlier_count} 个异常值")
    # 可以选择删除异常值或使用其他方法处理
    # 这里我们保留异常值，因为房价数据中的极端值可能是有意义的
    print("保留异常值（房价数据中的极端值可能是有意义的）")
else:
    print("未发现明显异常值")

# 2.4 数据分布可视化
print("\n2.4 绘制数据分布图...")
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    if i < len(axes):
        axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{col} 分布', fontsize=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('频数')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
distribution_path = os.path.join(script_dir, 'data_distribution.png')
plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
print(f"数据分布图已保存为: {distribution_path}")
plt.close()

# 2.5 特征间相关性分析
print("\n2.5 特征间相关性分析")
print("-" * 80)
correlation_matrix = df.corr()
print("\n特征与目标变量的相关性:")
target_corr = correlation_matrix['PRICE'].sort_values(ascending=False)
print(target_corr)

# 绘制相关性热力图
print("\n绘制特征相关性热力图...")
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('特征相关性热力图', fontsize=16, pad=20)
plt.tight_layout()
correlation_path = os.path.join(script_dir, 'feature_correlation.png')
plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
print(f"相关性热力图已保存为: {correlation_path}")
plt.close()

# 2.6 数据集划分
print("\n2.6 数据集划分")
print("-" * 80)
X = df.drop('PRICE', axis=1).values
y = df['PRICE'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")

# 2.7 特征标准化
print("\n2.7 特征标准化")
print("-" * 80)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("特征标准化完成（均值为0，标准差为1）")

# =====================================================================
# 3. 多元线性回归实现
# =====================================================================
print("\n" + "=" * 80)
print("3. 多元线性回归实现")
print("=" * 80)

# 3.1 训练多元线性回归模型
print("\n3.1 训练多元线性回归模型")
print("-" * 80)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("模型训练完成")

# 3.2 分析各特征的权重系数
print("\n3.2 特征权重系数分析")
print("-" * 80)
coefficients = pd.DataFrame({
    '特征': boston.feature_names,
    '权重系数': lr_model.coef_,
    '绝对值': np.abs(lr_model.coef_)
})
coefficients = coefficients.sort_values('绝对值', ascending=False)
print("\n各特征权重系数（按绝对值排序）:")
print(coefficients.to_string(index=False))
print(f"\n截距项: {lr_model.intercept_:.4f}")

# 可视化特征权重
plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'blue' for x in coefficients['权重系数']]
plt.barh(range(len(coefficients)), coefficients['权重系数'], color=colors, alpha=0.7)
plt.yticks(range(len(coefficients)), coefficients['特征'])
plt.xlabel('权重系数', fontsize=12)
plt.title('多元线性回归 - 特征权重系数', fontsize=14)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
coefficients_path = os.path.join(script_dir, 'linear_regression_coefficients.png')
plt.savefig(coefficients_path, dpi=300, bbox_inches='tight')
print(f"\n特征权重系数图已保存为: {coefficients_path}")
plt.close()

# =====================================================================
# 4. 多元非线性回归实现（SVM）
# =====================================================================
print("\n" + "=" * 80)
print("4. 多元非线性回归实现（SVM）")
print("=" * 80)

# 4.1 训练SVM回归模型
print("\n4.1 训练SVM回归模型")
print("-" * 80)
# 使用RBF核（非线性）
svm_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
print("开始训练SVM模型（这可能需要一些时间）...")
svm_model.fit(X_train_scaled, y_train)
print("SVM模型训练完成")

# 4.2 与原始数据对比
print("\n4.2 模型预测结果")
print("-" * 80)
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
svm_train_pred = svm_model.predict(X_train_scaled)
svm_test_pred = svm_model.predict(X_test_scaled)

# 可视化预测结果对比
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 线性回归 - 训练集
axes[0, 0].scatter(y_train, lr_train_pred, alpha=0.6, s=50)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'r--', lw=2, label='完美预测线')
axes[0, 0].set_xlabel('真实价格', fontsize=12)
axes[0, 0].set_ylabel('预测价格', fontsize=12)
axes[0, 0].set_title('线性回归 - 训练集', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 线性回归 - 测试集
axes[0, 1].scatter(y_test, lr_test_pred, alpha=0.6, s=50, color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='完美预测线')
axes[0, 1].set_xlabel('真实价格', fontsize=12)
axes[0, 1].set_ylabel('预测价格', fontsize=12)
axes[0, 1].set_title('线性回归 - 测试集', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# SVM回归 - 训练集
axes[1, 0].scatter(y_train, svm_train_pred, alpha=0.6, s=50, color='green')
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'r--', lw=2, label='完美预测线')
axes[1, 0].set_xlabel('真实价格', fontsize=12)
axes[1, 0].set_ylabel('预测价格', fontsize=12)
axes[1, 0].set_title('SVM回归 - 训练集', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# SVM回归 - 测试集
axes[1, 1].scatter(y_test, svm_test_pred, alpha=0.6, s=50, color='purple')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='完美预测线')
axes[1, 1].set_xlabel('真实价格', fontsize=12)
axes[1, 1].set_ylabel('预测价格', fontsize=12)
axes[1, 1].set_title('SVM回归 - 测试集', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
prediction_comparison_path = os.path.join(script_dir, 'prediction_comparison.png')
plt.savefig(prediction_comparison_path, dpi=300, bbox_inches='tight')
print(f"预测结果对比图已保存为: {prediction_comparison_path}")
plt.close()

# =====================================================================
# 5. 模型评估
# =====================================================================
print("\n" + "=" * 80)
print("5. 模型评估")
print("=" * 80)

def evaluate_model(y_true, y_pred, dataset_name):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R²': r2}

# 评估线性回归模型
print("\n5.1 线性回归模型评估")
print("-" * 80)
lr_train_metrics = evaluate_model(y_train, lr_train_pred, '训练集')
lr_test_metrics = evaluate_model(y_test, lr_test_pred, '测试集')

print("\n训练集性能:")
print(f"  MSE (均方误差): {lr_train_metrics['MSE']:.4f}")
print(f"  MAE (平均绝对误差): {lr_train_metrics['MAE']:.4f}")
print(f"  R² (决定系数): {lr_train_metrics['R²']:.4f}")

print("\n测试集性能:")
print(f"  MSE (均方误差): {lr_test_metrics['MSE']:.4f}")
print(f"  MAE (平均绝对误差): {lr_test_metrics['MAE']:.4f}")
print(f"  R² (决定系数): {lr_test_metrics['R²']:.4f}")

# 评估SVM回归模型
print("\n5.2 SVM回归模型评估")
print("-" * 80)
svm_train_metrics = evaluate_model(y_train, svm_train_pred, '训练集')
svm_test_metrics = evaluate_model(y_test, svm_test_pred, '测试集')

print("\n训练集性能:")
print(f"  MSE (均方误差): {svm_train_metrics['MSE']:.4f}")
print(f"  MAE (平均绝对误差): {svm_train_metrics['MAE']:.4f}")
print(f"  R² (决定系数): {svm_train_metrics['R²']:.4f}")

print("\n测试集性能:")
print(f"  MSE (均方误差): {svm_test_metrics['MSE']:.4f}")
print(f"  MAE (平均绝对误差): {svm_test_metrics['MAE']:.4f}")
print(f"  R² (决定系数): {svm_test_metrics['R²']:.4f}")

# 5.3 分析模型拟合效果
print("\n5.3 模型拟合效果分析")
print("-" * 80)

# 计算过拟合/欠拟合指标
lr_train_r2 = lr_train_metrics['R²']
lr_test_r2 = lr_test_metrics['R²']
lr_gap = lr_train_r2 - lr_test_r2

svm_train_r2 = svm_train_metrics['R²']
svm_test_r2 = svm_test_metrics['R²']
svm_gap = svm_train_r2 - svm_test_r2

print("\n线性回归模型:")
print(f"  训练集 R²: {lr_train_r2:.4f}")
print(f"  测试集 R²: {lr_test_r2:.4f}")
print(f"  R² 差异: {lr_gap:.4f}")
if lr_gap > 0.1:
    print("  判断: 可能存在过拟合")
elif lr_test_r2 < 0.5:
    print("  判断: 可能存在欠拟合")
else:
    print("  判断: 拟合效果良好")

print("\nSVM回归模型:")
print(f"  训练集 R²: {svm_train_r2:.4f}")
print(f"  测试集 R²: {svm_test_r2:.4f}")
print(f"  R² 差异: {svm_gap:.4f}")
if svm_gap > 0.1:
    print("  判断: 可能存在过拟合")
elif svm_test_r2 < 0.5:
    print("  判断: 可能存在欠拟合")
else:
    print("  判断: 拟合效果良好")

# 可视化模型性能对比
print("\n绘制模型性能对比图...")
metrics_df = pd.DataFrame({
    '模型': ['线性回归-训练集', '线性回归-测试集', 'SVM回归-训练集', 'SVM回归-测试集'],
    'MSE': [lr_train_metrics['MSE'], lr_test_metrics['MSE'],
            svm_train_metrics['MSE'], svm_test_metrics['MSE']],
    'MAE': [lr_train_metrics['MAE'], lr_test_metrics['MAE'],
            svm_train_metrics['MAE'], svm_test_metrics['MAE']],
    'R²': [lr_train_metrics['R²'], lr_test_metrics['R²'],
           svm_train_metrics['R²'], svm_test_metrics['R²']]
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MSE对比
axes[0].bar(range(len(metrics_df)), metrics_df['MSE'],
            color=['#3498db', '#2980b9', '#2ecc71', '#27ae60'], alpha=0.7)
axes[0].set_xticks(range(len(metrics_df)))
axes[0].set_xticklabels(metrics_df['模型'], rotation=45, ha='right')
axes[0].set_ylabel('MSE', fontsize=12)
axes[0].set_title('均方误差 (MSE) 对比', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics_df['MSE']):
    axes[0].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# MAE对比
axes[1].bar(range(len(metrics_df)), metrics_df['MAE'],
            color=['#3498db', '#2980b9', '#2ecc71', '#27ae60'], alpha=0.7)
axes[1].set_xticks(range(len(metrics_df)))
axes[1].set_xticklabels(metrics_df['模型'], rotation=45, ha='right')
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('平均绝对误差 (MAE) 对比', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics_df['MAE']):
    axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

# R²对比
axes[2].bar(range(len(metrics_df)), metrics_df['R²'],
            color=['#3498db', '#2980b9', '#2ecc71', '#27ae60'], alpha=0.7)
axes[2].set_xticks(range(len(metrics_df)))
axes[2].set_xticklabels(metrics_df['模型'], rotation=45, ha='right')
axes[2].set_ylabel('R²', fontsize=12)
axes[2].set_title('决定系数 (R²) 对比', fontsize=14)
axes[2].set_ylim([0, 1.1])
axes[2].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics_df['R²']):
    axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
performance_comparison_path = os.path.join(script_dir, 'model_performance_comparison.png')
plt.savefig(performance_comparison_path, dpi=300, bbox_inches='tight')
print(f"模型性能对比图已保存为: {performance_comparison_path}")
plt.close()

# =====================================================================
# 6. 正则化应用
# =====================================================================
print("\n" + "=" * 80)
print("6. 正则化应用")
print("=" * 80)

# 6.1 测试不同的正则化参数
print("\n6.1 测试不同的正则化参数")
print("-" * 80)

# Lasso回归参数范围
lasso_alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
# Ridge回归参数范围
ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

lasso_results = []
ridge_results = []

print("\n测试Lasso回归（L1正则化）...")
for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    train_pred = lasso.predict(X_train_scaled)
    test_pred = lasso.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # 统计非零特征数量（Lasso的特征选择效果）
    n_features = np.sum(np.abs(lasso.coef_) > 1e-5)

    lasso_results.append({
        'alpha': alpha,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_features': n_features
    })

    print(f"  α={alpha:6.2f}: 测试集MSE={test_mse:.4f}, R²={test_r2:.4f}, 非零特征数={n_features}")

print("\n测试Ridge回归（L2正则化）...")
for alpha in ridge_alphas:
    ridge = Ridge(alpha=alpha, max_iter=10000)
    ridge.fit(X_train_scaled, y_train)

    train_pred = ridge.predict(X_train_scaled)
    test_pred = ridge.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    ridge_results.append({
        'alpha': alpha,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    })

    print(f"  α={alpha:8.2f}: 测试集MSE={test_mse:.4f}, R²={test_r2:.4f}")

# 6.2 选择最佳正则化参数
print("\n6.2 选择最佳正则化参数")
print("-" * 80)

best_lasso_idx = np.argmin([r['test_mse'] for r in lasso_results])
best_lasso = lasso_results[best_lasso_idx]
print(f"\n最佳Lasso参数: α={best_lasso['alpha']:.2f}")
print(f"  测试集MSE: {best_lasso['test_mse']:.4f}")
print(f"  测试集R²: {best_lasso['test_r2']:.4f}")
print(f"  非零特征数: {best_lasso['n_features']}")

best_ridge_idx = np.argmin([r['test_mse'] for r in ridge_results])
best_ridge = ridge_results[best_ridge_idx]
print(f"\n最佳Ridge参数: α={best_ridge['alpha']:.2f}")
print(f"  测试集MSE: {best_ridge['test_mse']:.4f}")
print(f"  测试集R²: {best_ridge['test_r2']:.4f}")

# 6.3 训练最佳正则化模型
print("\n6.3 训练最佳正则化模型")
print("-" * 80)

best_lasso_model = Lasso(alpha=best_lasso['alpha'], max_iter=10000)
best_lasso_model.fit(X_train_scaled, y_train)
lasso_train_pred = best_lasso_model.predict(X_train_scaled)
lasso_test_pred = best_lasso_model.predict(X_test_scaled)

best_ridge_model = Ridge(alpha=best_ridge['alpha'], max_iter=10000)
best_ridge_model.fit(X_train_scaled, y_train)
ridge_train_pred = best_ridge_model.predict(X_train_scaled)
ridge_test_pred = best_ridge_model.predict(X_test_scaled)

# 6.4 对比正则化前后模型性能
print("\n6.4 正则化前后模型性能对比")
print("-" * 80)

comparison_df = pd.DataFrame({
    '模型': ['线性回归（无正则化）', 'Lasso回归', 'Ridge回归'],
    '训练集MSE': [lr_train_metrics['MSE'],
                  mean_squared_error(y_train, lasso_train_pred),
                  mean_squared_error(y_train, ridge_train_pred)],
    '测试集MSE': [lr_test_metrics['MSE'],
                  mean_squared_error(y_test, lasso_test_pred),
                  mean_squared_error(y_test, ridge_test_pred)],
    '训练集R²': [lr_train_metrics['R²'],
                 r2_score(y_train, lasso_train_pred),
                 r2_score(y_train, ridge_train_pred)],
    '测试集R²': [lr_test_metrics['R²'],
                 r2_score(y_test, lasso_test_pred),
                 r2_score(y_test, ridge_test_pred)]
})

print("\n模型性能对比表:")
print(comparison_df.to_string(index=False))

# 6.5 观察Lasso回归对特征的选择效果
print("\n6.5 Lasso回归特征选择效果分析")
print("-" * 80)

lasso_coefficients = pd.DataFrame({
    '特征': boston.feature_names,
    'Lasso系数': best_lasso_model.coef_,
    '线性回归系数': lr_model.coef_,
    '绝对值(Lasso)': np.abs(best_lasso_model.coef_)
})
lasso_coefficients = lasso_coefficients.sort_values('绝对值(Lasso)', ascending=False)

print("\nLasso回归特征系数（按绝对值排序）:")
print(lasso_coefficients.to_string(index=False))

# 统计被Lasso选择为0的特征
zero_features = lasso_coefficients[lasso_coefficients['绝对值(Lasso)'] < 1e-5]
print(f"\n被Lasso回归置零的特征数量: {len(zero_features)}")
if len(zero_features) > 0:
    print("被置零的特征:")
    print(zero_features[['特征', 'Lasso系数']].to_string(index=False))

# 可视化Lasso特征选择效果
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Lasso系数 vs 线性回归系数
axes[0].scatter(lr_model.coef_, best_lasso_model.coef_, s=100, alpha=0.6)
axes[0].plot([lr_model.coef_.min(), lr_model.coef_.max()],
             [lr_model.coef_.min(), lr_model.coef_.max()],
             'r--', lw=2, label='y=x')
for i, feature in enumerate(boston.feature_names):
    axes[0].annotate(feature, (lr_model.coef_[i], best_lasso_model.coef_[i]),
                     fontsize=8, alpha=0.7)
axes[0].set_xlabel('线性回归系数', fontsize=12)
axes[0].set_ylabel('Lasso回归系数', fontsize=12)
axes[0].set_title('Lasso特征选择效果：系数对比', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 特征系数对比柱状图
x_pos = np.arange(len(boston.feature_names))
width = 0.35
axes[1].bar(x_pos - width/2, lr_model.coef_, width, label='线性回归', alpha=0.7)
axes[1].bar(x_pos + width/2, best_lasso_model.coef_, width, label='Lasso回归', alpha=0.7)
axes[1].set_xlabel('特征', fontsize=12)
axes[1].set_ylabel('系数值', fontsize=12)
axes[1].set_title('特征系数对比：线性回归 vs Lasso回归', fontsize=14)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(boston.feature_names, rotation=45, ha='right')
axes[1].legend()
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
lasso_feature_selection_path = os.path.join(script_dir, 'lasso_feature_selection.png')
plt.savefig(lasso_feature_selection_path, dpi=300, bbox_inches='tight')
print(f"\nLasso特征选择效果图已保存为: {lasso_feature_selection_path}")
plt.close()

# 6.6 可视化正则化参数对模型性能的影响
print("\n6.6 绘制正则化参数影响分析图...")

# 确保负号正确显示 - 使用ASCII负号
ensure_ascii_minus()
# 导入ticker用于格式化坐标轴
import matplotlib.ticker as ticker

# 创建自定义formatter，强制使用ASCII负号
def ascii_log_formatter(x, pos):
    """对数坐标轴的ASCII formatter"""
    if x == 0:
        return '0'
    # 使用科学计数法，确保负号是ASCII
    s = f'{x:.2e}'
    return s.replace('\u2212', '-').replace('−', '-')

def ascii_float_formatter(x, pos):
    """浮点数ASCII formatter"""
    s = f'{x:.3f}'
    return s.replace('\u2212', '-').replace('−', '-')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 在绘图前为所有子图设置formatter
for ax in axes.flat:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(ascii_log_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(ascii_float_formatter))

# Lasso: MSE vs alpha
lasso_alphas_plot = [r['alpha'] for r in lasso_results]
lasso_train_mse = [r['train_mse'] for r in lasso_results]
lasso_test_mse = [r['test_mse'] for r in lasso_results]

axes[0, 0].semilogx(lasso_alphas_plot, lasso_train_mse, 'o-', label='训练集MSE', linewidth=2, markersize=8)
axes[0, 0].semilogx(lasso_alphas_plot, lasso_test_mse, 's-', label='测试集MSE', linewidth=2, markersize=8)
axes[0, 0].axvline(x=best_lasso['alpha'], color='r', linestyle='--', label=f'最佳α={best_lasso["alpha"]:.2f}')
axes[0, 0].set_xlabel('正则化参数 α', fontsize=12)
axes[0, 0].set_ylabel('MSE', fontsize=12)
axes[0, 0].set_title('Lasso回归：MSE vs 正则化参数', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
# 设置formatter避免Unicode负号
axes[0, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2e}'.replace('\u2212', '-')))
axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}'.replace('\u2212', '-')))

# Lasso: R² vs alpha
lasso_train_r2 = [r['train_r2'] for r in lasso_results]
lasso_test_r2 = [r['test_r2'] for r in lasso_results]

axes[0, 1].semilogx(lasso_alphas_plot, lasso_train_r2, 'o-', label='训练集R²', linewidth=2, markersize=8)
axes[0, 1].semilogx(lasso_alphas_plot, lasso_test_r2, 's-', label='测试集R²', linewidth=2, markersize=8)
axes[0, 1].axvline(x=best_lasso['alpha'], color='r', linestyle='--', label=f'最佳α={best_lasso["alpha"]:.2f}')
axes[0, 1].set_xlabel('正则化参数 α', fontsize=12)
axes[0, 1].set_ylabel('R²', fontsize=12)
axes[0, 1].set_title('Lasso回归：R² vs 正则化参数', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
# 设置formatter避免Unicode负号
axes[0, 1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2e}'.replace('\u2212', '-')))
axes[0, 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3f}'.replace('\u2212', '-')))

# Ridge: MSE vs alpha
ridge_alphas_plot = [r['alpha'] for r in ridge_results]
ridge_train_mse = [r['train_mse'] for r in ridge_results]
ridge_test_mse = [r['test_mse'] for r in ridge_results]

axes[1, 0].semilogx(ridge_alphas_plot, ridge_train_mse, 'o-', label='训练集MSE', linewidth=2, markersize=8)
axes[1, 0].semilogx(ridge_alphas_plot, ridge_test_mse, 's-', label='测试集MSE', linewidth=2, markersize=8)
axes[1, 0].axvline(x=best_ridge['alpha'], color='r', linestyle='--', label=f'最佳α={best_ridge["alpha"]:.2f}')
axes[1, 0].set_xlabel('正则化参数 α', fontsize=12)
axes[1, 0].set_ylabel('MSE', fontsize=12)
axes[1, 0].set_title('Ridge回归：MSE vs 正则化参数', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
# 设置formatter避免Unicode负号
axes[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2e}'.replace('\u2212', '-')))
axes[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2f}'.replace('\u2212', '-')))

# Ridge: R² vs alpha
ridge_train_r2 = [r['train_r2'] for r in ridge_results]
ridge_test_r2 = [r['test_r2'] for r in ridge_results]

axes[1, 1].semilogx(ridge_alphas_plot, ridge_train_r2, 'o-', label='训练集R²', linewidth=2, markersize=8)
axes[1, 1].semilogx(ridge_alphas_plot, ridge_test_r2, 's-', label='测试集R²', linewidth=2, markersize=8)
axes[1, 1].axvline(x=best_ridge['alpha'], color='r', linestyle='--', label=f'最佳α={best_ridge["alpha"]:.2f}')
axes[1, 1].set_xlabel('正则化参数 α', fontsize=12)
axes[1, 1].set_ylabel('R²', fontsize=12)
axes[1, 1].set_title('Ridge回归：R² vs 正则化参数', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
# 设置formatter避免Unicode负号
axes[1, 1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.2e}'.replace('\u2212', '-')))
axes[1, 1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3f}'.replace('\u2212', '-')))

# 在所有绘图完成后，再次确保所有坐标轴使用ASCII负号
for ax in axes.flat:
    # 直接修改tick labels，替换Unicode负号为ASCII负号
    for label in ax.get_xticklabels():
        label.set_text(label.get_text().replace('\u2212', '-').replace('−', '-'))
    for label in ax.get_yticklabels():
        label.set_text(label.get_text().replace('\u2212', '-').replace('−', '-'))
    # 重新设置formatter
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x:.2e}'.replace('\u2212', '-').replace('−', '-')))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x:.3f}'.replace('\u2212', '-').replace('−', '-')))

plt.tight_layout()
regularization_analysis_path = os.path.join(script_dir, 'regularization_analysis.png')
# 确保保存时使用ASCII负号
ensure_ascii_minus()
# 在保存前强制刷新所有tick labels
for ax in axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
plt.savefig(regularization_analysis_path, dpi=300, bbox_inches='tight')
print(f"正则化参数影响分析图已保存为: {regularization_analysis_path}")
plt.close()

# =====================================================================
# 总结
# =====================================================================
print("\n" + "=" * 80)
print("实验完成！")
print("=" * 80)
print("\n生成的文件:")
print(f"  1. {distribution_path} - 数据分布图")
print(f"  2. {correlation_path} - 特征相关性热力图")
print(f"  3. {coefficients_path} - 线性回归特征权重系数图")
print(f"  4. {prediction_comparison_path} - 预测结果对比图")
print(f"  5. {performance_comparison_path} - 模型性能对比图")
print(f"  6. {lasso_feature_selection_path} - Lasso特征选择效果图")
print(f"  7. {regularization_analysis_path} - 正则化参数影响分析图")
print("\n模型性能总结:")
print(comparison_df.to_string(index=False))
print("=" * 80)

