"""
LDPE生产过程故障诊断：基于PCA和PLS的故障诊断
使用主成分分析（PCA）和偏最小二乘（PLS）对LDPE生产过程进行故障检测和隔离
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体支持和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 10

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)


# =====================================================================
# 1. 数据预处理
# =====================================================================
def load_and_preprocess_data(data_path):
    """
    加载和预处理数据

    参数:
        data_path: 数据文件路径

    返回:
        X_normal: 正常样本的过程变量
        Y_normal: 正常样本的质量变量
        X_fault: 故障样本的过程变量
        Y_fault: 故障样本的质量变量
        X_normal_std: 标准化后的正常样本过程变量
        Y_normal_std: 标准化后的正常样本质量变量
        X_fault_std: 标准化后的故障样本过程变量
        Y_fault_std: 标准化后的故障样本质量变量
        scaler_X: X的标准化器
        scaler_Y: Y的标准化器
    """
    print("\n" + "=" * 80)
    print("1. 数据加载与预处理")
    print("=" * 80)

    # 1.1 数据加载
    print("\n1.1 加载数据...")
    df = pd.read_csv(data_path)
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")

    # 提取过程变量（前14列）和质量变量（后5列）
    # 跳过第一列（索引列）
    process_vars = df.columns[1:15].tolist()  # 14个过程变量
    quality_vars = df.columns[15:].tolist()   # 5个质量变量

    print(f"\n  过程变量 ({len(process_vars)}个): {process_vars}")
    print(f"  质量变量 ({len(quality_vars)}个): {quality_vars}")

    # 分离正常样本（前50行）和故障样本（后4行）
    X_normal = df.iloc[:50, 1:15].values  # 前50行，14个过程变量
    Y_normal = df.iloc[:50, 15:].values    # 前50行，5个质量变量
    X_fault = df.iloc[50:, 1:15].values    # 后4行，14个过程变量
    Y_fault = df.iloc[50:, 15:].values      # 后4行，5个质量变量

    print(f"\n  正常样本数: {X_normal.shape[0]}")
    print(f"  故障样本数: {X_fault.shape[0]}")
    print(f"  过程变量数: {X_normal.shape[1]}")
    print(f"  质量变量数: {Y_normal.shape[1]}")

    # 1.2 数据标准化
    print("\n1.2 数据标准化...")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_normal_std = scaler_X.fit_transform(X_normal)
    Y_normal_std = scaler_Y.fit_transform(Y_normal)
    X_fault_std = scaler_X.transform(X_fault)
    Y_fault_std = scaler_Y.transform(Y_fault)

    print(f"  标准化后X_normal均值: {np.mean(X_normal_std, axis=0).round(4)}")
    print(f"  标准化后X_normal标准差: {np.std(X_normal_std, axis=0, ddof=1).round(4)}")

    # 1.3 数据验证
    print("\n1.3 数据验证...")
    # 检查NaN
    nan_count_X = np.isnan(X_normal_std).sum() + np.isnan(X_fault_std).sum()
    nan_count_Y = np.isnan(Y_normal_std).sum() + np.isnan(Y_fault_std).sum()
    print(f"  NaN值数量 - X: {nan_count_X}, Y: {nan_count_Y}")

    # 检查无穷大
    inf_count_X = np.isinf(X_normal_std).sum() + np.isinf(X_fault_std).sum()
    inf_count_Y = np.isinf(Y_normal_std).sum() + np.isinf(Y_fault_std).sum()
    print(f"  无穷大值数量 - X: {inf_count_X}, Y: {inf_count_Y}")

    if nan_count_X > 0 or nan_count_Y > 0 or inf_count_X > 0 or inf_count_Y > 0:
        print("  警告: 发现异常值，需要进行处理")
        # 处理NaN和无穷大
        X_normal_std = np.nan_to_num(X_normal_std, nan=0.0, posinf=0.0, neginf=0.0)
        X_fault_std = np.nan_to_num(X_fault_std, nan=0.0, posinf=0.0, neginf=0.0)
        Y_normal_std = np.nan_to_num(Y_normal_std, nan=0.0, posinf=0.0, neginf=0.0)
        Y_fault_std = np.nan_to_num(Y_fault_std, nan=0.0, posinf=0.0, neginf=0.0)
        print("  已使用0值替换异常值")
    else:
        print("  数据验证通过，无异常值")

    return (X_normal, Y_normal, X_fault, Y_fault,
            X_normal_std, Y_normal_std, X_fault_std, Y_fault_std,
            scaler_X, scaler_Y, process_vars, quality_vars)


# =====================================================================
# 2. PCA故障诊断
# =====================================================================
def calculate_t2_spe(X, pca_model):
    """
    计算T²和SPE统计量

    参数:
        X: 标准化后的数据 (n_samples, n_features)
        pca_model: 训练好的PCA模型

    返回:
        t2_values: T²统计量
        spe_values: SPE统计量
    """
    # 计算主成分得分
    scores = pca_model.transform(X)

    # 计算T²统计量: T² = score * inv(cov(score)) * score^T
    # 简化计算: T² = score * score^T / lambda (lambda为特征值)
    eigenvalues = pca_model.explained_variance_
    t2_values = np.sum(scores**2 / eigenvalues, axis=1)

    # 计算SPE统计量: SPE = ||X - X_reconstructed||^2
    X_reconstructed = pca_model.inverse_transform(scores)
    residuals = X - X_reconstructed
    spe_values = np.sum(residuals**2, axis=1)

    return t2_values, spe_values


def calculate_contribution(X, pca_model, t2_values, spe_values, sample_idx):
    """
    计算贡献度

    参数:
        X: 标准化后的数据
        pca_model: PCA模型
        t2_values: T²统计量
        spe_values: SPE统计量
        sample_idx: 样本索引

    返回:
        t2_contrib: T²贡献度
        spe_contrib: SPE贡献度
    """
    x_sample = X[sample_idx:sample_idx+1, :]
    scores = pca_model.transform(x_sample)

    # T²贡献度
    eigenvalues = pca_model.explained_variance_
    loadings = pca_model.components_.T  # (n_features, n_components)

    # T²贡献度计算
    t2_contrib = np.zeros(X.shape[1])
    for i in range(pca_model.n_components_):
        t2_contrib += (scores[0, i] / eigenvalues[i]) * loadings[:, i] * x_sample[0, :]

    # SPE贡献度
    X_reconstructed = pca_model.inverse_transform(scores)
    residuals = x_sample - X_reconstructed
    spe_contrib = residuals[0, :]**2

    return t2_contrib, spe_contrib


def pca_fault_diagnosis(X_normal_std, X_fault_std, process_vars):
    """
    PCA故障诊断

    参数:
        X_normal_std: 标准化后的正常样本过程变量
        X_fault_std: 标准化后的故障样本过程变量
        process_vars: 过程变量名称列表
    """
    print("\n" + "=" * 80)
    print("2. PCA故障诊断")
    print("=" * 80)

    # 2.1 正常模型训练
    print("\n2.1 正常模型训练...")

    # 确定主成分个数（累计方差贡献率≥85%）
    pca_temp = PCA()
    pca_temp.fit(X_normal_std)
    cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components = np.where(cumsum_variance >= 0.85)[0][0] + 1

    print(f"  累计方差贡献率: {cumsum_variance}")
    print(f"  选择主成分数: {n_components} (累计方差贡献率: {cumsum_variance[n_components-1]:.4f})")

    # 训练PCA模型
    pca_model = PCA(n_components=n_components)
    pca_model.fit(X_normal_std)

    print(f"  主成分解释的方差比例: {pca_model.explained_variance_ratio_}")
    print(f"  累计方差贡献率: {np.sum(pca_model.explained_variance_ratio_):.4f}")

    # 计算控制限
    n = X_normal_std.shape[0]  # 正常样本数
    k = n_components  # 主成分个数
    m = X_normal_std.shape[1]  # 过程变量个数
    alpha = 0.01  # 置信水平99%

    # T²控制限: T²_lim = k(n-1)/(n-k) * F(alpha, k, n-k)
    f_critical = stats.f.ppf(1 - alpha, k, n - k)
    t2_limit = (k * (n - 1) / (n - k)) * f_critical

    # SPE控制限: SPE_lim = theta1 * [C_alpha * sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/theta1^2]^h0
    # 简化计算: 使用卡方分布近似
    # SPE_lim = g * chi2(alpha, h)
    # 其中 g = theta2/theta1, h = theta1^2/theta2
    # 更简单的近似: SPE_lim = theta1 * [C_alpha * sqrt(2*theta2) + theta2]^2
    # 或者使用更常用的方法: SPE_lim = theta1 * [C_alpha * sqrt(2*theta2*h0^2)/theta1 + 1 + theta2*h0*(h0-1)/theta1^2]^h0

    # 计算正常样本的SPE值用于估计参数
    t2_normal, spe_normal = calculate_t2_spe(X_normal_std, pca_model)

    # SPE控制限的简化计算（基于卡方分布）
    # 使用Jackson-Mudholkar方法
    theta1 = np.mean(spe_normal)
    theta2 = np.var(spe_normal)
    h0 = 2 * theta1**2 / theta2 if theta2 > 0 else 1
    g = theta2 / (2 * theta1) if theta1 > 0 else 1
    chi2_critical = stats.chi2.ppf(1 - alpha, h0)
    spe_limit = g * chi2_critical

    print(f"\n  控制限计算:")
    print(f"    T²控制限 (99%置信水平): {t2_limit:.4f}")
    print(f"    SPE控制限 (99%置信水平): {spe_limit:.4f}")

    # 2.2 故障检测
    print("\n2.2 故障检测...")
    t2_normal, spe_normal = calculate_t2_spe(X_normal_std, pca_model)
    t2_fault, spe_fault = calculate_t2_spe(X_fault_std, pca_model)

    print(f"\n  正常样本统计量:")
    print(f"    T² - 均值: {np.mean(t2_normal):.4f}, 最大值: {np.max(t2_normal):.4f}")
    print(f"    SPE - 均值: {np.mean(spe_normal):.4f}, 最大值: {np.max(spe_normal):.4f}")

    print(f"\n  故障样本统计量:")
    for i in range(len(t2_fault)):
        t2_exceed = "超出" if t2_fault[i] > t2_limit else "未超出"
        spe_exceed = "超出" if spe_fault[i] > spe_limit else "未超出"
        print(f"    样本 {i+1}: T²={t2_fault[i]:.4f} ({t2_exceed}), SPE={spe_fault[i]:.4f} ({spe_exceed})")

    # 统计故障检测率
    t2_detected = np.sum(t2_fault > t2_limit)
    spe_detected = np.sum(spe_fault > spe_limit)
    either_detected = np.sum((t2_fault > t2_limit) | (spe_fault > spe_limit))

    print(f"\n  故障检测率:")
    print(f"    T²检测率: {t2_detected}/{len(t2_fault)} ({t2_detected/len(t2_fault)*100:.1f}%)")
    print(f"    SPE检测率: {spe_detected}/{len(spe_fault)} ({spe_detected/len(spe_fault)*100:.1f}%)")
    print(f"    综合检测率: {either_detected}/{len(t2_fault)} ({either_detected/len(t2_fault)*100:.1f}%)")

    # 2.3 故障隔离
    print("\n2.3 故障隔离（贡献图分析）...")

    # 找出超出控制限的故障样本
    fault_indices = []
    for i in range(len(t2_fault)):
        if t2_fault[i] > t2_limit or spe_fault[i] > spe_limit:
            fault_indices.append(i)

    if len(fault_indices) > 0:
        print(f"  分析 {len(fault_indices)} 个超出控制限的故障样本")

        # 为每个故障样本绘制贡献图
        for idx in fault_indices:
            t2_contrib, spe_contrib = calculate_contribution(
                X_fault_std, pca_model, t2_fault, spe_fault, idx)

            # 找出贡献最大的前3个变量
            t2_top3_idx = np.argsort(np.abs(t2_contrib))[-3:][::-1]
            spe_top3_idx = np.argsort(np.abs(spe_contrib))[-3:][::-1]

            print(f"\n  故障样本 {idx+1} (原始索引 {idx+51}):")
            print(f"    T²贡献最大的前3个变量:")
            for i, var_idx in enumerate(t2_top3_idx):
                print(f"      {i+1}. {process_vars[var_idx]}: {t2_contrib[var_idx]:.4f}")
            print(f"    SPE贡献最大的前3个变量:")
            for i, var_idx in enumerate(spe_top3_idx):
                print(f"      {i+1}. {process_vars[var_idx]}: {spe_contrib[var_idx]:.4f}")

    # 2.4 结果可视化
    print("\n2.4 结果可视化...")

    # T²-SPE散点图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制正常样本
    ax.scatter(t2_normal, spe_normal, c='blue', marker='o', s=50,
               alpha=0.6, label='正常样本', edgecolors='black', linewidths=0.5)

    # 绘制故障样本
    ax.scatter(t2_fault, spe_fault, c='red', marker='s', s=100,
               alpha=0.8, label='故障样本', edgecolors='black', linewidths=1)

    # 绘制控制限
    ax.axvline(x=t2_limit, color='green', linestyle='--', linewidth=2, label=f'T²控制限 ({t2_limit:.2f})')
    ax.axhline(y=spe_limit, color='orange', linestyle='--', linewidth=2, label=f'SPE控制限 ({spe_limit:.2f})')

    ax.set_xlabel('T²统计量', fontsize=12)
    ax.set_ylabel('SPE统计量', fontsize=12)
    ax.set_title('PCA故障诊断：T²-SPE散点图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = os.path.join(results_dir, 'pca_t2_spe_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"  已保存: {scatter_path}")
    plt.close()

    # 贡献图（为第一个超出控制限的故障样本绘制）
    if len(fault_indices) > 0:
        idx = fault_indices[0]
        t2_contrib, spe_contrib = calculate_contribution(
            X_fault_std, pca_model, t2_fault, spe_fault, idx)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # T²贡献图
        t2_top10_idx = np.argsort(np.abs(t2_contrib))[-10:][::-1]
        ax1.barh(range(len(t2_top10_idx)), t2_contrib[t2_top10_idx], color='steelblue')
        ax1.set_yticks(range(len(t2_top10_idx)))
        ax1.set_yticklabels([process_vars[i] for i in t2_top10_idx])
        ax1.set_xlabel('T²贡献度', fontsize=12)
        ax1.set_title(f'故障样本 {idx+1} 的T²贡献图', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # SPE贡献图
        spe_top10_idx = np.argsort(np.abs(spe_contrib))[-10:][::-1]
        ax2.barh(range(len(spe_top10_idx)), spe_contrib[spe_top10_idx], color='coral')
        ax2.set_yticks(range(len(spe_top10_idx)))
        ax2.set_yticklabels([process_vars[i] for i in spe_top10_idx])
        ax2.set_xlabel('SPE贡献度', fontsize=12)
        ax2.set_title(f'故障样本 {idx+1} 的SPE贡献图', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        contrib_path = os.path.join(results_dir, 'pca_contribution_plot.png')
        plt.savefig(contrib_path, dpi=300, bbox_inches='tight')
        print(f"  已保存: {contrib_path}")
        plt.close()

    return pca_model, t2_limit, spe_limit, t2_fault, spe_fault


# =====================================================================
# 3. PLS故障诊断（选做）
# =====================================================================
def pls_fault_diagnosis(X_normal_std, Y_normal_std, X_fault_std, Y_fault_std,
                        process_vars, quality_vars):
    """
    PLS故障诊断

    参数:
        X_normal_std: 标准化后的正常样本过程变量
        Y_normal_std: 标准化后的正常样本质量变量
        X_fault_std: 标准化后的故障样本过程变量
        Y_fault_std: 标准化后的故障样本质量变量
        process_vars: 过程变量名称列表
        quality_vars: 质量变量名称列表
    """
    print("\n" + "=" * 80)
    print("3. PLS故障诊断（选做实验）")
    print("=" * 80)

    # 3.1 正常模型训练
    print("\n3.1 正常模型训练...")

    # 通过交叉验证确定最优主成分个数
    print("  交叉验证确定最优主成分个数...")
    max_components = min(X_normal_std.shape[1], Y_normal_std.shape[1], 10)
    cv_scores = []

    for n_comp in range(1, max_components + 1):
        pls_temp = PLSRegression(n_components=n_comp)
        scores = cross_val_score(pls_temp, X_normal_std, Y_normal_std,
                                cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-np.mean(scores))

    optimal_n_components = np.argmin(cv_scores) + 1
    print(f"  交叉验证MSE: {cv_scores}")
    print(f"  最优主成分数: {optimal_n_components}")

    # 训练PLS模型
    pls_model = PLSRegression(n_components=optimal_n_components)
    pls_model.fit(X_normal_std, Y_normal_std)

    print(f"  模型训练完成，主成分数: {optimal_n_components}")

    # 计算PRESS控制限（基于正常样本的预测误差分布）
    Y_normal_pred = pls_model.predict(X_normal_std)
    press_normal = np.sum((Y_normal_std - Y_normal_pred)**2, axis=1)

    # 使用99%置信水平
    alpha = 0.01
    press_limit = np.percentile(press_normal, (1 - alpha) * 100)

    print(f"\n  正常样本PRESS统计:")
    print(f"    均值: {np.mean(press_normal):.4f}")
    print(f"    最大值: {np.max(press_normal):.4f}")
    print(f"    PRESS控制限 (99%置信水平): {press_limit:.4f}")

    # 3.2 故障检测
    print("\n3.2 故障检测...")
    Y_fault_pred = pls_model.predict(X_fault_std)
    press_fault = np.sum((Y_fault_std - Y_fault_pred)**2, axis=1)

    print(f"\n  故障样本PRESS值:")
    for i in range(len(press_fault)):
        exceed = "超出" if press_fault[i] > press_limit else "未超出"
        print(f"    样本 {i+1}: PRESS={press_fault[i]:.4f} ({exceed})")

    # 统计故障检测率
    detected = np.sum(press_fault > press_limit)
    print(f"\n  故障检测率: {detected}/{len(press_fault)} ({detected/len(press_fault)*100:.1f}%)")

    # 3.3 故障溯源
    print("\n3.3 故障溯源（载荷矩阵分析）...")

    # 分析PLS载荷矩阵
    # PLS的载荷矩阵表示过程变量与质量变量的关联
    x_loadings = pls_model.x_loadings_  # (n_features, n_components)
    y_loadings = pls_model.y_loadings_  # (n_targets, n_components)

    # 计算过程变量与质量变量的相关性（通过载荷矩阵）
    # 对于每个质量变量，找出相关性最强的过程变量
    print("\n  过程变量与质量变量的关联分析（前3个过程变量）:")
    for i, q_var in enumerate(quality_vars):
        # 计算每个过程变量对该质量变量的总贡献
        contributions = np.zeros(len(process_vars))
        for comp in range(optimal_n_components):
            contributions += np.abs(x_loadings[:, comp] * y_loadings[i, comp])

        top3_idx = np.argsort(contributions)[-3:][::-1]
        print(f"\n  {q_var}:")
        for j, var_idx in enumerate(top3_idx):
            print(f"    {j+1}. {process_vars[var_idx]}: {contributions[var_idx]:.4f}")

    # 3.4 结果可视化
    print("\n3.4 结果可视化...")

    # PRESS值对比图
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_indices = np.arange(1, len(press_normal) + 1)
    fault_indices = np.arange(len(press_normal) + 1, len(press_normal) + len(press_fault) + 1)

    ax.plot(sample_indices, press_normal, 'o-', color='blue',
            markersize=4, alpha=0.6, label='正常样本', linewidth=1)
    ax.plot(fault_indices, press_fault, 's-', color='red',
            markersize=8, alpha=0.8, label='故障样本', linewidth=2)
    ax.axhline(y=press_limit, color='orange', linestyle='--',
               linewidth=2, label=f'PRESS控制限 ({press_limit:.2f})')

    ax.set_xlabel('样本编号', fontsize=12)
    ax.set_ylabel('PRESS值', fontsize=12)
    ax.set_title('PLS故障诊断：PRESS值对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    press_path = os.path.join(results_dir, 'pls_press_comparison.png')
    plt.savefig(press_path, dpi=300, bbox_inches='tight')
    print(f"  已保存: {press_path}")
    plt.close()

    # 载荷矩阵热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # X载荷矩阵
    x_loadings_df = pd.DataFrame(x_loadings, index=process_vars,
                                  columns=[f'PC{i+1}' for i in range(optimal_n_components)])
    sns.heatmap(x_loadings_df, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, ax=ax1, cbar_kws={'label': '载荷值'})
    ax1.set_title('PLS X载荷矩阵（过程变量）', fontsize=12, fontweight='bold')
    ax1.set_xlabel('主成分', fontsize=10)
    ax1.set_ylabel('过程变量', fontsize=10)

    # Y载荷矩阵
    y_loadings_df = pd.DataFrame(y_loadings, index=quality_vars,
                                  columns=[f'PC{i+1}' for i in range(optimal_n_components)])
    sns.heatmap(y_loadings_df, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, ax=ax2, cbar_kws={'label': '载荷值'})
    ax2.set_title('PLS Y载荷矩阵（质量变量）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('主成分', fontsize=10)
    ax2.set_ylabel('质量变量', fontsize=10)

    plt.tight_layout()
    loading_path = os.path.join(results_dir, 'pls_loadings_heatmap.png')
    plt.savefig(loading_path, dpi=300, bbox_inches='tight')
    print(f"  已保存: {loading_path}")
    plt.close()

    return pls_model, press_limit, press_fault


# =====================================================================
# 主程序
# =====================================================================
def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("LDPE生产过程故障诊断：基于PCA和PLS的故障诊断")
    print("=" * 80)

    # 数据路径
    data_path = os.path.join(script_dir, '..', 'data', 'exp16_data_LDPE.csv')

    # 1. 数据预处理
    (X_normal, Y_normal, X_fault, Y_fault,
     X_normal_std, Y_normal_std, X_fault_std, Y_fault_std,
     scaler_X, scaler_Y, process_vars, quality_vars) = load_and_preprocess_data(data_path)

    # 2. PCA故障诊断（必做）
    pca_model, t2_limit, spe_limit, t2_fault, spe_fault = pca_fault_diagnosis(
        X_normal_std, X_fault_std, process_vars)

    # 3. PLS故障诊断（选做）
    pls_model, press_limit, press_fault = pls_fault_diagnosis(
        X_normal_std, Y_normal_std, X_fault_std, Y_fault_std,
        process_vars, quality_vars)

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"\n结果文件保存在: {results_dir}")
    print("\n生成的文件:")
    print("  - pca_t2_spe_scatter.png: PCA T²-SPE散点图")
    print("  - pca_contribution_plot.png: PCA贡献图")
    print("  - pls_press_comparison.png: PLS PRESS值对比图")
    print("  - pls_loadings_heatmap.png: PLS载荷矩阵热力图")


if __name__ == '__main__':
    main()

