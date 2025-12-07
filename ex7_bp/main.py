"""
BP神经网络主程序
整合手动实现、PyTorch实现和实验分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

from bp_manual import BPNeuralNetwork
from bp_pytorch import BPNeuralNetworkPyTorch, BPNeuralNetworkTrainer, EarlyStopping
from bp_analysis import comprehensive_analysis

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取脚本所在目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def load_iris_data():
    """加载鸢尾花数据集"""
    data_file_path = os.path.join(project_root, 'data', 'exp04_iris_data.txt')
    print(f"数据文件路径: {data_file_path}")

    # 读取数据
    data = pd.read_csv(data_file_path, header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target_name'],
                       skip_blank_lines=True)
    data = data.dropna()

    # 处理类别标签
    target_names_unique = data['target_name'].unique()
    target_name_to_code = {name: idx for idx, name in enumerate(target_names_unique)}
    data['target'] = data['target_name'].map(target_name_to_code)

    # 提取特征和标签
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = data[feature_names].values
    y = data['target'].values

    return X, y, target_names_unique


def one_hot_encode(y, num_classes):
    """将类别标签转换为one-hot编码"""
    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1
    return y_one_hot


def main():
    """主函数"""
    print("=" * 80)
    print("BP神经网络实验 - 鸢尾花分类")
    print("=" * 80)

    # =====================================================================
    # 1. 数据加载与预处理
    # =====================================================================
    print("\n" + "=" * 80)
    print("1. 数据加载与预处理")
    print("=" * 80)

    X, y, target_names = load_iris_data()
    num_classes = len(target_names)

    print(f"\n数据形状: {X.shape}")
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {target_names}")

    # 数据集划分：训练集（60%）、验证集（20%）、测试集（20%）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"\n训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 转换为one-hot编码（用于手动实现）
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes)
    y_test_onehot = one_hot_encode(y_test, num_classes)

    # =====================================================================
    # 2. 手动实现BP神经网络
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. 手动实现BP神经网络")
    print("=" * 80)

    # 设置网络参数
    input_dim = X_train_scaled.shape[1]
    hidden_dim = 10
    output_dim = num_classes
    learning_rate = 0.01
    epochs = 1000

    print(f"\n网络结构:")
    print(f"  输入层: {input_dim} 个神经元")
    print(f"  隐藏层: {hidden_dim} 个神经元 (ReLU激活)")
    print(f"  输出层: {output_dim} 个神经元 (Softmax激活)")
    print(f"  学习率: {learning_rate}")
    print(f"  迭代次数: {epochs}")

    # 创建并训练模型
    manual_model = BPNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        learning_rate=learning_rate
    )

    print("\n开始训练...")
    manual_history = manual_model.train(
        X_train_scaled, y_train_onehot,
        X_val_scaled, y_val_onehot,
        epochs=epochs,
        verbose=True
    )

    # 测试集评估
    manual_test_pred = manual_model.predict(X_test_scaled)
    manual_test_acc = accuracy_score(y_test, manual_test_pred)

    print(f"\n手动实现模型 - 测试集准确率: {manual_test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, manual_test_pred, target_names=target_names))

    # 绘制损失曲线
    manual_model.plot_loss_curve(
        history=manual_history,
        save_path=os.path.join(script_dir, 'manual_bp_loss_curve.png')
    )

    # =====================================================================
    # 3. PyTorch框架实现BP神经网络
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. PyTorch框架实现BP神经网络")
    print("=" * 80)

    # 创建模型
    pytorch_model = BPNeuralNetworkPyTorch(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )

    # 创建训练器
    trainer = BPNeuralNetworkTrainer(
        model=pytorch_model,
        learning_rate=0.001,
        optimizer_name='Adam'
    )

    # 早停法
    early_stopping = EarlyStopping(patience=50, min_delta=0.0001, restore_best_weights=True)

    print("\n开始训练...")
    pytorch_history = trainer.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=epochs,
        batch_size=16,
        early_stopping=early_stopping,
        verbose=True
    )

    # 测试集评估
    pytorch_test_pred = trainer.predict(X_test_scaled)
    pytorch_test_acc = accuracy_score(y_test, pytorch_test_pred)

    print(f"\nPyTorch实现模型 - 测试集准确率: {pytorch_test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, pytorch_test_pred, target_names=target_names))

    # 绘制训练曲线
    trainer.plot_training_curves(
        save_path=os.path.join(script_dir, 'pytorch_bp_training_curves.png')
    )

    # =====================================================================
    # 4. 对比手动实现与框架实现
    # =====================================================================
    print("\n" + "=" * 80)
    print("4. 手动实现 vs PyTorch框架实现对比")
    print("=" * 80)

    comparison_results = {
        '模型': ['手动实现', 'PyTorch实现'],
        '测试集准确率': [manual_test_acc, pytorch_test_acc],
        '最终训练损失': [manual_history['train_loss'][-1], pytorch_history['train_loss'][-1]],
        '最终验证损失': [manual_history['val_loss'][-1] if len(manual_history['val_loss']) > 0 else None,
                       pytorch_history['val_loss'][-1] if len(pytorch_history['val_loss']) > 0 else None]
    }

    comparison_df = pd.DataFrame(comparison_results)
    print("\n模型性能对比:")
    print(comparison_df.to_string(index=False))

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线对比
    axes[0].plot(manual_history['train_loss'], label='手动实现-训练损失', linewidth=2, linestyle='--')
    if len(manual_history['val_loss']) > 0:
        axes[0].plot(manual_history['val_loss'], label='手动实现-验证损失', linewidth=2, linestyle='--')
    axes[0].plot(pytorch_history['train_loss'], label='PyTorch-训练损失', linewidth=2)
    if len(pytorch_history['val_loss']) > 0:
        axes[0].plot(pytorch_history['val_loss'], label='PyTorch-验证损失', linewidth=2)
    axes[0].set_xlabel('迭代次数', fontsize=12)
    axes[0].set_ylabel('损失值', fontsize=12)
    axes[0].set_title('损失曲线对比', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 准确率对比
    axes[1].bar(['手动实现', 'PyTorch实现'], [manual_test_acc, pytorch_test_acc],
                color=['#3498db', '#2ecc71'], alpha=0.7)
    axes[1].set_ylabel('测试集准确率', fontsize=12)
    axes[1].set_title('测试集准确率对比', fontsize=14)
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, acc in enumerate([manual_test_acc, pytorch_test_acc]):
        axes[1].text(i, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    comparison_path = os.path.join(script_dir, 'manual_vs_pytorch_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存为: {comparison_path}")
    plt.close()

    # 混淆矩阵对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_manual = confusion_matrix(y_test, manual_test_pred)
    cm_pytorch = confusion_matrix(y_test, pytorch_test_pred)

    sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0])
    axes[0].set_xlabel('预测标签', fontsize=12)
    axes[0].set_ylabel('真实标签', fontsize=12)
    axes[0].set_title('手动实现 - 混淆矩阵', fontsize=14)

    sns.heatmap(cm_pytorch, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[1])
    axes[1].set_xlabel('预测标签', fontsize=12)
    axes[1].set_ylabel('真实标签', fontsize=12)
    axes[1].set_title('PyTorch实现 - 混淆矩阵', fontsize=14)

    plt.tight_layout()
    confusion_path = os.path.join(script_dir, 'confusion_matrices_comparison.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵对比图已保存为: {confusion_path}")
    plt.close()

    # =====================================================================
    # 5. 实验分析
    # =====================================================================
    print("\n" + "=" * 80)
    print("5. 实验分析（学习率、迭代次数、隐藏层神经元数量的影响）")
    print("=" * 80)

    comprehensive_analysis(
        X_train_scaled, y_train_onehot,
        X_val_scaled, y_val_onehot,
        X_test_scaled, y_test_onehot,
        save_dir=script_dir
    )

    # =====================================================================
    # 总结
    # =====================================================================
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print(f"  1. manual_bp_loss_curve.png - 手动实现损失曲线")
    print(f"  2. pytorch_bp_training_curves.png - PyTorch实现训练曲线")
    print(f"  3. manual_vs_pytorch_comparison.png - 模型对比图")
    print(f"  4. confusion_matrices_comparison.png - 混淆矩阵对比")
    print(f"  5. learning_rate_analysis.png - 学习率分析图")
    print(f"  6. learning_rate_vs_accuracy.png - 学习率-准确率关系图")
    print(f"  7. epochs_analysis.png - 迭代次数分析图")
    print(f"  8. epochs_vs_accuracy.png - 迭代次数-准确率关系图")
    print(f"  9. hidden_neurons_analysis.png - 隐藏层神经元数量分析图")
    print(f"  10. hidden_neurons_vs_accuracy.png - 隐藏层神经元数量-准确率关系图")
    print(f"  11. learning_rate_results.csv - 学习率实验结果")
    print(f"  12. epochs_results.csv - 迭代次数实验结果")
    print(f"  13. hidden_neurons_results.csv - 隐藏层神经元数量实验结果")
    print("=" * 80)


if __name__ == '__main__':
    main()

