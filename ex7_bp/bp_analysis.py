"""
BP神经网络实验分析
分析学习率、迭代次数、隐藏层神经元数量对模型训练效果和性能的影响
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bp_manual import BPNeuralNetwork
from bp_pytorch import BPNeuralNetworkPyTorch, BPNeuralNetworkTrainer, EarlyStopping
import torch
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_learning_rate(X_train, y_train, X_val, y_val, X_test, y_test,
                          learning_rates=[0.001, 0.01, 0.1, 0.5],
                          hidden_dim=10, epochs=1000, save_dir='.'):
    """
    分析学习率对模型性能的影响

    参数:
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集
        learning_rates: 学习率列表
        hidden_dim: 隐藏层神经元数量
        epochs: 迭代次数
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("实验1: 分析学习率对模型性能的影响")
    print("=" * 80)

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        print(f"\n学习率: {lr}")

        # 手动实现
        num_classes = y_test.shape[1] if len(y_test.shape) > 1 else len(np.unique(y_test))
        model = BPNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            learning_rate=lr
        )

        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, verbose=False
        )

        # 测试集评估
        test_pred = model.predict(X_test)
        if len(y_test.shape) > 1:
            test_true = np.argmax(y_test, axis=1)
        else:
            test_true = y_test
        test_acc = np.mean(test_pred == test_true)
        final_loss = history['train_loss'][-1]

        results.append({
            'learning_rate': lr,
            'test_accuracy': test_acc,
            'final_loss': final_loss,
            'epochs_to_converge': len(history['train_loss'])
        })

        # 绘制损失曲线
        axes[idx].plot(history['train_loss'], label='训练损失', linewidth=2)
        if len(history['val_loss']) > 0:
            axes[idx].plot(history['val_loss'], label='验证损失', linewidth=2)
        axes[idx].set_xlabel('迭代次数', fontsize=10)
        axes[idx].set_ylabel('损失值', fontsize=10)
        axes[idx].set_title(f'学习率 = {lr}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

        print(f"  测试集准确率: {test_acc:.4f}")
        print(f"  最终损失: {final_loss:.4f}")

    plt.suptitle('不同学习率下的训练损失曲线', fontsize=14, y=1.0)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'learning_rate_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n学习率分析图已保存为: {save_path}")
    plt.close()

    # 绘制学习率与准确率的关系
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df['learning_rate'], df['test_accuracy'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('学习率', fontsize=12)
    plt.ylabel('测试集准确率', fontsize=12)
    plt.title('学习率对模型准确率的影响', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, 'learning_rate_vs_accuracy.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"学习率-准确率关系图已保存为: {save_path2}")
    plt.close()

    return results


def analyze_epochs(X_train, y_train, X_val, y_val, X_test, y_test,
                   epochs_list=[100, 500, 1000, 2000],
                   hidden_dim=10, learning_rate=0.01, save_dir='.'):
    """
    分析迭代次数对模型性能的影响

    参数:
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集
        epochs_list: 迭代次数列表
        hidden_dim: 隐藏层神经元数量
        learning_rate: 学习率
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("实验2: 分析迭代次数对模型性能的影响")
    print("=" * 80)

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, epochs in enumerate(epochs_list):
        print(f"\n迭代次数: {epochs}")

        # 手动实现
        num_classes = y_test.shape[1] if len(y_test.shape) > 1 else len(np.unique(y_test))
        model = BPNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            learning_rate=learning_rate
        )

        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, verbose=False
        )

        # 测试集评估
        test_pred = model.predict(X_test)
        if len(y_test.shape) > 1:
            test_true = np.argmax(y_test, axis=1)
        else:
            test_true = y_test
        test_acc = np.mean(test_pred == test_true)
        final_loss = history['train_loss'][-1]

        results.append({
            'epochs': epochs,
            'test_accuracy': test_acc,
            'final_loss': final_loss
        })

        # 绘制损失曲线
        axes[idx].plot(history['train_loss'], label='训练损失', linewidth=2)
        if len(history['val_loss']) > 0:
            axes[idx].plot(history['val_loss'], label='验证损失', linewidth=2)
        axes[idx].set_xlabel('迭代次数', fontsize=10)
        axes[idx].set_ylabel('损失值', fontsize=10)
        axes[idx].set_title(f'迭代次数 = {epochs}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

        print(f"  测试集准确率: {test_acc:.4f}")
        print(f"  最终损失: {final_loss:.4f}")

    plt.suptitle('不同迭代次数下的训练损失曲线', fontsize=14, y=1.0)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'epochs_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n迭代次数分析图已保存为: {save_path}")
    plt.close()

    # 绘制迭代次数与准确率的关系
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epochs'], df['test_accuracy'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('测试集准确率', fontsize=12)
    plt.title('迭代次数对模型准确率的影响', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, 'epochs_vs_accuracy.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"迭代次数-准确率关系图已保存为: {save_path2}")
    plt.close()

    return results


def analyze_hidden_neurons(X_train, y_train, X_val, y_val, X_test, y_test,
                          hidden_dims=[5, 10, 20, 50],
                          learning_rate=0.01, epochs=1000, save_dir='.'):
    """
    分析隐藏层神经元数量对模型性能的影响

    参数:
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集
        hidden_dims: 隐藏层神经元数量列表
        learning_rate: 学习率
        epochs: 迭代次数
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("实验3: 分析隐藏层神经元数量对模型性能的影响")
    print("=" * 80)

    results = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, hidden_dim in enumerate(hidden_dims):
        print(f"\n隐藏层神经元数量: {hidden_dim}")

        # 手动实现
        num_classes = y_test.shape[1] if len(y_test.shape) > 1 else len(np.unique(y_test))
        model = BPNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            learning_rate=learning_rate
        )

        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, verbose=False
        )

        # 测试集评估
        test_pred = model.predict(X_test)
        if len(y_test.shape) > 1:
            test_true = np.argmax(y_test, axis=1)
        else:
            test_true = y_test
        test_acc = np.mean(test_pred == test_true)
        final_loss = history['train_loss'][-1]

        results.append({
            'hidden_neurons': hidden_dim,
            'test_accuracy': test_acc,
            'final_loss': final_loss
        })

        # 绘制损失曲线
        axes[idx].plot(history['train_loss'], label='训练损失', linewidth=2)
        if len(history['val_loss']) > 0:
            axes[idx].plot(history['val_loss'], label='验证损失', linewidth=2)
        axes[idx].set_xlabel('迭代次数', fontsize=10)
        axes[idx].set_ylabel('损失值', fontsize=10)
        axes[idx].set_title(f'隐藏层神经元数 = {hidden_dim}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

        print(f"  测试集准确率: {test_acc:.4f}")
        print(f"  最终损失: {final_loss:.4f}")

    plt.suptitle('不同隐藏层神经元数量下的训练损失曲线', fontsize=14, y=1.0)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'hidden_neurons_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n隐藏层神经元数量分析图已保存为: {save_path}")
    plt.close()

    # 绘制隐藏层神经元数量与准确率的关系
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df['hidden_neurons'], df['test_accuracy'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('隐藏层神经元数量', fontsize=12)
    plt.ylabel('测试集准确率', fontsize=12)
    plt.title('隐藏层神经元数量对模型准确率的影响', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, 'hidden_neurons_vs_accuracy.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"隐藏层神经元数量-准确率关系图已保存为: {save_path2}")
    plt.close()

    return results


def comprehensive_analysis(X_train, y_train, X_val, y_val, X_test, y_test, save_dir='.'):
    """
    综合实验分析

    参数:
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("综合实验分析")
    print("=" * 80)

    # 实验1: 学习率分析
    lr_results = analyze_learning_rate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        learning_rates=[0.001, 0.01, 0.1, 0.5],
        hidden_dim=10,
        epochs=1000,
        save_dir=save_dir
    )

    # 实验2: 迭代次数分析
    epochs_results = analyze_epochs(
        X_train, y_train, X_val, y_val, X_test, y_test,
        epochs_list=[100, 500, 1000, 2000],
        hidden_dim=10,
        learning_rate=0.01,
        save_dir=save_dir
    )

    # 实验3: 隐藏层神经元数量分析
    hidden_results = analyze_hidden_neurons(
        X_train, y_train, X_val, y_val, X_test, y_test,
        hidden_dims=[5, 10, 20, 50],
        learning_rate=0.01,
        epochs=1000,
        save_dir=save_dir
    )

    # 生成综合报告
    print("\n" + "=" * 80)
    print("实验分析总结")
    print("=" * 80)

    print("\n1. 学习率分析结果:")
    lr_df = pd.DataFrame(lr_results)
    print(lr_df.to_string(index=False))

    print("\n2. 迭代次数分析结果:")
    epochs_df = pd.DataFrame(epochs_results)
    print(epochs_df.to_string(index=False))

    print("\n3. 隐藏层神经元数量分析结果:")
    hidden_df = pd.DataFrame(hidden_results)
    print(hidden_df.to_string(index=False))

    # 保存结果到CSV
    lr_df.to_csv(os.path.join(save_dir, 'learning_rate_results.csv'), index=False, encoding='utf-8-sig')
    epochs_df.to_csv(os.path.join(save_dir, 'epochs_results.csv'), index=False, encoding='utf-8-sig')
    hidden_df.to_csv(os.path.join(save_dir, 'hidden_neurons_results.csv'), index=False, encoding='utf-8-sig')

    print("\n所有实验结果已保存到CSV文件")

