"""
模型优化模块
包含模型结构优化、训练参数优化、数据增强等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from model import ImprovedLeNet5
from main import load_mnist_data, train_model, evaluate_model, set_seed

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))


class EnhancedLeNet5(nn.Module):
    """
    增强版LeNet-5模型（增加卷积层数量）
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(EnhancedLeNet5, self).__init__()

        # 三个卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 全连接层（更深的网络）
        self.fc1 = nn.Linear(3 * 3 * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # 14×14

        # 第二个卷积块
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # 7×7

        # 第三个卷积块
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # 3×3

        # 扁平化
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


def analyze_learning_rate(train_loader, test_loader, learning_rates=[0.0001, 0.001, 0.01, 0.1],
                         epochs=10, device='cpu', save_dir=None):
    """
    分析不同学习率对模型性能的影响

    参数:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        learning_rates: 要测试的学习率列表
        epochs: 训练轮数
        device: 训练设备
        save_dir: 保存目录

    返回:
        results: 实验结果列表
    """
    print("\n" + "=" * 80)
    print("学习率优化分析")
    print("=" * 80)

    results = []

    for lr in learning_rates:
        print(f"\n测试学习率: {lr}")
        print("-" * 80)

        set_seed(42)  # 确保每次实验的初始状态一致

        model = ImprovedLeNet5(num_classes=10, dropout_rate=0.5)
        history = train_model(model, train_loader, test_loader, epochs=epochs,
                            learning_rate=lr, device=device)

        test_acc, _, _ = evaluate_model(model, test_loader, device=device)

        results.append({
            'learning_rate': lr,
            'test_accuracy': test_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        })

    # 绘制结果
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(df['learning_rate'], df['test_accuracy'], 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('学习率', fontsize=12)
    axes[0].set_ylabel('测试准确率 (%)', fontsize=12)
    axes[0].set_title('学习率对测试准确率的影响', fontsize=14, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['learning_rate'], df['final_val_loss'], 'r-s', linewidth=2, markersize=8)
    axes[1].set_xlabel('学习率', fontsize=12)
    axes[1].set_ylabel('验证损失', fontsize=12)
    axes[1].set_title('学习率对验证损失的影响', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'learning_rate_analysis.png') if save_dir else os.path.join(script_dir, 'learning_rate_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n学习率分析图已保存至: {save_path}")
    plt.close()

    # 保存结果到CSV
    csv_path = os.path.join(save_dir, 'learning_rate_results.csv') if save_dir else os.path.join(script_dir, 'learning_rate_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"学习率实验结果已保存至: {csv_path}")

    print("\n学习率分析结果:")
    print(df.to_string(index=False))

    return results


def analyze_batch_size(train_loader, test_loader, batch_sizes=[16, 32, 64, 128],
                      epochs=10, learning_rate=0.001, device='cpu', save_dir=None):
    """
    分析不同批次大小对模型性能的影响

    参数:
        train_loader: 训练数据加载器（原始）
        test_loader: 测试数据加载器
        batch_sizes: 要测试的批次大小列表
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_dir: 保存目录

    返回:
        results: 实验结果列表
    """
    print("\n" + "=" * 80)
    print("批次大小优化分析")
    print("=" * 80)

    # 获取原始数据集
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    results = []

    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        print("-" * 80)

        set_seed(42)

        # 创建新的数据加载器
        new_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        model = ImprovedLeNet5(num_classes=10, dropout_rate=0.5)
        history = train_model(model, new_train_loader, test_loader, epochs=epochs,
                            learning_rate=learning_rate, device=device)

        test_acc, _, _ = evaluate_model(model, test_loader, device=device)

        results.append({
            'batch_size': batch_size,
            'test_accuracy': test_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1]
        })

    # 绘制结果
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(df['batch_size'], df['test_accuracy'], 'b-o', linewidth=2, markersize=8)
    plt.xlabel('批次大小', fontsize=12)
    plt.ylabel('测试准确率 (%)', fontsize=12)
    plt.title('批次大小对测试准确率的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'batch_size_analysis.png') if save_dir else os.path.join(script_dir, 'batch_size_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n批次大小分析图已保存至: {save_path}")
    plt.close()

    print("\n批次大小分析结果:")
    print(df.to_string(index=False))

    return results


def analyze_dropout_rate(train_loader, test_loader, dropout_rates=[0.0, 0.3, 0.5, 0.7],
                        epochs=10, learning_rate=0.001, device='cpu', save_dir=None):
    """
    分析不同Dropout比率对模型性能的影响

    参数:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        dropout_rates: 要测试的Dropout比率列表
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_dir: 保存目录

    返回:
        results: 实验结果列表
    """
    print("\n" + "=" * 80)
    print("Dropout比率优化分析")
    print("=" * 80)

    results = []

    for dropout_rate in dropout_rates:
        print(f"\n测试Dropout比率: {dropout_rate}")
        print("-" * 80)

        set_seed(42)

        model = ImprovedLeNet5(num_classes=10, dropout_rate=dropout_rate)
        history = train_model(model, train_loader, test_loader, epochs=epochs,
                            learning_rate=learning_rate, device=device)

        test_acc, _, _ = evaluate_model(model, test_loader, device=device)

        results.append({
            'dropout_rate': dropout_rate,
            'test_accuracy': test_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1]
        })

    # 绘制结果
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(df['dropout_rate'], df['test_accuracy'], 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Dropout比率', fontsize=12)
    plt.ylabel('测试准确率 (%)', fontsize=12)
    plt.title('Dropout比率对测试准确率的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'dropout_analysis.png') if save_dir else os.path.join(script_dir, 'dropout_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nDropout分析图已保存至: {save_path}")
    plt.close()

    print("\nDropout分析结果:")
    print(df.to_string(index=False))

    return results


def compare_data_augmentation(train_loader_no_aug, test_loader, epochs=10,
                             learning_rate=0.001, device='cpu', save_dir=None):
    """
    比较使用数据增强前后的模型性能

    参数:
        train_loader_no_aug: 无数据增强的训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("数据增强效果对比分析")
    print("=" * 80)

    results = []

    # 不使用数据增强
    print("\n1. 不使用数据增强")
    print("-" * 80)
    set_seed(42)
    model_no_aug = ImprovedLeNet5(num_classes=10, dropout_rate=0.5)
    history_no_aug = train_model(model_no_aug, train_loader_no_aug, test_loader,
                                 epochs=epochs, learning_rate=learning_rate, device=device)
    test_acc_no_aug, _, _ = evaluate_model(model_no_aug, test_loader, device=device)

    results.append({
        'data_augmentation': '否',
        'test_accuracy': test_acc_no_aug,
        'final_train_acc': history_no_aug['train_acc'][-1],
        'final_val_acc': history_no_aug['val_acc'][-1]
    })

    # 使用数据增强
    print("\n2. 使用数据增强")
    print("-" * 80)
    train_loader_aug, _ = load_mnist_data(data_augmentation=True)

    set_seed(42)
    model_aug = ImprovedLeNet5(num_classes=10, dropout_rate=0.5)
    history_aug = train_model(model_aug, train_loader_aug, test_loader,
                             epochs=epochs, learning_rate=learning_rate, device=device)
    test_acc_aug, _, _ = evaluate_model(model_aug, test_loader, device=device)

    results.append({
        'data_augmentation': '是',
        'test_accuracy': test_acc_aug,
        'final_train_acc': history_aug['train_acc'][-1],
        'final_val_acc': history_aug['val_acc'][-1]
    })

    # 绘制对比图
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 准确率对比
    axes[0].bar(df['data_augmentation'], df['test_accuracy'], color=['skyblue', 'lightcoral'])
    axes[0].set_ylabel('测试准确率 (%)', fontsize=12)
    axes[0].set_title('数据增强对测试准确率的影响', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 训练过程对比
    epochs_range = range(1, len(history_no_aug['val_acc']) + 1)
    axes[1].plot(epochs_range, history_no_aug['val_acc'], 'b-o', label='无数据增强', linewidth=2)
    axes[1].plot(epochs_range, history_aug['val_acc'], 'r-s', label='有数据增强', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('验证准确率 (%)', fontsize=12)
    axes[1].set_title('训练过程对比', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'data_augmentation_comparison.png') if save_dir else os.path.join(script_dir, 'data_augmentation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n数据增强对比图已保存至: {save_path}")
    plt.close()

    print("\n数据增强对比结果:")
    print(df.to_string(index=False))

    improvement = test_acc_aug - test_acc_no_aug
    print(f"\n数据增强带来的准确率提升: {improvement:.2f}%")

    return results


def comprehensive_optimization(train_loader, test_loader, device='cpu', save_dir=None):
    """
    综合优化分析

    参数:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 训练设备
        save_dir: 保存目录
    """
    print("\n" + "=" * 80)
    print("综合模型优化分析")
    print("=" * 80)

    if save_dir is None:
        save_dir = script_dir

    # 1. 学习率分析
    lr_results = analyze_learning_rate(
        train_loader, test_loader,
        learning_rates=[0.0001, 0.001, 0.01],
        epochs=5,  # 减少epochs以节省时间
        device=device,
        save_dir=save_dir
    )

    # 2. 批次大小分析
    batch_results = analyze_batch_size(
        train_loader, test_loader,
        batch_sizes=[16, 32, 64],
        epochs=5,
        device=device,
        save_dir=save_dir
    )

    # 3. Dropout分析
    dropout_results = analyze_dropout_rate(
        train_loader, test_loader,
        dropout_rates=[0.3, 0.5, 0.7],
        epochs=5,
        device=device,
        save_dir=save_dir
    )

    print("\n" + "=" * 80)
    print("优化分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    # 示例：运行综合优化分析
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data(data_augmentation=False)
    comprehensive_optimization(train_loader, test_loader, device=device)

