"""
MNIST手写数字识别 - 主程序
使用改进的LeNet-5 CNN模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import random

from model import ImprovedLeNet5

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))


def load_mnist_data(data_augmentation=False):
    """
    加载MNIST数据集

    参数:
        data_augmentation: 是否使用数据增强

    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("\n" + "=" * 80)
    print("1. 数据集准备")
    print("=" * 80)

    # 数据预处理：归一化到[0,1]
    transform_train = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量，并自动归一化到[0,1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量，并自动归一化到[0,1]
    ])

    # 可选的数据增强
    if data_augmentation:
        print("\n使用数据增强：随机旋转、平移")
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            transforms.ToTensor(),
        ])

    # 加载训练集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    # 加载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    print(f"\n训练集大小: {len(train_dataset)} 张图像")
    print(f"测试集大小: {len(test_dataset)} 张图像")
    print(f"图像尺寸: 28×28 灰度图")
    print(f"类别数量: 10 (数字 0-9)")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Windows系统设置为0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0  # Windows系统设置为0
    )

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001, device='cpu'):
    """
    训练模型

    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器（作为验证集）
        epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备（cpu或cuda）

    返回:
        history: 训练历史记录字典
    """
    print("\n" + "=" * 80)
    print("4. 模型训练与监控")
    print("=" * 80)

    # 将模型移动到指定设备
    model = model.to(device)

    # 配置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 稀疏分类交叉熵（CrossEntropyLoss包含Softmax）

    print(f"\n优化器: Adam")
    print(f"学习率: {learning_rate}")
    print(f"损失函数: 交叉熵损失（CrossEntropyLoss）")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: 32")
    print(f"训练设备: {device}")

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print("\n开始训练...")
    print("-" * 80)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # 计算训练准确率和损失
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # 验证阶段（使用测试集作为验证集）
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # 计算验证准确率和损失
        val_loss /= len(test_loader)
        val_acc = 100.0 * val_correct / val_total

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% | "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

    print("-" * 80)
    print("训练完成！")

    return history


def plot_training_history(history, save_path=None):
    """
    可视化训练过程

    参数:
        history: 训练历史记录字典
        save_path: 保存路径（可选）
    """
    print("\n" + "=" * 80)
    print("可视化训练过程")
    print("=" * 80)

    epochs = range(1, len(history['train_loss']) + 1)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制准确率曲线
    axes[0].plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_acc'], 'r-s', label='验证准确率', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('准确率 (%)', fontsize=12)
    axes[0].set_title('训练准确率与验证准确率曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])

    # 绘制损失曲线
    axes[1].plot(epochs, history['train_loss'], 'b-o', label='训练损失', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_loss'], 'r-s', label='验证损失', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('损失值', fontsize=12)
    axes[1].set_title('训练损失与验证损失曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存至: {save_path}")
    else:
        plt.savefig(os.path.join(script_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存至: training_curves.png")

    plt.close()

    # 分析过拟合情况
    print("\n过拟合分析:")
    print("-" * 80)
    if len(history['val_acc']) >= 3:
        # 检查验证准确率是否在后期下降
        last_three_val_acc = history['val_acc'][-3:]
        if last_three_val_acc[0] > last_three_val_acc[-1]:
            print("⚠ 检测到过拟合：验证准确率在后期出现下降趋势")
        else:
            print("✓ 未检测到明显的过拟合现象")

        # 检查验证损失是否在后期上升
        last_three_val_loss = history['val_loss'][-3:]
        if last_three_val_loss[0] < last_three_val_loss[-1]:
            print("⚠ 检测到过拟合：验证损失在后期出现上升趋势")
        else:
            print("✓ 验证损失保持下降或稳定趋势")


def evaluate_model(model, test_loader, device='cpu', save_dir=None):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备（cpu或cuda）
        save_dir: 保存目录

    返回:
        test_acc: 测试准确率
        cm: 混淆矩阵
        report: 分类报告
    """
    print("\n" + "=" * 80)
    print("5. 模型评估与预测")
    print("=" * 80)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # 计算测试准确率
    test_acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\n测试准确率: {test_acc:.2f}%")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else os.path.join(script_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {cm_path}")
    plt.close()

    # 分类报告
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])
    print("\n分类报告:")
    print("-" * 80)
    print(report)

    # 分析每个类别的准确率
    print("\n各类别分类效果分析:")
    print("-" * 80)
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    for i, acc in enumerate(class_acc):
        print(f"数字 {i}: {acc:.2f}%")

    return test_acc, cm, report


def predict_random_samples(model, test_loader, num_samples=10, device='cpu', save_dir=None):
    """
    随机选取测试集中的图像进行预测

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_samples: 预测的样本数量
        device: 设备（cpu或cuda）
        save_dir: 保存目录
    """
    print(f"\n随机预测 {num_samples} 张测试图像:")
    print("-" * 80)

    model.eval()

    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # 随机选择样本
    indices = random.sample(range(len(images)), min(num_samples, len(images)))

    # 预测
    with torch.no_grad():
        outputs = model(images[indices])
        _, predicted = torch.max(outputs, 1)

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = images[idx].cpu().squeeze()
        true_label = labels[idx].item()
        pred_label = predicted[i].item()
        confidence = torch.softmax(outputs[i], dim=0)[pred_label].item() * 100

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'真实: {true_label}\n预测: {pred_label} ({confidence:.1f}%)',
                         fontsize=10, color='green' if true_label == pred_label else 'red')
        axes[i].axis('off')

    plt.suptitle('随机预测结果对比', fontsize=14, fontweight='bold')
    plt.tight_layout()

    pred_path = os.path.join(save_dir, 'random_predictions.png') if save_dir else os.path.join(script_dir, 'random_predictions.png')
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    print(f"预测结果可视化已保存至: {pred_path}")
    plt.close()

    # 打印预测结果
    correct = 0
    for i, idx in enumerate(indices):
        true_label = labels[idx].item()
        pred_label = predicted[i].item()
        is_correct = true_label == pred_label
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"{status} 样本 {i+1}: 真实标签={true_label}, 预测标签={pred_label}")

    print(f"\n预测准确率: {correct}/{num_samples} ({100.0*correct/num_samples:.1f}%)")


def main():
    """主函数"""
    print("=" * 80)
    print("MNIST手写数字识别 - 基于改进LeNet-5 CNN模型")
    print("=" * 80)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # =====================================================================
    # 1. 数据集准备
    # =====================================================================
    train_loader, test_loader = load_mnist_data(data_augmentation=False)

    # =====================================================================
    # 2. 图像数据预处理（已在数据加载时完成）
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. 图像数据预处理")
    print("=" * 80)
    print("\n预处理步骤:")
    print("  - 数据格式转换: 图像数据已转换为PyTorch张量格式")
    print("  - 归一化: 像素值已从 [0,255] 转换为 [0,1]")
    print("  - 数据增强: 当前未启用（可在main函数中启用）")

    # =====================================================================
    # 3. 构建CNN模型
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. 构建CNN模型（基于LeNet-5改进）")
    print("=" * 80)

    model = ImprovedLeNet5(num_classes=10, dropout_rate=0.5)
    model.print_model_info()

    # =====================================================================
    # 4. 模型训练与监控
    # =====================================================================
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=10,
        learning_rate=0.001,
        device=device
    )

    # 可视化训练过程
    plot_training_history(history, save_path=os.path.join(script_dir, 'training_curves.png'))

    # =====================================================================
    # 5. 模型评估与预测
    # =====================================================================
    test_acc, cm, report = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=script_dir
    )

    # 随机预测
    predict_random_samples(
        model=model,
        test_loader=test_loader,
        num_samples=10,
        device=device,
        save_dir=script_dir
    )

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"\n最终测试准确率: {test_acc:.2f}%")
    print(f"\n生成的文件:")
    print(f"  - training_curves.png: 训练过程曲线")
    print(f"  - confusion_matrix.png: 混淆矩阵")
    print(f"  - random_predictions.png: 随机预测结果")


if __name__ == '__main__':
    main()

