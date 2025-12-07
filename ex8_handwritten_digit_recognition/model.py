"""
CNN模型定义 - 基于LeNet-5改进
用于MNIST手写数字识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedLeNet5(nn.Module):
    """
    改进的LeNet-5模型用于MNIST手写数字识别

    网络结构：
    - 输入层（28×28×1）
    - 卷积层1（32个3×3卷积核，ReLU激活）
    - 池化层1（2×2最大池化）
    - 卷积层2（64个3×3卷积核，ReLU激活）
    - 池化层2（2×2最大池化）
    - 扁平化层
    - 全连接层（128个神经元，ReLU激活）
    - Dropout层（dropout rate=0.5）
    - 输出层（10个神经元，Softmax激活）
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        初始化模型

        参数:
            num_classes: 分类类别数（MNIST为10）
            dropout_rate: Dropout比率，默认0.5
        """
        super(ImprovedLeNet5, self).__init__()

        # 第一个卷积块
        # 输入: 28×28×1
        # 输出: 28×28×32 (padding=1保持尺寸)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 池化后: 14×14×32 (28/2=14)

        # 第二个卷积块
        # 输入: 14×14×32
        # 输出: 14×14×64 (padding=1保持尺寸)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 池化后: 7×7×64 (14/2=7)

        # 扁平化后: 7×7×64 = 3136
        # 全连接层
        self.fc1 = nn.Linear(7 * 7 * 64, 128)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)

        返回:
            out: 输出张量，形状为 (batch_size, num_classes)
        """
        # 第一个卷积块
        x = self.conv1(x)  # (batch_size, 32, 28, 28)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (batch_size, 32, 14, 14)

        # 第二个卷积块
        x = self.conv2(x)  # (batch_size, 64, 14, 14)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (batch_size, 64, 7, 7)

        # 扁平化
        x = x.view(x.size(0), -1)  # (batch_size, 7*7*64) = (batch_size, 3136)

        # 全连接层
        x = self.fc1(x)  # (batch_size, 128)
        x = F.relu(x)

        # Dropout（仅在训练时生效）
        x = self.dropout(x)

        # 输出层（Softmax在损失函数中处理，这里不需要显式添加）
        x = self.fc2(x)  # (batch_size, 10)

        return x

    def get_model_info(self):
        """
        获取模型信息（层信息、参数数量等）

        返回:
            model_info: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 计算各层参数
        layer_info = []
        for name, param in self.named_parameters():
            layer_info.append({
                'layer': name,
                'shape': list(param.shape),
                'parameters': param.numel(),
                'trainable': param.requires_grad
            })

        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_details': layer_info,
            'model_structure': str(self)
        }

        return model_info

    def print_model_info(self):
        """打印模型结构和参数信息"""
        print("\n" + "=" * 80)
        print("模型结构信息")
        print("=" * 80)
        print(self)

        print("\n" + "=" * 80)
        print("模型参数统计")
        print("=" * 80)

        total_params = 0
        trainable_params = 0

        print("\n各层参数详情:")
        print("-" * 80)
        print(f"{'层名称':<30} {'形状':<25} {'参数数量':<15} {'是否可训练':<10}")
        print("-" * 80)

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            print(f"{name:<30} {str(list(param.shape)):<25} {num_params:<15} {str(param.requires_grad):<10}")

        print("-" * 80)
        print(f"\n总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"不可训练参数数量: {total_params - trainable_params:,}")

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }

