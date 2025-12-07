"""
使用PyTorch框架实现BP神经网络
包含：Sequential模型、优化器配置、早停法、验证集监控
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class EarlyStopping:
    """早停法类，用于防止过拟合"""

    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        参数:
            patience: 容忍验证损失不下降的轮数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        检查是否应该早停

        返回:
            should_stop: 是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_weights(self, model):
        """恢复最佳权重"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class BPNeuralNetworkPyTorch(nn.Module):
    """使用PyTorch实现的BP神经网络"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化BP神经网络

        参数:
            input_dim: 输入层维度
            hidden_dim: 隐藏层维度（神经元数量）
            output_dim: 输出层维度
        """
        super(BPNeuralNetworkPyTorch, self).__init__()

        # 使用Sequential模型构建网络
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # 注意：Softmax在损失函数中处理，这里不需要显式添加
        )

    def forward(self, x):
        """前向传播"""
        return self.model(x)


class BPNeuralNetworkTrainer:
    """BP神经网络训练器"""

    def __init__(self, model, learning_rate=0.001, optimizer_name='Adam'):
        """
        初始化训练器

        参数:
            model: PyTorch模型
            learning_rate: 学习率
            optimizer_name: 优化器名称（'Adam' 或 'SGD'）
        """
        self.model = model
        self.learning_rate = learning_rate

        # 配置优化器
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 损失函数：交叉熵损失（包含Softmax）
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            # 前向传播
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1000, batch_size=32, early_stopping=None, verbose=True):
        """
        训练模型

        参数:
            X_train: 训练集特征 (numpy array)
            y_train: 训练集标签 (numpy array，类别索引)
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选，类别索引）
            epochs: 迭代次数
            batch_size: 批次大小
            early_stopping: 早停法对象（可选）
            verbose: 是否打印训练过程

        返回:
            history: 训练历史字典
        """
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练循环
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # 验证
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # 早停检查
                if early_stopping is not None:
                    if early_stopping(val_loss, self.model):
                        if verbose:
                            print(f"早停触发于 Epoch {epoch+1}")
                        if early_stopping.restore_best_weights:
                            early_stopping.restore_weights(self.model)
                        break
            else:
                val_loss, val_acc = None, None

            # 打印进度
            if verbose and (epoch + 1) % 100 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        return self.history

    def predict(self, X):
        """
        预测

        参数:
            X: 输入数据 (numpy array)

        返回:
            predictions: 预测类别 (numpy array)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.numpy()

    def plot_training_curves(self, save_path=None):
        """
        绘制训练曲线

        参数:
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        axes[0].plot(self.history['train_loss'], label='训练损失', linewidth=2)
        if len(self.history['val_loss']) > 0:
            axes[0].plot(self.history['val_loss'], label='验证损失', linewidth=2)
        axes[0].set_xlabel('迭代次数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('训练过程 - 损失曲线', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(self.history['train_acc'], label='训练准确率', linewidth=2)
        if len(self.history['val_acc']) > 0:
            axes[1].plot(self.history['val_acc'], label='验证准确率', linewidth=2)
        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('准确率', fontsize=12)
        axes[1].set_title('训练过程 - 准确率曲线', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存为: {save_path}")
        else:
            plt.show()

        plt.close()

