"""
手动实现BP神经网络
包含：前向传播、反向传播、参数更新、训练过程
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class BPNeuralNetwork:
    """手动实现的BP神经网络类"""

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, random_seed=42):
        """
        初始化BP神经网络

        参数:
            input_dim: 输入层维度
            hidden_dim: 隐藏层维度（神经元数量）
            output_dim: 输出层维度
            learning_rate: 学习率
            random_seed: 随机种子
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # 设置随机种子
        np.random.seed(random_seed)

        # 初始化权重和偏置
        # 输入层到隐藏层
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))

        # 隐藏层到输出层
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # 存储训练历史
        self.loss_history = []

    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax激活函数（用于多分类）"""
        # 防止数值溢出
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, return_intermediate=False):
        """
        前向传播

        参数:
            X: 输入数据 (batch_size, input_dim)
            return_intermediate: 是否返回中间结果

        返回:
            output: 输出层输出 (batch_size, output_dim)
            如果 return_intermediate=True，还返回 (z1, a1)
        """
        # 输入层到隐藏层
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)  # 隐藏层使用ReLU激活

        # 隐藏层到输出层
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)  # 输出层使用Softmax激活（多分类）

        if return_intermediate:
            return a2, z1, a1
        return a2

    def backward(self, X, y, output, z1, a1):
        """
        反向传播

        参数:
            X: 输入数据 (batch_size, input_dim)
            y: 真实标签 (batch_size, output_dim) - one-hot编码
            output: 前向传播的输出 (batch_size, output_dim)
            z1: 隐藏层激活前的值 (batch_size, hidden_dim)
            a1: 隐藏层激活后的值 (batch_size, hidden_dim)
        """
        m = X.shape[0]  # 样本数量

        # 输出层误差
        delta2 = output - y  # (batch_size, output_dim)

        # 隐藏层误差
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(z1)  # (batch_size, hidden_dim)

        # 计算梯度
        dW2 = (1.0 / m) * np.dot(a1.T, delta2)  # (hidden_dim, output_dim)
        db2 = (1.0 / m) * np.sum(delta2, axis=0, keepdims=True)  # (1, output_dim)

        dW1 = (1.0 / m) * np.dot(X.T, delta1)  # (input_dim, hidden_dim)
        db1 = (1.0 / m) * np.sum(delta1, axis=0, keepdims=True)  # (1, hidden_dim)

        # 更新参数（梯度下降）
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def compute_loss(self, y_true, y_pred):
        """
        计算交叉熵损失

        参数:
            y_true: 真实标签 (batch_size, output_dim) - one-hot编码
            y_pred: 预测输出 (batch_size, output_dim)

        返回:
            loss: 平均损失值
        """
        # 防止log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # 交叉熵损失
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def predict(self, X):
        """
        预测

        参数:
            X: 输入数据 (batch_size, input_dim)

        返回:
            predictions: 预测类别 (batch_size,)
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, verbose=True):
        """
        训练模型

        参数:
            X_train: 训练集特征 (n_samples, input_dim)
            y_train: 训练集标签 (n_samples, output_dim) - one-hot编码
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选，one-hot编码）
            epochs: 迭代次数
            verbose: 是否打印训练过程

        返回:
            history: 训练历史字典
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            # 前向传播（返回中间结果用于反向传播）
            output, z1, a1 = self.forward(X_train, return_intermediate=True)

            # 计算损失
            train_loss = self.compute_loss(y_train, output)
            self.loss_history.append(train_loss)
            history['train_loss'].append(train_loss)

            # 计算训练集准确率
            train_pred = np.argmax(output, axis=1)
            train_true = np.argmax(y_train, axis=1)
            train_acc = np.mean(train_pred == train_true)
            history['train_acc'].append(train_acc)

            # 验证集评估
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_output)
                history['val_loss'].append(val_loss)

                val_pred = np.argmax(val_output, axis=1)
                val_true = np.argmax(y_val, axis=1)
                val_acc = np.mean(val_pred == val_true)
                history['val_acc'].append(val_acc)

            # 反向传播（使用训练集的前向传播中间结果）
            self.backward(X_train, y_train, output, z1, a1)

            # 打印训练进度
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        return history

    def plot_loss_curve(self, history=None, save_path=None):
        """
        绘制损失曲线

        参数:
            history: 训练历史（可选，如果提供则绘制训练和验证损失）
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=(10, 6))

        if history is not None and 'val_loss' in history and len(history['val_loss']) > 0:
            # 绘制训练和验证损失
            plt.plot(history['train_loss'], label='训练损失', linewidth=2)
            plt.plot(history['val_loss'], label='验证损失', linewidth=2)
            plt.xlabel('迭代次数', fontsize=12)
            plt.ylabel('损失值', fontsize=12)
            plt.title('BP神经网络训练过程 - 损失曲线', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
        else:
            # 只绘制训练损失
            plt.plot(self.loss_history, label='训练损失', linewidth=2)
            plt.xlabel('迭代次数', fontsize=12)
            plt.ylabel('损失值', fontsize=12)
            plt.title('BP神经网络训练过程 - 损失曲线', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"损失曲线已保存为: {save_path}")
        else:
            plt.show()

        plt.close()

