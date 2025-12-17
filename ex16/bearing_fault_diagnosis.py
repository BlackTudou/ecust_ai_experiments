"""
工业故障诊断：基于CNN和LSTM的轴承故障诊断
使用深度学习模型（CNN和LSTM）对轴承故障进行诊断
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy import signal
from scipy.signal import stft
import warnings
import os

# 尝试导入PyWavelets，如果没有安装则使用替代方案
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("警告: PyWavelets未安装，将使用简化的去噪方法")

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
# 1. 数据生成（模拟CWRU轴承故障数据）
# =====================================================================
def generate_bearing_fault_data(num_samples_per_class=500, sample_length=1024,
                                 sampling_rate=12000, noise_level=0.1, random_seed=42):
    """
    生成模拟的轴承故障振动数据

    参数:
        num_samples_per_class: 每个类别的样本数
        sample_length: 每个样本的长度（数据点数）
        sampling_rate: 采样频率（Hz）
        noise_level: 噪声水平
        random_seed: 随机种子

    返回:
        X: 振动数据 (num_samples, sample_length)
        y: 故障类别标签 (num_samples,)
    """
    print("\n" + "=" * 80)
    print("1. 生成模拟轴承故障数据")
    print("=" * 80)

    np.random.seed(random_seed)

    # 故障类别：0-正常, 1-内圈故障, 2-外圈故障, 3-滚动体故障
    num_classes = 4
    X = []
    y = []

    # 时间轴
    t = np.arange(sample_length) / sampling_rate

    for class_id in range(num_classes):
        print(f"\n生成类别 {class_id} 的数据...")
        for i in range(num_samples_per_class):
            # 基础频率（轴承特征频率）
            if class_id == 0:  # 正常
                base_freq = 50  # 转频
                amplitude = 1.0
            elif class_id == 1:  # 内圈故障
                base_freq = 50
                fault_freq = 5.4 * base_freq  # 内圈故障频率
                amplitude = 1.5
            elif class_id == 2:  # 外圈故障
                base_freq = 50
                fault_freq = 3.6 * base_freq  # 外圈故障频率
                amplitude = 1.3
            else:  # 滚动体故障
                base_freq = 50
                fault_freq = 4.7 * base_freq  # 滚动体故障频率
                amplitude = 1.2

            # 生成振动信号
            signal_data = np.zeros(sample_length)

            # 转频分量
            signal_data += amplitude * np.sin(2 * np.pi * base_freq * t)

            # 故障特征频率分量（正常状态没有）
            if class_id > 0:
                # 故障频率及其谐波
                for harmonic in range(1, 4):
                    signal_data += 0.3 * amplitude * np.sin(2 * np.pi * fault_freq * harmonic * t)
                    signal_data += 0.2 * amplitude * np.sin(2 * np.pi * (fault_freq * harmonic + base_freq) * t)

            # 添加随机噪声
            signal_data += noise_level * np.random.randn(sample_length)

            # 添加一些随机变化
            signal_data += 0.1 * np.random.randn(sample_length) * np.sin(2 * np.pi * np.random.uniform(10, 100) * t)

            X.append(signal_data)
            y.append(class_id)

    X = np.array(X)
    y = np.array(y)

    print(f"\n数据生成完成:")
    print(f"  总样本数: {len(X)}")
    print(f"  样本长度: {sample_length}")
    print(f"  采样频率: {sampling_rate} Hz")
    print(f"  类别分布: {np.bincount(y)}")

    return X, y


# =====================================================================
# 2. 数据预处理
# =====================================================================
def wavelet_denoise(data, wavelet='db4', threshold_mode='soft'):
    """
    小波阈值去噪

    参数:
        data: 输入信号
        wavelet: 小波基函数
        threshold_mode: 阈值模式 ('soft' 或 'hard')

    返回:
        去噪后的信号
    """
    if HAS_PYWT:
        # 小波分解
        coeffs = pywt.wavedec(data, wavelet, level=4)

        # 计算阈值
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        # 阈值处理
        coeffs_thresh = [pywt.threshold(c, threshold, threshold_mode) for c in coeffs]

        # 小波重构
        denoised = pywt.waverec(coeffs_thresh, wavelet)

        # 确保长度一致
        if len(denoised) != len(data):
            denoised = denoised[:len(data)]

        return denoised
    else:
        # 简化的去噪方法：使用移动平均滤波
        from scipy.signal import savgol_filter
        try:
            denoised = savgol_filter(data, window_length=min(51, len(data)//4*2+1), polyorder=3)
        except:
            # 如果savgol_filter失败，使用简单的移动平均
            window_size = min(5, len(data))
            denoised = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return denoised


def preprocess_data_for_cnn(X, sampling_rate=12000, nperseg=256, noverlap=128):
    """
    为CNN预处理数据：将时间序列转换为STFT频谱图

    参数:
        X: 时间序列数据 (num_samples, sample_length)
        sampling_rate: 采样频率
        nperseg: STFT窗口长度
        noverlap: STFT重叠长度

    返回:
        X_stft: STFT频谱图 (num_samples, freq_bins, time_bins, 1)
    """
    print("\n" + "-" * 80)
    print("数据预处理（CNN用）：STFT转换为频谱图")
    print("-" * 80)

    X_stft = []

    for i, signal_data in enumerate(X):
        # 执行STFT
        f, t, Zxx = stft(signal_data, sampling_rate, nperseg=nperseg, noverlap=noverlap)

        # 取幅值
        magnitude = np.abs(Zxx)

        # 归一化到[0, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        # 调整大小到64x64（通过插值）
        from scipy.ndimage import zoom
        target_size = (64, 64)
        current_size = magnitude.shape
        zoom_factors = (target_size[0] / current_size[0], target_size[1] / current_size[1])
        magnitude_resized = zoom(magnitude, zoom_factors, order=1)

        X_stft.append(magnitude_resized)

        if (i + 1) % 200 == 0:
            print(f"  处理进度: {i+1}/{len(X)}")

    X_stft = np.array(X_stft)
    # 添加通道维度
    X_stft = X_stft[..., np.newaxis]

    print(f"\nSTFT转换完成:")
    print(f"  频谱图形状: {X_stft.shape}")
    print(f"  频率分辨率: {len(f)} bins")
    print(f"  时间分辨率: {len(t)} bins")

    return X_stft


def preprocess_data_for_lstm(X, time_steps=4):
    """
    为LSTM预处理数据：将时间序列重塑为LSTM输入格式

    参数:
        X: 时间序列数据 (num_samples, sample_length)
        time_steps: 时间步数

    返回:
        X_lstm: LSTM输入格式 (num_samples, time_steps, features_per_step)
    """
    print("\n" + "-" * 80)
    print("数据预处理（LSTM用）：时间序列重塑")
    print("-" * 80)

    sample_length = X.shape[1]
    features_per_step = sample_length // time_steps

    X_lstm = []

    for signal_data in X:
        # 将信号分割为time_steps个时间步
        segments = []
        for i in range(time_steps):
            start_idx = i * features_per_step
            end_idx = (i + 1) * features_per_step
            segment = signal_data[start_idx:end_idx]
            # 每个时间步的特征可以是该段的均值或其他统计量
            # 这里使用均值作为特征
            features = np.array([np.mean(segment), np.std(segment),
                                np.max(segment), np.min(segment)])
            segments.append(features)

        X_lstm.append(segments)

    X_lstm = np.array(X_lstm)

    print(f"\n时间序列重塑完成:")
    print(f"  LSTM输入形状: {X_lstm.shape}")
    print(f"  时间步数: {time_steps}")
    print(f"  每步特征数: {X_lstm.shape[2]}")

    return X_lstm


def clean_data(X, y):
    """
    数据清洗：小波阈值去噪

    参数:
        X: 振动数据
        y: 标签

    返回:
        X_clean: 清洗后的数据
        y_clean: 清洗后的标签
    """
    print("\n" + "-" * 80)
    print("数据清洗：小波阈值去噪")
    print("-" * 80)

    X_clean = []

    for i, signal_data in enumerate(X):
        denoised = wavelet_denoise(signal_data)
        X_clean.append(denoised)

        if (i + 1) % 200 == 0:
            print(f"  处理进度: {i+1}/{len(X)}")

    X_clean = np.array(X_clean)

    print(f"\n数据清洗完成:")
    print(f"  原始数据形状: {X.shape}")
    print(f"  清洗后数据形状: {X_clean.shape}")

    return X_clean, y


# =====================================================================
# 3. 数据集类
# =====================================================================
class BearingDataset(Dataset):
    """轴承故障数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        if len(y.shape) == 1:
            self.y = torch.LongTensor(y)
        else:
            self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# 4. CNN模型定义
# =====================================================================
class CNNFaultDiagnosis(nn.Module):
    """基于CNN的故障诊断模型"""
    def __init__(self, num_classes=4):
        super(CNNFaultDiagnosis, self).__init__()

        # 卷积层1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 扁平化
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入: (batch, 1, 64, 64)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch, 32, 32, 32)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch, 64, 16, 16)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


# =====================================================================
# 5. LSTM模型定义
# =====================================================================
class LSTMFaultDiagnosis(nn.Module):
    """基于LSTM的故障诊断模型"""
    def __init__(self, input_size=4, hidden_size1=64, hidden_size2=32, num_classes=4):
        super(LSTMFaultDiagnosis, self).__init__()

        # LSTM层1
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, return_sequences=True)
        self.relu1 = nn.ReLU()

        # LSTM层2
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.relu2 = nn.ReLU()

        # 全连接层
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入: (batch, time_steps, features)
        x, _ = self.lstm1(x)
        x = self.relu1(x)

        x, _ = self.lstm2(x)
        x = self.relu2(x[:, -1, :])  # 取最后一个时间步的输出

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


# =====================================================================
# 6. 模型训练
# =====================================================================
def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001,
                device='cpu', model_name='Model'):
    """
    训练模型

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        model_name: 模型名称

    返回:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        train_accs: 训练准确率历史
        val_accs: 验证准确率历史
    """
    print("\n" + "-" * 80)
    print(f"{model_name} 模型训练")
    print("-" * 80)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_model_state = None

    print(f"\n训练参数:")
    print(f"  优化器: Adam")
    print(f"  学习率: {learning_rate}")
    print(f"  损失函数: CrossEntropyLoss")
    print(f"  训练轮数: {num_epochs}")
    print(f"  批次大小: {train_loader.batch_size}")
    print(f"\n开始训练...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # 每5个epoch打印一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n最佳验证准确率: {best_val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs


# =====================================================================
# 7. 模型评估
# =====================================================================
def evaluate_model(model, test_loader, device='cpu', class_names=None):
    """
    评估模型性能

    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        class_names: 类别名称

    返回:
        y_true: 真实标签
        y_pred: 预测标签
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1: F1值
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 打印分类报告
    if class_names is None:
        class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']

    print(f"\n整体性能指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1值: {f1:.4f}")

    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return y_true, y_pred, accuracy, precision, recall, f1


# =====================================================================
# 8. 可视化
# =====================================================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        save_path=None, model_name='Model'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    axes[0].plot(train_losses, label='训练集损失', linewidth=2)
    axes[0].plot(val_losses, label='验证集损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - 训练损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(train_accs, label='训练集准确率', linewidth=2)
    axes[1].plot(val_accs, label='验证集准确率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - 训练准确率曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, model_name='Model'):
    """绘制混淆矩阵"""
    if class_names is None:
        class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数'})
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title(f'{model_name} - 混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    plt.close()


def plot_model_comparison(cnn_metrics, lstm_metrics, save_path=None):
    """绘制模型对比图"""
    metrics = ['准确率', '精确率', '召回率', 'F1值']
    cnn_values = [cnn_metrics['accuracy'], cnn_metrics['precision'],
                  cnn_metrics['recall'], cnn_metrics['f1']]
    lstm_values = [lstm_metrics['accuracy'], lstm_metrics['precision'],
                   lstm_metrics['recall'], lstm_metrics['f1']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cnn_values, width, label='CNN', alpha=0.8)
    bars2 = ax.bar(x + width/2, lstm_values, width, label='LSTM', alpha=0.8)

    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('CNN vs LSTM 模型性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存: {save_path}")
    plt.close()


# =====================================================================
# 9. 主程序
# =====================================================================
def main():
    """主程序"""
    print("\n" + "=" * 80)
    print("工业故障诊断：基于CNN和LSTM的轴承故障诊断")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 1. 生成数据
    X, y = generate_bearing_fault_data(
        num_samples_per_class=500,
        sample_length=1024,
        sampling_rate=12000,
        noise_level=0.1
    )

    # 2. 数据清洗
    X_clean, y_clean = clean_data(X, y)

    # 3. 数据划分 (7:2:1)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_clean, y_clean, test_size=0.1, random_state=42, stratify=y_clean
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.222, random_state=42, stratify=y_temp  # 0.222 ≈ 2/9
    )

    print(f"\n数据划分完成:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # 4. 数据预处理
    # CNN用：STFT转换
    X_train_cnn = preprocess_data_for_cnn(X_train)
    X_val_cnn = preprocess_data_for_cnn(X_val)
    X_test_cnn = preprocess_data_for_cnn(X_test)

    # LSTM用：时间序列重塑
    X_train_lstm = preprocess_data_for_lstm(X_train, time_steps=4)
    X_val_lstm = preprocess_data_for_lstm(X_val, time_steps=4)
    X_test_lstm = preprocess_data_for_lstm(X_test, time_steps=4)

    # 5. 创建数据加载器
    # CNN数据加载器
    train_dataset_cnn = BearingDataset(X_train_cnn, y_train)
    val_dataset_cnn = BearingDataset(X_val_cnn, y_val)
    test_dataset_cnn = BearingDataset(X_test_cnn, y_test)

    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
    val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=32, shuffle=False)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=32, shuffle=False)

    # LSTM数据加载器
    train_dataset_lstm = BearingDataset(X_train_lstm, y_train)
    val_dataset_lstm = BearingDataset(X_val_lstm, y_val)
    test_dataset_lstm = BearingDataset(X_test_lstm, y_test)

    train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=32, shuffle=True)
    val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=32, shuffle=False)
    test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=32, shuffle=False)

    # 6. 构建和训练CNN模型
    print("\n" + "=" * 80)
    print("CNN模型构建与训练")
    print("=" * 80)

    cnn_model = CNNFaultDiagnosis(num_classes=4).to(device)

    print(f"\nCNN模型结构:")
    print(f"  输入: (batch, 1, 64, 64) 频谱图")
    print(f"  卷积层1: 32个3×3卷积核 + ReLU + 2×2最大池化")
    print(f"  卷积层2: 64个3×3卷积核 + ReLU + 2×2最大池化")
    print(f"  全连接层1: 128神经元 + ReLU + Dropout(0.5)")
    print(f"  输出层: 4神经元 + Softmax")

    total_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"  总参数数量: {total_params:,}")

    train_losses_cnn, val_losses_cnn, train_accs_cnn, val_accs_cnn = train_model(
        cnn_model, train_loader_cnn, val_loader_cnn,
        num_epochs=30, learning_rate=0.001, device=device, model_name='CNN'
    )

    # 7. 构建和训练LSTM模型
    print("\n" + "=" * 80)
    print("LSTM模型构建与训练")
    print("=" * 80)

    lstm_model = LSTMFaultDiagnosis(input_size=4, hidden_size1=64,
                                    hidden_size2=32, num_classes=4).to(device)

    print(f"\nLSTM模型结构:")
    print(f"  输入: (batch, 4, 4) 时间序列")
    print(f"  LSTM层1: 64神经元 + ReLU (return_sequences=True)")
    print(f"  LSTM层2: 32神经元 + ReLU")
    print(f"  全连接层: 64神经元 + ReLU + Dropout(0.5)")
    print(f"  输出层: 4神经元 + Softmax")

    total_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"  总参数数量: {total_params:,}")

    train_losses_lstm, val_losses_lstm, train_accs_lstm, val_accs_lstm = train_model(
        lstm_model, train_loader_lstm, val_loader_lstm,
        num_epochs=30, learning_rate=0.001, device=device, model_name='LSTM'
    )

    # 8. 模型评估
    print("\n" + "=" * 80)
    print("模型评估")
    print("=" * 80)

    class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']

    # CNN评估
    print("\n" + "-" * 80)
    print("CNN模型测试集评估")
    print("-" * 80)
    y_true_cnn, y_pred_cnn, acc_cnn, prec_cnn, rec_cnn, f1_cnn = evaluate_model(
        cnn_model, test_loader_cnn, device=device, class_names=class_names
    )

    # LSTM评估
    print("\n" + "-" * 80)
    print("LSTM模型测试集评估")
    print("-" * 80)
    y_true_lstm, y_pred_lstm, acc_lstm, prec_lstm, rec_lstm, f1_lstm = evaluate_model(
        lstm_model, test_loader_lstm, device=device, class_names=class_names
    )

    # 9. 可视化
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)

    # CNN训练曲线
    plot_training_curves(
        train_losses_cnn, val_losses_cnn, train_accs_cnn, val_accs_cnn,
        save_path=os.path.join(results_dir, 'cnn_training_curves.png'),
        model_name='CNN'
    )

    # LSTM训练曲线
    plot_training_curves(
        train_losses_lstm, val_losses_lstm, train_accs_lstm, val_accs_lstm,
        save_path=os.path.join(results_dir, 'lstm_training_curves.png'),
        model_name='LSTM'
    )

    # CNN混淆矩阵
    plot_confusion_matrix(
        y_true_cnn, y_pred_cnn, class_names=class_names,
        save_path=os.path.join(results_dir, 'cnn_confusion_matrix.png'),
        model_name='CNN'
    )

    # LSTM混淆矩阵
    plot_confusion_matrix(
        y_true_lstm, y_pred_lstm, class_names=class_names,
        save_path=os.path.join(results_dir, 'lstm_confusion_matrix.png'),
        model_name='LSTM'
    )

    # 模型对比
    cnn_metrics = {
        'accuracy': acc_cnn,
        'precision': prec_cnn,
        'recall': rec_cnn,
        'f1': f1_cnn
    }
    lstm_metrics = {
        'accuracy': acc_lstm,
        'precision': prec_lstm,
        'recall': rec_lstm,
        'f1': f1_lstm
    }

    plot_model_comparison(
        cnn_metrics, lstm_metrics,
        save_path=os.path.join(results_dir, 'model_comparison.png')
    )

    # 10. 实验总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    print(f"\n数据采集过程:")
    print(f"  - 数据来源: 模拟CWRU轴承故障数据")
    print(f"  - 总样本数: {len(X)}")
    print(f"  - 样本长度: 1024 数据点")
    print(f"  - 采样频率: 12000 Hz")
    print(f"  - 故障类别: 正常、内圈故障、外圈故障、滚动体故障")

    print(f"\n数据预处理:")
    print(f"  - 数据清洗: 小波阈值去噪 (db4小波基)")
    print(f"  - CNN预处理: STFT转换为64×64频谱图")
    print(f"  - LSTM预处理: 时间序列重塑为4×4格式")
    print(f"  - 数据划分: 训练集70%、验证集20%、测试集10%")

    print(f"\nCNN模型结构:")
    print(f"  - 输入层: 64×64×1 频谱图")
    print(f"  - 卷积层1: 32个3×3卷积核 + ReLU + 2×2最大池化")
    print(f"  - 卷积层2: 64个3×3卷积核 + ReLU + 2×2最大池化")
    print(f"  - 全连接层1: 128神经元 + ReLU + Dropout(0.5)")
    print(f"  - 输出层: 4神经元 + Softmax")

    print(f"\nLSTM模型结构:")
    print(f"  - 输入层: 4×4 时间序列")
    print(f"  - LSTM层1: 64神经元 + ReLU (return_sequences=True)")
    print(f"  - LSTM层2: 32神经元 + ReLU")
    print(f"  - 全连接层: 64神经元 + ReLU + Dropout(0.5)")
    print(f"  - 输出层: 4神经元 + Softmax")

    print(f"\n训练参数:")
    print(f"  - 优化器: Adam")
    print(f"  - 学习率: 0.001")
    print(f"  - 损失函数: CrossEntropyLoss")
    print(f"  - 训练轮数: 30")
    print(f"  - 批次大小: 32")

    print(f"\nCNN模型性能 (测试集):")
    print(f"  - 准确率: {acc_cnn:.4f}")
    print(f"  - 精确率: {prec_cnn:.4f}")
    print(f"  - 召回率: {rec_cnn:.4f}")
    print(f"  - F1值: {f1_cnn:.4f}")

    print(f"\nLSTM模型性能 (测试集):")
    print(f"  - 准确率: {acc_lstm:.4f}")
    print(f"  - 精确率: {prec_lstm:.4f}")
    print(f"  - 召回率: {rec_lstm:.4f}")
    print(f"  - F1值: {f1_lstm:.4f}")

    print(f"\n模型分析:")
    print(f"  - CNN擅长处理频谱图的空间特征，能够从二维频谱中提取故障特征")
    print(f"  - LSTM擅长捕捉时间序列的时序特征，能够学习故障发展过程中的时序依赖关系")
    print(f"  - 两种模型各有优势，可根据实际应用场景选择")

    print(f"\n深度学习在工业故障诊断中的关键技术要点:")
    print(f"  1. 数据预处理: 小波去噪、STFT转换、时间序列重塑")
    print(f"  2. 模型结构设计: CNN用于空间特征提取，LSTM用于时序特征提取")
    print(f"  3. 泛化性提升: Dropout正则化、数据增强、早停法")
    print(f"  4. 评估指标: 准确率、精确率、召回率、F1值，重点关注故障类别召回率")

    print(f"\n实际工业应用中的改进方向:")
    print(f"  1. 小样本故障诊断: 使用迁移学习、数据增强、少样本学习")
    print(f"  2. 实时诊断优化: 模型压缩、边缘计算、在线学习")
    print(f"  3. 多传感器融合: 结合振动、温度、声音等多种传感器数据")
    print(f"  4. 可解释性提升: 使用注意力机制、特征可视化等方法")

    print(f"\n所有结果已保存到: {results_dir}")
    print("\n程序执行完成！")


if __name__ == '__main__':
    main()

