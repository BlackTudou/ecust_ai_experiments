"""
生物反应器产率BP神经网络建模
根据温度、叶轮转速、持续时间、反应器是否配备挡板，对生物反应器的产率百分比进行建模
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
data_path = os.path.join(script_dir, '..', 'data', 'exp14_data_bioreactor-yields.csv')


# =====================================================================
# 1. 数据加载与预处理
# =====================================================================
print("=" * 80)
print("1. 数据加载与预处理")
print("=" * 80)

def load_data(file_path):
    """加载数据"""
    df = pd.read_csv(file_path)
    print(f"\n数据形状: {df.shape}")
    print(f"\n数据前5行:")
    print(df.head())
    print(f"\n数据基本信息:")
    print(df.info())
    print(f"\n数据统计描述:")
    print(df.describe())
    return df


def clean_data(df):
    """数据清洗：使用3σ准则删除异常值"""
    print("\n" + "-" * 80)
    print("数据清洗：使用3σ准则删除异常值")
    print("-" * 80)

    original_size = len(df)

    # 对数值型列进行异常值检测
    numeric_cols = ['temperature', 'duration', 'speed', 'yield']

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        # 找出异常值
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"\n{col} 列异常值数量: {len(outliers)}")
            print(f"  均值: {mean:.2f}, 标准差: {std:.2f}")
            print(f"  范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  异常值: {outliers[col].tolist()}")

        # 删除异常值
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    removed = original_size - len(df)
    print(f"\n删除异常值数量: {removed}")
    print(f"剩余数据量: {len(df)}")

    return df


def preprocess_data(df):
    """数据预处理：编码分类变量、归一化、划分数据集"""
    print("\n" + "-" * 80)
    print("数据预处理：编码分类变量、归一化、划分数据集")
    print("-" * 80)

    # 复制数据
    df_processed = df.copy()

    # 编码分类变量：baffles (Yes/No -> 1/0)
    df_processed['baffles'] = df_processed['baffles'].map({'Yes': 1, 'No': 0})
    print(f"\n挡板编码: Yes -> 1, No -> 0")

    # 分离输入和输出
    X = df_processed[['temperature', 'duration', 'speed', 'baffles']].values
    y = df_processed['yield'].values.reshape(-1, 1)

    print(f"\n输入特征: temperature, duration, speed, baffles")
    print(f"输出变量: yield")
    print(f"输入数据形状: {X.shape}")
    print(f"输出数据形状: {y.shape}")

    # 数据归一化到[0,1]区间
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    print(f"\n数据归一化完成 (范围: [0, 1])")
    print(f"输入数据范围: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"输出数据范围: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

    # 数据划分：训练集80%，测试集20%
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"\n数据划分完成:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# =====================================================================
# 2. 定义数据集类
# =====================================================================
class BioreactorDataset(Dataset):
    """生物反应器数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# 3. BP神经网络模型定义
# =====================================================================
class BPNeuralNetwork(nn.Module):
    """BP神经网络模型"""
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        """
        初始化BP神经网络
        Args:
            input_dim: 输入层维度
            hidden_dims: 隐藏层维度列表，如[64, 32]表示两层隐藏层
            output_dim: 输出层维度
        """
        super(BPNeuralNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层（Linear激活函数，适用于连续输出）
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =====================================================================
# 4. 模型训练
# =====================================================================
def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=0.001,
                patience=5, device='cpu'):
    """训练模型，使用早停法防止过拟合"""
    print("\n" + "-" * 80)
    print("模型训练")
    print("-" * 80)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"\n训练参数:")
    print(f"  优化器: Adam")
    print(f"  学习率: {learning_rate}")
    print(f"  损失函数: MSE")
    print(f"  最大训练轮数: {num_epochs}")
    print(f"  早停耐心值: {patience}")
    print(f"\n开始训练...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发！验证集损失连续{patience}个epoch未下降")
            print(f"最佳验证损失: {best_val_loss:.6f} (Epoch {epoch+1-patience})")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


# =====================================================================
# 5. 模型评估指标
# =====================================================================
def calculate_rmse(y_true, y_pred):
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差 (MAPE)"""
    # 避免除零
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X, y, scaler_y, device='cpu'):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred_scaled = model(X_tensor).cpu().numpy()

    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y.reshape(-1, 1))

    # 计算指标
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    return y_true, y_pred, rmse, mape


# =====================================================================
# 6. 可视化
# =====================================================================
def plot_training_curves(train_losses, val_losses, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练集损失', linewidth=2)
    plt.plot(val_losses, label='验证集损失', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('模型训练曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存: {save_path}")
    plt.close()


def plot_prediction_comparison(y_true, y_pred, title='预测对比', save_path=None):
    """绘制真实输出-预测输出对比曲线"""
    plt.figure(figsize=(12, 6))

    indices = np.arange(len(y_true))
    plt.plot(indices, y_true, 'o-', label='真实值', linewidth=2, markersize=6, alpha=0.7)
    plt.plot(indices, y_pred, 's-', label='预测值', linewidth=2, markersize=6, alpha=0.7)

    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('产率百分比', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测对比图已保存: {save_path}")
    plt.close()


def plot_error_curve(y_true, y_pred, title='预测误差', save_path=None):
    """绘制预测误差曲线"""
    errors = y_true.flatten() - y_pred.flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(errors, 'o-', linewidth=2, markersize=6, alpha=0.7, color='red')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('误差 (真实值 - 预测值)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"误差曲线已保存: {save_path}")
    plt.close()


def plot_scatter_comparison(y_true, y_pred, title='预测散点图', save_path=None):
    """绘制真实值vs预测值散点图"""
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true, y_pred, alpha=0.6, s=50)

    # 绘制理想线 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线 (y=x)')

    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点对比图已保存: {save_path}")
    plt.close()


# =====================================================================
# 7. 主程序
# =====================================================================
def main():
    """主程序"""
    print("\n" + "=" * 80)
    print("生物反应器产率BP神经网络建模")
    print("=" * 80)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 1. 数据加载
    df = load_data(data_path)

    # 2. 数据清洗
    df_clean = clean_data(df)

    # 3. 数据预处理
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(df_clean)

    # 4. 创建数据集和数据加载器
    # 从训练集中划分出验证集 (20%的训练集作为验证集)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    train_dataset = BioreactorDataset(X_train_split, y_train_split)
    val_dataset = BioreactorDataset(X_val_split, y_val_split)
    test_dataset = BioreactorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"\n数据加载器创建完成:")
    print(f"  训练集批次: {len(train_loader)}")
    print(f"  验证集批次: {len(val_loader)}")
    print(f"  测试集批次: {len(test_loader)}")

    # 5. 构建BP神经网络模型
    print("\n" + "-" * 80)
    print("BP神经网络模型构建")
    print("-" * 80)

    input_dim = X_train.shape[1]  # 4个输入特征
    hidden_dims = [64, 32]  # 两层隐藏层，神经元数量分别为64和32
    output_dim = 1  # 1个输出

    model = BPNeuralNetwork(input_dim, hidden_dims, output_dim).to(device)

    print(f"\n网络结构:")
    print(f"  输入层: {input_dim} 个神经元")
    for i, dim in enumerate(hidden_dims):
        print(f"  隐藏层{i+1}: {dim} 个神经元 (ReLU激活)")
    print(f"  输出层: {output_dim} 个神经元 (Linear激活)")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数数量: {total_params}")
    print(f"  可训练参数数量: {trainable_params}")

    # 6. 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=200,
        learning_rate=0.001,
        patience=5,
        device=device
    )

    # 7. 模型评估
    print("\n" + "=" * 80)
    print("模型评估")
    print("=" * 80)

    # 训练集评估
    y_train_true, y_train_pred, train_rmse, train_mape = evaluate_model(
        model, X_train_split, y_train_split, scaler_y, device
    )

    # 验证集评估
    y_val_true, y_val_pred, val_rmse, val_mape = evaluate_model(
        model, X_val_split, y_val_split, scaler_y, device
    )

    # 测试集评估
    y_test_true, y_test_pred, test_rmse, test_mape = evaluate_model(
        model, X_test, y_test, scaler_y, device
    )

    print(f"\n训练集性能:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAPE: {train_mape:.4f}%")

    print(f"\n验证集性能:")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAPE: {val_mape:.4f}%")

    print(f"\n测试集性能:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAPE: {test_mape:.4f}%")

    # 8. 可视化
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)

    # 创建输出目录
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(output_dir, 'training_curves.png')
    )

    # 绘制测试集预测对比
    plot_prediction_comparison(
        y_test_true, y_test_pred,
        title='测试集：真实输出 vs 预测输出',
        save_path=os.path.join(output_dir, 'test_prediction_comparison.png')
    )

    # 绘制测试集误差曲线
    plot_error_curve(
        y_test_true, y_test_pred,
        title='测试集：预测误差曲线',
        save_path=os.path.join(output_dir, 'test_error_curve.png')
    )

    # 绘制测试集散点图
    plot_scatter_comparison(
        y_test_true, y_test_pred,
        title='测试集：真实值 vs 预测值',
        save_path=os.path.join(output_dir, 'test_scatter_comparison.png')
    )

    # 9. 实验总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    print(f"\n数据采集过程:")
    print(f"  - 数据来源: {data_path}")
    print(f"  - 原始数据量: {len(df)} 样本")
    print(f"  - 清洗后数据量: {len(df_clean)} 样本")
    print(f"  - 输入特征: 温度、持续时间、叶轮转速、挡板配置")
    print(f"  - 输出变量: 产率百分比")

    print(f"\n模型结构:")
    print(f"  - 输入层: {input_dim} 个神经元")
    for i, dim in enumerate(hidden_dims):
        print(f"  - 隐藏层{i+1}: {dim} 个神经元 (ReLU激活)")
    print(f"  - 输出层: {output_dim} 个神经元 (Linear激活)")
    print(f"  - 总参数数量: {total_params}")

    print(f"\n训练参数:")
    print(f"  - 优化器: Adam")
    print(f"  - 学习率: 0.001")
    print(f"  - 损失函数: MSE")
    print(f"  - 批次大小: 32")
    print(f"  - 早停耐心值: 5")

    print(f"\n精度指标 (测试集):")
    print(f"  - RMSE: {test_rmse:.4f}")
    print(f"  - MAPE: {test_mape:.4f}%")

    print(f"\n模型分析:")
    if test_rmse < 5 and test_mape < 10:
        print(f"  - 模型精度优秀，RMSE和MAPE均较小，误差曲线平稳")
    elif test_rmse < 10 and test_mape < 20:
        print(f"  - 模型精度良好，RMSE和MAPE在可接受范围内")
    else:
        print(f"  - 模型精度有待提升，建议调整网络结构或训练参数")

    print(f"\nBP神经网络在工业过程建模中的优势:")
    print(f"  - 能够学习非线性映射关系")
    print(f"  - 对多变量耦合关系有良好的建模能力")
    print(f"  - 训练速度快，易于实现")

    print(f"\n局限性:")
    print(f"  - 需要大量历史数据")
    print(f"  - 模型可解释性较差")
    print(f"  - 对数据质量要求较高")

    print(f"\n建模关键步骤:")
    print(f"  1. 数据清洗：使用3σ准则删除异常值")
    print(f"  2. 数据归一化：将数据归一化到[0,1]区间")
    print(f"  3. 网络结构设计：通过试错法确定隐藏层神经元数量")
    print(f"  4. 早停法：防止过拟合，提高模型泛化能力")
    print(f"  5. 模型评估：使用RMSE和MAPE评估模型精度")

    print(f"\n所有结果已保存到: {output_dir}")
    print("\n程序执行完成！")


if __name__ == '__main__':
    main()

