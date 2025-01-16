import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def make_linear_data(W, b, num, noise=1):
    """根据指定的参数生成线性数据: y = XW + b + noise
    W: 真实权重，形状 (dim,)
    b: 真实偏置，标量
    num: 生成数据的数量
    noise: 噪声标准差
    """
    W = W.reshape(-1, 1)
    # W.shape = (dim, 1)
    
    # 生成 -10 到 10 范围内的随机数
    X = np.random.uniform(-10, 10, (num, W.shape[0]))
    # X.shape = (num, dim)
    
    y = np.dot(X, W) + b
    y = y + np.random.randn(*y.shape) * noise
    # y.shape = (num, 1)
    
    return X, y

def draw_2d_scatter(X, y, y_hat=None):
    """绘制二维散点图，支持同时绘制原始数据和预测数据

    Args:
        X (_type_): 特征数据，形状为 (num, 1)
        y (_type_): 标签数据，形状为 (num, 1)
        y_hat (_type_): 预测数据，形状为 (num, 1)，不指定则不绘制
    """
    # 如果超过 1000 个点，随机选择 1000 个点绘制
    if len(X) > 1000:
        indices = np.random.choice(len(X), size=1000, replace=False)
        X = X[indices]
        y = y[indices]
        if y_hat is not None:
            y_hat = y_hat[indices]
    
    plt.scatter(X, y, s=5, label='Original Data')
    if y_hat is not None:
        plt.plot(X, y_hat, color='red', label='Fitting Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def draw_3d_scatter(X, y, y_hat=None):
    """绘制三维散点图

    Args:
        X (_type_): 特征数据，形状为 (num, 2)
        y (_type_): 标签数据，形状为 (num, 1)
        y_hat (_type_): 预测数据，形状为 (num, 1)，不指定则不绘制
    """
    # 如果超过 1000 个点，随机选择 1000 个点绘制
    if len(X) > 1000:
        indices = np.random.choice(len(X), size=1000, replace=False)
        X = X[indices]
        y = y[indices]
        if y_hat is not None:
            y_hat = y_hat[indices]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:,0], X[:,1], y[:, 0], c=y[:, 0], cmap='viridis', s=50, alpha=0.7)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label('Y')

    if y_hat is not None:
        # 预测值使用小红点表示
        ax.scatter(X[:,0], X[:,1], y_hat[:, 0], c='red', s=10, alpha=1)
    
    # 设置轴标签
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.show()

def read_csv_data(filename):
    """从 CSV 文件读取数据，支持多维度的特征

    参数:
    filename (str): CSV 文件的路径。

    返回:
    X (numpy.ndarray): 特征矩阵，形状为 (样本数, 特征维度)。
    Y (numpy.ndarray): 目标变量，形状为 (样本数, 1)。
    """
    df = pd.read_csv(filename)
    
    # 假设 'y' 是目标列，其他所有列都是特征
    if 'y' not in df.columns:
        raise ValueError("CSV 文件中必须包含 'y' 列作为目标变量。")
    
    # 提取特征列（所有列除 'y' 外）
    feature_columns = df.columns.drop('y')
    X = df[feature_columns].values  # 形状: (样本数, 特征维度)
    
    # 提取目标变量
    Y = df['y'].values.reshape(-1, 1)  # 形状: (样本数, 1)
    
    return X, Y

def save_to_csv(X, Y, filename):
    """保存数据到 CSV 文件。

    参数:
    X (numpy.ndarray): 特征矩阵，形状为 (样本数, 特征维度)。
    Y (numpy.ndarray): 目标变量，形状为 (样本数, 1)。
    filename (str): 保存的 CSV 文件路径。
    """
    # 检查 X 和 Y 的样本数量是否一致
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X 和 Y 的样本数量必须一致。")
    
    # 拼接 X 和 Y
    data = np.hstack((X, Y))  # 形状: (样本数, 特征维度 + 1)
    
    # 生成特征列名
    num_features = X.shape[1]
    feature_columns = [f'X{i+1}' for i in range(num_features)]
    
    # 定义 DataFrame 的列名，包括特征和目标变量
    columns = feature_columns + ['y']
    
    # 创建 DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 保存到 CSV 文件
    df.to_csv(filename, index=False)
    print(f"数据已保存到 '{filename}' 文件中")
