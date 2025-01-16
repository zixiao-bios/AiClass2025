import numpy as np

import utils

def data_1(X, noise=1):
    """y = -2x - 1704
    """
    data = -2 * X - 1704
    return data + np.random.randn(*data.shape) * noise

def data_2(X, noise=1):
    """y = 0.5x^2 + 2x + 1
    """
    data = 0.5 * X**2 + 2 * X + 1
    return data + np.random.randn(*data.shape) * noise

def data_3(X, noise=1):
    """y = sin(2x) + log(|x| + 1) - 0.05x^3
    """
    data = np.sin(2 * X) + np.log(np.abs(X) + 1) - 0.05 * X**3
    return data + np.random.randn(*data.shape) * noise

def main():
    # 设置随机种子以确保结果可重复
    np.random.seed(0)

    # 生成从 -5 到 5 的 1000 个均匀分布的点
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    # X.shape = (1000, 1)

    # 生成 y
    y1 = data_1(X)
    y2 = data_2(X, noise=0.5)
    y3 = data_3(X, noise=0.2)
    # y.shape = (1000, 1)
    
    utils.save_to_csv(X, y1, 'data_1.csv')
    utils.save_to_csv(X, y2, 'data_2.csv')
    utils.save_to_csv(X, y3, 'data_3.csv')
    
    # 生成 2 个特征维度的数据
    X = np.random.uniform(-15, 15, (4000, 2))
    y4 = data_2(X[:, 0].reshape(-1, 1), noise=0.5) + data_3(X[:, 1].reshape(-1, 1), noise=0.2)
    # X.shape = (1000, 2), y.shape = (1000, 1)
    
    utils.save_to_csv(X, y4, 'data_4.csv')


if __name__ == '__main__':
    main()
