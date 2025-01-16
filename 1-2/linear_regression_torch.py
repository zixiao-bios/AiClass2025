import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import utils


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        
        # y = wx + b，输出输出都是 1 维
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # 指定分布
    W_true = np.array([0.8, -0.4])
    b_true = -0.2
    
    # 生成数据
    data_num = 200
    X, Y = utils.make_linear_data(W_true, b_true, data_num)
    print(X.shape, Y.shape)
    utils.draw_3d_scatter(X, Y)
    
    # 实例化模型
    model = LinearRegressionModel()
    
    # 均方误差作为损失函数
    criterion = nn.MSELoss()

    # 随机梯度下降 (SGD) 作为优化器
    optimizer = optim.SGD(model.parameters(), lr=1e-5)  # 学习率 1e-5

    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0

        for i in range(data_num):
            x = torch.tensor(X[i], dtype=torch.float32)
            y = torch.tensor(Y[i], dtype=torch.float32)
            
            # 使用模型预测输出
            y_hat = model(x)
            
            # 计算损失
            loss = criterion(y_hat, y)
            
            # 梯度清零，防止累积
            optimizer.zero_grad()
            
            # 计算梯度
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch}, loss: {epoch_loss / data_num}')

    # 训练完成，打印参数
    print(f'W_true: {W_true}, W: {model.linear.weight.detach().numpy().flatten()}')
    print(f'b_true: {b_true}, b: {model.linear.bias.item()}')
    
    # 绘制拟合结果
    y_hat = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    utils.draw_3d_scatter(X, Y, y_hat)
