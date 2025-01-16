import torch
import numpy as np

import utils


if __name__ == '__main__':
    # 指定分布
    W_true = np.array([0.8, -0.4])
    b_true = -0.2
    
    # 生成数据
    data_num = 200
    X, Y = utils.make_linear_data(W_true, b_true, data_num)
    print(X.shape, Y.shape)
    utils.draw_3d_scatter(X, Y)
    
    # 初始化参数
    w = torch.normal(0, 1, size=(2,1), requires_grad=True, dtype=torch.float32)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
    
    # 指定超参数
    lr = 1e-5
    epochs = 50
    
    for epoch in range(epochs):
        # 本轮的损失
        epoch_loss = 0
        
        for i in range(data_num):
            # 取一个样本
            x = torch.tensor(X[i], dtype=torch.float32)
            y = torch.tensor(Y[i], dtype=torch.float32)

            # 计算预测、损失、梯度
            y_hat = x @ w + b
            loss = (y_hat - y) ** 2
            loss.backward()
            
            # 计算完梯度后，不再累积梯度
            with torch.no_grad():
                # 梯度下降，更新参数
                w -= lr * w.grad
                b -= lr * b.grad
                
                # 清除梯度，防止影响下一次计算
                w.grad.zero_()
                b.grad.zero_()

            # 累积损失
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch}, loss: {epoch_loss / data_num}')
    
    # 训练完成，打印参数
    print(f'W_true: {W_true}, W: {w.detach().numpy().flatten()}')
    print(f'b_true: {b_true}, b: {b.item()}')
    
    # 绘制拟合结果
    y_hat = X @ w.detach().numpy() + b.item()
    utils.draw_3d_scatter(X, Y, y_hat)
