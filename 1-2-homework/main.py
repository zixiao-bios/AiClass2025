import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils


# 在这里定义你的模型，不要直接复制代码！
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # 创建两个全连接层
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # 第一层全连接后使用ReLU激活函数
        x = self.fc1(x)
        x = F.relu(x)
        
        # 第二层全连接后直接输出
        return self.fc2(x)


if __name__ == '__main__':
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data('data_4.csv')
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)
    
    # 根据特征维度，绘制 2D 或 3D 散点图
    # utils.draw_2d_scatter(X, Y)
    utils.draw_3d_scatter(X, Y)
    
    # 在这里开始你的表演，不要直接复制代码！
    # 模型、损失函数、优化器
    model = SimpleMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 创建数据集和加载器
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_x, batch_y in data_loader:
            # 预测输出、计算损失
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # 计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()

        # 打印本轮的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')
    
    # 查看预测效果
    predicted = model(X)
    # utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
    utils.draw_3d_scatter(X, Y, predicted.detach().numpy())
