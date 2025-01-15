from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # 创建两个全连接层
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # 第一层全连接后使用ReLU激活函数
        x = self.fc1(x)
        x = F.relu(x)

        # 第二层全连接后直接输出
        return self.fc2(x)


if __name__ == '__main__':
    # 读入数据、转为 tensor、绘制
    X, Y = utils.read_csv_data('data_2.csv')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)
    # utils.draw_2d_scatter(X, Y)
    
    # 模型、损失函数、优化器
    model = SimpleMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # 创建数据集和加载器
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ===================== 4. 训练模型 =====================
    epochs = 50
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_x, batch_y in data_loader:
            # 使用模型预测输出
            predictions = model(batch_x)
            
            # 计算损失
            loss = criterion(predictions, batch_y)
            
            # 梯度清零，防止累积
            optimizer.zero_grad()
            
            # 计算梯度
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()

        # 打印训练过程中的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')


    # ===================== 5. 查看结果 =====================
    # 使用训练好的模型预测
    predicted = model(X)
    utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
