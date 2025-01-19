import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd

from mlp import SimpleMLP
from cnn import SimpleCNN
from preprocess import preprocess
import utils

# tensorboard 记录的文件夹名称
run_name = '01'

# 超参数
num_epochs = 20
lr = 0.01
batch_size = 500

input_dim = 28 * 28
hidden_dim = 16
hidden_num = 2


def main():
    # 选择设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # 读入处理后的数据
    train_data, train_label, test_data = preprocess('dataset/train.csv', 'dataset/test.csv')
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    
    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.int8).reshape(-1, 1)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    # 从训练集随机挑选9张图片绘制
    utils.draw_imgs(X_train, y_train)
    
    # 把标签变为 ont-hot 编码
    y_train = torch.tensor(np.eye(10)[y_train.reshape(-1)], dtype=torch.float32)
    
    # 构建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义模型、损失函数和优化器
    model = SimpleMLP(
        input_dim=input_dim, 
        hidden_num=hidden_num, 
        hidden_dim=hidden_dim, 
        output_dim=10
    ).to(device)
    # model = SimpleCNN().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 训练模型
    writer = SummaryWriter(f'runs/{run_name}')
    for epoch in range(num_epochs):
        model.train()
        
        # 每个 epoch 的损失
        epoch_loss = 0
        
        # 预测正确的个数
        correct_num = 0
        step = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            # y_pred.shape = (batch_size, 10)
            
            # 计算预测正确的个数
            correct_num += (torch.argmax(y_pred, dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
            
            l = loss(y_pred, y_batch)
            epoch_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            step += 1
        
        print(f'Epoch: {epoch}, Epoch Loss: {epoch_loss}, Accuracy: {correct_num / n_train}')
        writer.add_scalar(f'accuracy', correct_num / n_train, epoch)
        writer.add_scalar(f'loss', epoch_loss, epoch)
    
    # 预测测试集
    print('\n======== 预测测试集')
    # 设置为评估模式
    model.eval()
    
    # 不计算梯度
    with torch.no_grad():
        # 预测测试集
        y_pred = model(X_test.to(device))
        
        # 计算预测结果
        y_pred = torch.argmax(y_pred, dim=1)
    
    # 保存到 CSV 文件，第一列为图片id，第二列为预测类别
    sub = pd.DataFrame({'ImageId': np.arange(1, n_test + 1), 'Label': y_pred.cpu().numpy()})
    print(sub)
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
