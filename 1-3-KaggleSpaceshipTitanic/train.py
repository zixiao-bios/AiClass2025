import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

from mlp import SimpleMLP

# tensorboard 记录的文件夹名称
run_name = '01'

# 超参数
num_epochs = 50
lr = 0.01
batch_size = 64

hidden_dim = 16
hidden_num = 2


def main():
    # 读入处理后的数据
    print('\n======== 读入处理后的数据')
    df_train = pd.read_csv('dataset/train_processed.csv')
    df_test = pd.read_csv('dataset/test_processed.csv')
    
    df_train_features = df_train.drop(['Transported', 'PassengerId'], axis=1)
    df_train_target = df_train['Transported']
    df_test_features = df_test.drop(['Transported', 'PassengerId'], axis=1)
    print(df_train_features)
    print(df_train_target)
    
    # 将数据转换为 PyTorch 的 Tensor
    print('\n======== 将数据转换为 PyTorch 的 Tensor')
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    
    X_train = torch.tensor(df_train_features.values, dtype=torch.float32)
    y_train = torch.tensor(df_train_target.values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(df_test_features.values, dtype=torch.float32)
    print(X_train.shape)
    print(y_train.shape)
    
    # 构建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义模型、损失函数和优化器
    model = SimpleMLP(input_dim=X_train.shape[1], hidden_num=hidden_num, hidden_dim=hidden_dim, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.BCELoss()
    
    # 训练模型
    print('\n======== 训练模型')
    writer = SummaryWriter(f'runs/{run_name}')
    for epoch in range(num_epochs):
        model.train()
        
        # 每个 epoch 的损失
        epoch_loss = 0
        
        # 预测正确的个数
        correct_num = 0
        step = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            
            # 计算预测正确的个数，阈值为0.5
            correct_num += torch.sum((y_pred > 0.5) == y_batch).item()
            
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
    y_pred = model(X_test)
    
    # 计算预测结果，阈值为0.5，转换为 bool 类型
    y_pred = (y_pred > 0.5).reshape(-1).cpu().numpy().astype(bool)
    
    # 保存到 CSV 文件
    sub = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Transported': y_pred})
    print(sub)
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
