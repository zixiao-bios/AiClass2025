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

hidden_dim = 4
hidden_num = 1


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
    
    # 开始你的表演
    

if __name__ == '__main__':
    main()
