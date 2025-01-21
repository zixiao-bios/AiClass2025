import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path


def preprocess(train_path, test_path):
    # 读入 CSV 数据
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 处理训练集数据
    # 划分出标签（第一列）与图片（其余列）
    train_label = df_train.values[:, 0]
    train_data = df_train.values[:, 1:]
    
    # 将每个图片 reshape 为一通道二维数组
    train_data = train_data.reshape(-1, 1, 28, 28)
    
    # 处理测试集数据
    test_data = df_test.values
    test_data = test_data.reshape(-1, 1, 28, 28)
    
    # 归一化图片像素值
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    
    return train_data, train_label, test_data
