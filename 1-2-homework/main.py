import torch

import utils


# 在这里定义你的模型



if __name__ == '__main__':
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data('data_1.csv')
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)
    
    # 根据特征维度，绘制 2D 或 3D 散点图
    utils.draw_2d_scatter(X, Y)
    # utils.draw_3d_scatter(X, Y)
    
    # 在这里开始你的表演！
    
    
    
    
    
    
    
    # 查看预测效果
    predicted = model(X)
    utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
    # utils.draw_3d_scatter(X, Y, predicted.detach().numpy())
