import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个简单的多层感知机模型
# pytorch 中所有自定义的神经网络类，都要继承 nn.Module
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_num: int, hidden_dim: int, output_dim: int):
        # 调用基类的构造函数
        super(SimpleMLP, self).__init__()
        
        # 验证 hidden_num
        assert hidden_num > 0, "hidden_num must be a positive integer"

        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 创建隐藏层
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过所有隐藏层
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # 输出层使用Sigmoid激活函数
        x = F.sigmoid(self.output_layer(x))
        return x
