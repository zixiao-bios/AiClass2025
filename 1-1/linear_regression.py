import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ===================== 1. 构建人工数据集 =====================
# 假设真实的数据关系是 y = 3x + 2，并加入一些随机噪声

# 生成从0到10的100个点
X = torch.linspace(-10, 10, 100)
# X.shape = (100)

X = X.unsqueeze(1)
# X.shape = (100, 1)

# 生成 y，加入随机噪声
y = 3 * X + 2 + torch.randn(100, 1) * 5
# y.shape = (100, 1)

# 画出示例数据
plt.scatter(X.numpy(), y.numpy())
plt.show()


# ===================== 2. 定义线性回归模型 =====================
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        
        # y = wx + b，输出输出都是 1 维
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel()


# ===================== 3. 定义损失函数和优化器 =====================
# 均方误差作为损失函数
criterion = nn.MSELoss()

# 随机梯度下降 (SGD) 作为优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 学习率 0.01


# ===================== 4. 训练模型 =====================
epochs = 1000  # 训练1000个回合
for epoch in range(epochs):
    # 使用模型预测输出
    predictions = model(X)
    
    # 计算损失
    loss = criterion(predictions, y)
    
    # 梯度清零，防止累积
    optimizer.zero_grad()
    
    # 计算梯度
    loss.backward()
    
    # 更新参数
    optimizer.step()

    # 打印训练过程中的损失值
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# ===================== 5. 查看结果 =====================
# 使用训练好的模型预测
predicted = model(X)

# 原始数据
plt.scatter(X.numpy(), y.numpy(), label='Original Data')

# 预测数据
plt.plot(X.numpy(), predicted.detach().numpy(), color='red', label='Fitted Line')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
