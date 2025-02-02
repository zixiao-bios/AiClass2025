import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)  # 输入10维，输出1维

    def forward(self, x):
        return self.fc(x)


model = SimpleNet()

# 初始学习率 0.1
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 创建学习率调度器，每 10 个 epoch 把学习率衰减 0.1 倍
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

criterion = nn.MSELoss()

for epoch in range(50):
    optimizer.zero_grad()
    
    # 生成随机输入和目标值
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)
    
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    # 学习率递减，每个 epoch 调用一次
    scheduler.step()
    
    # 打印学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

print("训练完成！")