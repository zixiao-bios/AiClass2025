from torch import nn

# 适配 MNIST 数据集的简单卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # x.shape = (b, 1, 28, 28)
        self.conv1 = nn.Sequential(
            # 卷积层，输入通道1，输出通道16，卷积核5x5，步长1，填充2
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=5, 
                stride=1, 
                padding=2, 
            ), 
            # x.shape = (b, 16, 28, 28)
            nn.ReLU(), 
            # 最大池化，核大小2，步长默认与核大小相同
            nn.MaxPool2d(kernel_size=2), 
            # x.shape = (b, 16, 14, 14)
        )

        self.conv2 = nn.Sequential(
            # 卷积层，输入通道16，输出通道32，卷积核5x5，步长1，填充2
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1, 
                padding=2, 
            ), 
            # x.shape = (b, 32, 14, 14)
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2), 
            # x.shape = (b, 32, 7, 7)
        )
        
        # 用 mlp 做分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), 
            nn.ReLU(), 
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # x.shape = (b, 1, 28, 28)
        x = self.conv1(x)
        # x.shape = (b, 16, 14, 14)
        x = self.conv2(x)
        # x.shape = (b, 32, 7, 7)
        
        # 将特征展平为一维向量
        x = x.view(x.size(0), -1)
        # x.shape = (b, 32*7*7)

        # 通过分类器
        x = self.classifier(x)
        # x.shape = (b, 10)

        # 在使用nn.CrossEntropyLoss时，不需要在这里应用Softmax
        return x
