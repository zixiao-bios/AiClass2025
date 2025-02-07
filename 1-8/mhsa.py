import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 维度校验 (确保可以整除)
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_head整除"

        # 定义四个全连接层
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # Q投影矩阵
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # K投影矩阵
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # V投影矩阵
        self.W_o = nn.Linear(hidden_dim, hidden_dim)  # 输出投影矩阵

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # === 步骤1：生成Q/K/V并分割多头 ===
        # 线性投影
        # [batch_size, seq_len, hidden_dim]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑维度分割多头 (新增num_heads维度)
        # [batch_size, seq_len, hidden_dim] -> [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === 步骤2：计算缩放点积注意力 ===
        # 计算注意力分数矩阵
        # [batch_size, num_heads, seq_len, head_dim] × [batch_size, num_heads, head_dim, seq_len] 
        # -> [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Softmax归一化得到注意力权重
        # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        # [batch_size, num_heads, seq_len, seq_len] × [batch_size, num_heads, seq_len, head_dim] 
        # -> [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # === 步骤3：拼接多头并输出投影 ===
        # 合并多头维度，contiguous()用于确保内存连续，否则后续view会报错
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 展平最后两个维度
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # 最终线性投影
        output = self.W_o(attn_output)
        # [batch_size, seq_len, hidden_dim]

        return output


# 示例用法
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    hidden_dim = 256
    num_heads = 4

    # 输入张量 [2, 5, 256]
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 初始化多头注意力层
    mha = MultiHeadAttention(hidden_dim, num_heads)
    
    # 前向传播
    output = mha(x)
    
    print("输入尺寸:", x.shape)     # torch.Size([2, 5, 256])
    print("输出尺寸:", output.shape) # torch.Size([2, 5, 256])
