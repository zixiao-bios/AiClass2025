import torch
import torch.nn as nn

from layers.scale_dot_product_attention import ScaleDotProductAttention

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
        
        # 缩放点积注意力
        self.attention = ScaleDotProductAttention()
    
    def _split(self, tensor):
        """根据 head 数量拆分 QKV
        """
        batch_size, length, d_model = tensor.size()

        # 重塑维度分割多头 (新增num_heads维度)
        # [batch_size, seq_len, hidden_dim] -> [batch_size, num_heads, seq_len, head_dim]
        tensor = tensor.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

        return tensor

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        Args:
            x_q: 用于生成 Q 的序列 [batch_size, seq_len, hidden_dim]
            x_k: 用于生成 K 的序列 [batch_size, seq_len, hidden_dim]
            x_v: 用于生成 V 的序列 [batch_size, seq_len, hidden_dim]
            mask: 注意力权重的 mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x_q.shape

        # === 步骤1：生成Q/K/V并分割多头 ===
        # 线性投影
        # [batch_size, seq_len, hidden_dim]
        Q = self.W_q(x_q)
        K = self.W_k(x_k)
        V = self.W_v(x_v)
        
        # 分割多头
        Q = self._split(Q)
        K = self._split(K)
        V = self._split(V)

        # mask 也要扩展到多头
        # [batch_size, seq_len, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # === 步骤2：计算缩放点积注意力 ===
        attn_output = self.attention(Q, K, V, mask)

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
