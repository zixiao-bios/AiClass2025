from torch import nn
import torch


class ScaleDotProductAttention(nn.Module):
    """根据指定的QKV，计算缩放点积注意力，支持多头注意力和 Mask 功能
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        """
        :param q: Query，形状为 [batch_size, head, length, d_tensor]
        :param k: Key，形状为 [batch_size, head, length, d_tensor]
        :param v: Value，形状为 [batch_size, head, length, d_tensor]
        :param mask: 掩码，用于屏蔽某些位置，形状为 [batch_size, head, length, length]
        :return: 返回计算后的 v（值）和注意力权重 score
        """
        # 获取尺寸信息
        batch_size, head, length, d_tensor = K.size()
        
        # 计算注意力分数矩阵
        # [batch_size, num_heads, seq_len, head_dim] × [batch_size, num_heads, head_dim, seq_len] 
        # -> [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_tensor, dtype=torch.float32))

        # 2. 应用掩码，屏蔽掉某些位置
        if mask is not None:
            # 将掩码为 0 的位置设置为一个非常小的数，使其注意力权重趋近于 0
            scores = scores.masked_fill(mask == 0, -10000)
        
        # Softmax归一化得到注意力权重
        # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        # [batch_size, num_heads, seq_len, seq_len] × [batch_size, num_heads, seq_len, head_dim] 
        # -> [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        return attn_output
