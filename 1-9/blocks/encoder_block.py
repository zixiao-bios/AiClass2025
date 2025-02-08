from torch import nn

from layers.multi_head_attention import MultiHeadAttention
from layers.ffn import FFN


class EncoderBlock(nn.Module):
    """Transformer 中的一个编码器 Block
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        :param d_model: 模型的维度、词嵌入的维度
        :param ffn_hidden: 前馈神经网络的隐藏层维度
        :param n_head: 多头注意力的头数
        :param drop_prob: Dropout 概率
        """
        super(EncoderBlock, self).__init__()
        
        # 1. 多头自注意力机制
        self.attention = MultiHeadAttention(hidden_dim=d_model, num_heads=n_head)
        
        # 2. 第一个 LayerNorm 和 dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        # 3. 位置前馈神经网络 Positionwise Feed-Forward Network（就是个MLP）
        self.ffn = FFN(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        
        # 4. 第二个 LayerNorm 和 dropout
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        :param x: 输入的张量，形状为 [batch_size, seq_len, d_model]
        :return: 经过编码器层处理后的输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # 1. 计算自注意力
        x_origin = x  # 保存输入值以用于后续的残差连接
        x = self.attention(x, x, x)  # 使用多头自注意力计算注意力输出
        
        # 2. 残差连接和层归一化
        x = self.dropout1(x)  # 对注意力输出应用 dropout
        x = self.norm1(x + x_origin)  # 将原始输入与输出相加，做残差连接后进行层归一化
        
        # 3. 位置前馈神经网络
        x_origin = x  # 保存输入值以用于后续的残差连接
        x = self.ffn(x)  # 通过位置前馈网络处理输入

        # 4. 残差连接和层归一化
        x = self.dropout2(x)  # 对前馈网络输出应用 dropout
        x = self.norm2(x + x_origin)  # 将原始输入与输出相加，做残差连接后进行层归一化

        return x
