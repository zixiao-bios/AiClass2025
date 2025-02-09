from torch import nn

from layers.multi_head_attention import MultiHeadAttention
from layers.ffn import FFN


class DecoderBlock(nn.Module):
    """Transformer Decoder Block
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        :param d_model: 模型的维度（词嵌入的维度）
        :param ffn_hidden: 前馈神经网络的隐藏层维度
        :param n_head: 多头注意力的头数
        :param drop_prob: Dropout 概率
        """
        super(DecoderBlock, self).__init__()
        
        # 1. 自注意力机制（Self-Attention）
        self.self_attention = MultiHeadAttention(hidden_dim=d_model, num_heads=n_head)
        
        # 2. 第一层归一化和 dropout（自注意力层）
        self.norm1 = nn.LayerNorm(d_model)  # 第一层 Layer Normalization
        self.dropout1 = nn.Dropout(p=drop_prob)  # 第一层 Dropout
        
        # 3. Cross-Attention
        self.enc_dec_attention = MultiHeadAttention(hidden_dim=d_model, num_heads=n_head)
        
        # 4. 第二层归一化和 dropout（编码器-解码器注意力层）
        self.norm2 = nn.LayerNorm(d_model)  # 第二层 Layer Normalization
        self.dropout2 = nn.Dropout(p=drop_prob)  # 第二层 Dropout
        
        # 5. 位置前馈神经网络（Positionwise Feed-Forward Network）
        self.ffn = FFN(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        
        # 6. 第三层归一化和 dropout（前馈网络层）
        self.norm3 = nn.LayerNorm(d_model)  # 第三层 Layer Normalization
        self.dropout3 = nn.Dropout(p=drop_prob)  # 第三层 Dropout

    def forward(self, x, enc, trg_mask):
        """
        :param dec: 解码器的输入，形状为 [batch_size, seq_len, d_model]
        :param enc: 编码器的输出，形状为 [batch_size, seq_len, d_model]
        :param trg_mask: 目标序列的掩码，形状为 [batch_size, seq_len, seq_len]
        :return: 经过解码器层处理后的输出，形状为 [batch_size, seq_len, d_model]
        """

        # 1. 计算自注意力（Self-Attention）
        x_origin = x
        x = self.self_attention(x, x, x, mask=trg_mask)
        
        # 2. 残差连接和层归一化（自注意力层）
        x = self.dropout1(x)
        x = self.norm1(x + x_origin)
        
        x_origin = x
        # 3. 计算编码器-解码器的交叉注意力（Cross-Attention）
        x = self.enc_dec_attention(x_q=x, x_k=enc, x_v=enc)
        
        # 4. 残差连接和层归一化（编码器-解码器注意力层）
        x = self.dropout2(x)
        x = self.norm2(x + x_origin)

        # 5. 位置前馈神经网络
        x_origin = x
        x = self.ffn(x)

        # 6. 残差连接和层归一化
        x = self.dropout3(x)
        x = self.norm3(x + x_origin)

        return x
