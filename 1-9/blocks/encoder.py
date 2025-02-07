from torch import nn

from blocks.encoder_block import EncoderBlock
from embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    Transformer 编码器（Encoder），包括：
    1. 嵌入层（Embedding Layer）
    2. 多层编码器块（Encoder Blocks）

    用于将输入的句子进行编码，并产生每个位置的上下文信息。
    """

    def __init__(self, voc_size, max_len, d_model, ffn_hidden, n_head, n_blocks, drop_prob, pad_idx, device):
        """
        :param voc_size: 词汇表的大小
        :param max_len: 最大输入序列长度
        :param d_model: 模型的维度（即嵌入维度）
        :param ffn_hidden: 前馈神经网络的隐藏层维度
        :param n_head: 多头注意力机制中的头数
        :param n_blocks: 编码器 Block 的数量
        :param drop_prob: Dropout 概率
        :param pad_idx: 填充 token 的索引
        :param device: 设备
        """
        super().__init__()

        # 1. Transformer 嵌入层
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=voc_size,
                                        drop_prob=drop_prob,
                                        pad_idx=pad_idx,
                                        device=device)

        # 2. 编码器层（由多个 EncoderBlock 组成）
        block_list = []
        for _ in range(n_blocks):
            block_list.append(EncoderBlock(d_model=d_model,
                                           ffn_hidden=ffn_hidden,
                                           n_head=n_head,
                                           drop_prob=drop_prob))
        self.blocks = nn.ModuleList(block_list)

    def forward(self, x):
        """
        :param x: 输入的张量，形状为 [batch_size, seq_len]，表示输入的词汇索引
        :return: 经过编码器处理后的输出张量，形状为 [batch_size, seq_len, d_model]
        """

        # 1. 通过嵌入层将输入转化为向量表示
        x = self.emb(x)

        # 2. 通过每个 EncoderBlock 进行处理
        for layer in self.blocks:
            x = layer(x)

        return x
