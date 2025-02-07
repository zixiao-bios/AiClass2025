from torch import nn

from blocks.decoder_block import DecoderBlock
from embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    Transformer 解码器（Decoder），包括：
    1. 嵌入层（Embedding Layer）
    2. 多层解码器块（Decoder Blocks）
    3. 线性层（Linear Layer）

    用于将编码器的输出和目标输入结合，生成最终的输出。
    """

    def __init__(self, voc_size, max_len, d_model, ffn_hidden, n_head, n_blocks, drop_prob, pad_idx, device):
        """
        :param voc_size: 词汇表的大小
        :param max_len: 最大输入序列长度
        :param d_model: 模型的维度（即嵌入维度）
        :param ffn_hidden: 前馈神经网络的隐藏层维度
        :param n_head: 多头注意力机制中的头数
        :param n_blocks: 解码器 Block 的数量
        :param drop_prob: Dropout 概率
        :param pad_idx: 填充 token 的索引
        :param device: 设备
        """
        super().__init__()

        # 1. Transformer 嵌入层
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=voc_size,
                                        pad_idx=pad_idx,
                                        device=device)

        # 2. 解码器层（由多个 DecoderBlock 组成）
        block_list = []
        for _ in range(n_blocks):
            block_list.append(DecoderBlock(d_model=d_model,
                                           ffn_hidden=ffn_hidden,
                                           n_head=n_head,
                                           drop_prob=drop_prob))
        self.layers = nn.ModuleList(block_list)

        # 3. 线性层，将输出转换为词汇表大小的分布
        self.linear = nn.Linear(d_model, voc_size)

    def forward(self, trg, enc_src, trg_mask):
        """
        :param trg: 目标输入的张量，形状为 [batch_size, seq_len]，表示目标词汇索引
        :param enc_src: 编码器的输出，形状为 [batch_size, seq_len, d_model]
        :param trg_mask: 目标掩码，形状为 [batch_size, seq_len, seq_len]，用于屏蔽目标输入中不需要关注的位置
        :return: 经过解码器处理后的输出张量，形状为 [batch_size, seq_len, voc_size]
        """

        # 1. 通过嵌入层将目标输入转化为向量表示
        trg = self.emb(trg)

        # 2. 通过每个 DecoderBlock 进行处理
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask)

        # 3. 将解码器输出传入线性层，生成最终的预测词汇分布
        output = self.linear(trg)

        return output
