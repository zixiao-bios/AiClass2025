import torch
from torch import nn

from blocks.decoder import Decoder
from blocks.encoder import Encoder


class Transformer(nn.Module):
    """
    Transformer 模型，包括：
    1. 编码器（Encoder）
    2. 解码器（Decoder）

    用于将源序列转换为目标序列
    """

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, 
                 d_model, n_head, max_len, ffn_hidden, n_blocks, drop_prob, device):
        """
        Transformer 初始化函数

        :param src_pad_idx: 源序列的填充索引
        :param trg_pad_idx: 目标序列的填充索引
        :param trg_sos_idx: 目标序列的开始符号索引
        :param enc_voc_size: 源语言词汇表大小
        :param dec_voc_size: 目标语言词汇表大小
        :param d_model: 模型的维度（即词嵌入维度）
        :param n_head: 多头注意力机制中的头数
        :param max_len: 最大输入序列长度
        :param ffn_hidden: 前馈神经网络的隐藏层维度
        :param n_layers: 编码器和解码器的层数
        :param drop_prob: Dropout 概率
        :param device: 设备类型（'cuda' 或 'cpu'）
        """
        super().__init__()
        
        # 1. 保存输入的填充索引、设备等信息
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 2. 初始化编码器（Encoder）
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_blocks=n_blocks,
                               pad_idx=src_pad_idx,
                               device=device)

        # 3. 初始化解码器（Decoder）
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_blocks=n_blocks,
                               pad_idx=trg_pad_idx,
                               device=device)

    def forward(self, src, trg):
        """
        :param src: 源序列的输入张量，形状为 [batch_size, src_len]
        :param trg: 目标序列的输入张量，形状为 [batch_size, trg_len]
        :return: 经过解码器处理后的输出张量，形状为 [batch_size, trg_len, dec_voc_size]
        """
        
        # 生成目标序列的掩码
        mask = self.make_mask(trg)

        # 通过编码器计算源序列的表示
        enc_src = self.encoder(src)

        # 通过解码器计算目标序列的输出
        output = self.decoder(trg, enc_src, mask)

        return output

    def make_mask(self, trg):
        """
        生成目标序列的掩码

        :param trg: 目标序列，形状为 [batch_size, trg_len]
        :return: 目标序列的掩码，形状为 [batch_size, trg_len, trg_len]
        """
        # 生成目标序列的下三角掩码，防止模型在生成时“偷看”未来位置
        trg_len = trg.shape[1]
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        # trg_mask = [trg_len, trg_len]

        # 将掩码扩展为 [batch_size, trg_len, trg_len]
        trg_mask = trg_mask.unsqueeze(0).repeat(trg.size(0), 1, 1)
        
        return trg_mask
