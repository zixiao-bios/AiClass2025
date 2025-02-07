import json
import torch

from transformer import Transformer
from config import *
from text_process import *


def main():
    # 读入词表
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # id -> token
    id2token = {i: token for token, i in vocab.items()}
    
    # 加载模型
    model = Transformer(
        src_pad_idx=vocab['<pad>'],
        trg_pad_idx=vocab['<pad>'],
        trg_sos_idx=vocab['<bos>'],
        enc_voc_size=len(vocab),
        dec_voc_size=len(vocab),
        d_model=d_model,
        n_head=n_head,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_blocks=n_blcoks,
        drop_prob=drop_prob,
        device=device
    ).to(device)
    model.load_state_dict(torch.load('transformer.pth'))
    
    # 将模型设置为评估模式
    model.eval()

    # 评估模型
    while True:
        input_text = input('Input english: ')
        input_data = process_text(input_text, vocab, max_len, 'en').to(device)
        input_data = input_data.unsqueeze(0)  # 增加 batch 维度
        # input_data.shape = [1, max_len]
        
        # 初始化目标序列（开始符号）
        target = torch.tensor([[vocab['<bos>']]]).to(device)
        # target.shape = [1, 1]
        
        # 生成目标序列
        for _ in range(max_len - 1):  # 最大生成长度减1，因为已经有一个开始符号
            # 将当前目标序列输入到模型，获取模型输出
            output = model(input_data, target)
            # output.shape = [1, seq_len, voc_size]
            
            # 获取输出的最后一个 token（对当前时间步的预测）
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            # 形状 [1, 1]
            
            # 将生成的 token 添加到目标序列中
            target = torch.cat([target, next_token], dim=1)  # 将当前生成的 token 拼接到目标序列
            
            # 如果生成的 token 是 <eos>，停止生成
            if next_token.item() == vocab['<eos>']:  # 根据实际情况判断 eos 索引
                break
        
        # 将生成的目标序列转为文本
        output_text = idx_to_text(target[0].tolist(), id2token, 'zh')
        print(f'Output chinese: {output_text}')


if __name__ == '__main__':
    # 选择设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    main()
