import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import json

from config import *
from text_process import *
from transformer import Transformer


dataset_raw = [
    ['Hello World!', '你好，世界！'],
    ['This is my first time visiting this beautiful city.', '这是我第一次来这座美丽的城市。'],
    ['Good morning! I hope you have a wonderful day ahead.', '早上好！希望你接下来的一天都过得愉快。'],
    ['It is a pleasure to meet you. I have heard so much about you.', '很高兴见到你，我听说了很多关于你的事情。'],
    ['Could you please tell me how to get to the nearest subway station?', '请问，你能告诉我最近的地铁站怎么走吗？'],
    ['Thank you very much for your help. I really appreciate it.', '非常感谢你的帮助，我真的很感激。'],
    ['I am looking forward to our meeting next week. It will be exciting.', '我很期待我们下周的会议，这将会非常令人兴奋。'],
    ['The weather today is absolutely perfect for a walk in the park.', '今天的天气非常适合去公园散步。'],
    ['If you have any questions, please feel free to ask me anytime.', '如果你有任何问题，请随时问我。'],
    ['I have been learning Chinese for a few months, and I find it fascinating.', '我已经学习中文几个月了，我觉得这门语言很有趣。'],
    ['This restaurant has the best food in town. You should definitely try it.', '这家餐厅的食物是全城最棒的，你一定要试试。']
]

def dataset_process(vocab, dataset):
    """处理数据集，将文本转为 tensor
    """
    data_input = []
    data_target = []
    
    for text_en, text_zh in dataset:
        input_tensor = process_text(text_en, vocab, max_len, 'en')
        target_tensor = process_text(text_zh, vocab, max_len, 'zh')
        
        data_input.append(input_tensor)
        data_target.append(target_tensor)
    
    # 将 tensor 列表转为一个大 tensor
    data_input = torch.stack(data_input)
    data_target = torch.stack(data_target)
    
    return data_input, data_target


def main():
    make_vocab(dataset_raw)
    vocab = json.load(open('vocab.json', 'r'))
    id2token = {i: token for token, i in vocab.items()}
    
    data_input, data_target = dataset_process(vocab, dataset_raw)
    # data_input, data_target, vocab, id2token = data_process()
    print(data_input.shape, data_target.shape, len(vocab))
    
    dataset_train = TensorDataset(data_input, data_target)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    # 调用示例
    for input, target in dataloader_train:
        # 这里的 input 和 target 的形状都是 [batch_size, max_len]
        print(input.shape, target.shape)
        
        # 将第一句话转为文本
        print(idx_to_text(input[0].tolist(), id2token, 'en'))
        print(idx_to_text(target[0].tolist(), id2token, 'zh'))
        break
    
    # 创建模型
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
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])  # 忽略填充部分的损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for input, target in dataloader_train:
            input, target = input.to(device), target.to(device)
            
            # 把目标序列的最后一个 token 去掉，作为解码器输入（teacher forcing）
            output = model(input, target[:, :-1])
            # output: [batch_size, trg_len - 1, dec_voc_size]
            
            # 计算损失
            loss = criterion(output.reshape(-1, len(vocab)), target[:, 1:].reshape(-1))
            epoch_loss += loss.item()
            
            # 反向传播、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(dataloader_train)}')
        
        # 打印目标句子和模型输出的句子
        print('\t目标序列：', idx_to_text(target[0].tolist(), id2token, 'zh'))
        print('\t预测序列：', idx_to_text(output.argmax(dim=-1)[0].tolist(), id2token, 'zh'))
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer.pth')


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
