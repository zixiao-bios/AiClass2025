import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR


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

def data_process():
    # 1. 分词，将句子转为 token 列表
    dataset_tokenized = [
        [tokenize(text[0], 'en'), tokenize(text[1], 'zh')]
        for text in dataset_raw
    ]
    print(dataset_tokenized[0])
    

    # 2. 构建词表
    # 特殊 token
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    special_tokens = [bos, eos, pad, unk]

    # token -> id
    vocab = {token: i for i, token in enumerate(special_tokens)}
    for text_en, text_zh in dataset_tokenized:
        for token in text_en + text_zh:
            if token not in vocab:
                vocab[token] = len(vocab)
    print(vocab)

    # id -> token
    id2token = {i: token for token, i in vocab.items()}


    # 3. 添加特殊 token，并填充
    dataset_processed = []
    for text_en, text_zh in dataset_tokenized:
        text_en = add_special_token(text_en, bos, eos, pad, max_len)
        text_zh = add_special_token(text_zh, bos, eos, pad, max_len)
        dataset_processed.append([text_en, text_zh])
    print(dataset_processed[5])
    
    
    # 4. 将 token 转为 id
    dataset_train = []
    for text_en, text_zh in dataset_processed:
        text_en = [vocab.get(token, vocab[unk]) for token in text_en]
        text_zh = [vocab.get(token, vocab[unk]) for token in text_zh]
        dataset_train.append([text_en, text_zh])
    print(dataset_train[5])
    
    
    # 5. 将源语言和目标语言的 id 列表转为 2 个 tensor
    dataset_train = torch.tensor(dataset_train, dtype=torch.long)
    # 将两个句子拆开，变成两个 tensor
    data_input = dataset_train[:, 0]
    data_target = dataset_train[:, 1]
    return data_input, data_target, vocab, id2token


def main():
    data_input, data_target, vocab, id2token = data_process()
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
    
    # 创建学习率调度器，每 100 个 epoch 把学习率衰减 0.1 倍
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for input, target in dataloader_train:
            input, target = input.to(device), target.to(device)
            
            # 模型输出
            output = model(input, target[:, :-1])
            # output: [batch_size, trg_len - 1, dec_voc_size]
            
            # 计算损失
            loss = criterion(output.reshape(-1, len(vocab)), target[:, 1:].reshape(-1))
            epoch_loss += loss.item()
            
            # 反向传播、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(dataloader_train)}')
        
        # 打印目标句子和模型输出的句子
        print('\t目标序列：', idx_to_text(target[0].tolist(), id2token, 'zh'))
        print('\t预测序列：', idx_to_text(output.argmax(dim=-1)[0].tolist(), id2token, 'zh'))


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
