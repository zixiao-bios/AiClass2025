import torch
import json
# nltk 用于英文分词
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')


def tokenize(text: str, lang='en') -> list[str]:
    """分词函数，将文本转为 token 列表
    英文按空格分割，中文按字分割
    Args:
        text (str): 输入文本
        lang (str): 语言，支持 'en', 'zh'
    """
    if lang == 'en':
        return word_tokenize(text)
    elif lang == 'zh':
        return list(text)
    else:
        raise ValueError('Invalid language')

def add_special_token(tokens, bos, eos, pad, pad_len):
    """添加特殊 token，如果长度不足则填充，长度超过则截断，截断时保留 eos
    Args:
        tokens (list): token 列表
        bos (str): 起始 token
        eos (str): 结束 token
        pad (str): 填充 token
        pad_len (int): 填充长度
    """
    tokens = [bos] + tokens + [eos]
    if len(tokens) < pad_len:
        tokens += [pad] * (pad_len - len(tokens))
    else:
        tokens = tokens[:pad_len]
        if tokens[-1] != eos:
            tokens[-1] = eos
    return tokens

def make_vocab(dataset):
    # 分词，将句子转为 token 列表
    dataset_tokenized = [
        [tokenize(text[0], 'en'), tokenize(text[1], 'zh')]
        for text in dataset
    ]
    print(dataset_tokenized[0])
    
    # 特殊 token
    special_tokens = ['<bos>', '<eos>', '<pad>', '<unk>']

    # token -> id
    vocab = {token: i for i, token in enumerate(special_tokens)}
    for text_en, text_zh in dataset_tokenized:
        for token in text_en + text_zh:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    # 把词表保存，以便之后推理时使用
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
        
    return vocab

def process_text(text, vocab, pad_len, lang):
    """将文本变为可以直接输入到模型的tensor
    """
    # 分词
    tokens = tokenize(text, lang)
    
    # 添加特殊 token
    tokens = add_special_token(tokens, '<bos>', '<eos>', '<pad>', pad_len)
    
    # token -> id
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    # 转为 tensor
    tensor = torch.tensor(ids, dtype=torch.long)
    return tensor

def idx_to_text(idx_list, id2token, lang):
    """将 id 列表转为文本
    Args:
        idx_list (list): id 列表
        id2token (dict): id 到 token 的映射
        lang (str): 语言
    """
    if lang == 'en':
        return ' '.join([id2token[i] for i in idx_list])
    elif lang == 'zh':
        return ''.join([id2token[i] for i in idx_list])
    else:
        raise ValueError('Invalid language')
