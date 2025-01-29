import torch
from transformers import AutoTokenizer, AutoModel

# 本地模型路径或 HuggingFace 模型名称
source = '../../docker_offline/model/BertBaseChinese'

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(source)
model = AutoModel.from_pretrained(source)


def get_bert_encode(text_1, text_2, mark=102, max_len=10):
    """
    对句子进行编码
    :param text_1: 第一个文本
    :param text_2: 第二个文本
    :param mark: 分隔符
    :param max_len: 句子的最大长度
    :return:
    """
    indexed_token = tokenizer.encode(text_1, text_2)  # 得到的是每个字的编号 我:55

    k = indexed_token.index(mark)

    # 处理第一句话
    if len(indexed_token[:k]) >= max_len:
        indexed_token_1 = indexed_token[:max_len]
    else:
        indexed_token_1 = indexed_token[:k] + (max_len - len(indexed_token[:k])) * [0]

    # 处理第二句话
    if len(indexed_token[k:]) >= max_len:
        indexed_token_2 = indexed_token[k:k + max_len]
    else:
        indexed_token_2 = indexed_token[k:] + (max_len - len(indexed_token[k:])) * [0]

    # 把两个处理后的句子拼接成一个
    indexed_token = indexed_token_1 + indexed_token_2

    segments_ids = [0] * max_len + [1] * max_len

    segments_tensor = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_token])

    # 对每个字的编号进行编码操作，得到每个字的字向量，向量长度768
    with torch.no_grad():
        encoded_layers = model(tokens_tensor, token_type_ids=segments_tensor)[0]

    return encoded_layers


if __name__ == '__main__':
    text_1 = "人生该如何起头"
    text_2 = "改变要如何起手"
    encoded_layers = get_bert_encode(text_1, text_2)[0]
    print(encoded_layers)
    print(encoded_layers.shape)
