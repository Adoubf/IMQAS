import torch
from transformers import AutoTokenizer, AutoModel

# 本地模型路径或 HuggingFace 模型名称
source = '../../docker_offline/model/BertBaseChinese'

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(source)
model = AutoModel.from_pretrained(source)


def get_bert_encode(text_1, text_2, max_len=10):
    """
    对句子进行编码，生成 input_ids, attention_mask 和 token_type_ids
    :param text_1: 第一个文本
    :param text_2: 第二个文本
    :param max_len: 最大长度
    :return: 返回编码后的 input_ids, attention_mask, token_type_ids
    """
    # 使用 tokenizer 对文本进行编码
    encoded = tokenizer(
        text_1,
        text_2,
        padding='max_length',  # 自动填充到 max_len
        truncation=True,       # 截断超长文本
        max_length=max_len * 2,  # 两句话拼接长度
        return_tensors='pt'    # 返回 PyTorch 张量
    )

    input_ids = encoded['input_ids']         # 编码后的输入 ID
    attention_mask = encoded['attention_mask']  # 注意力掩码
    token_type_ids = encoded['token_type_ids']  # 分段标记

    # 使用模型生成嵌入
    with torch.no_grad():
        encoded_layers = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    return encoded_layers


# if __name__ == '__main__':
#     text_1 = "人生该如何起头"
#     text_2 = "改变要如何起手"
#     encoded_layers = get_bert_encode(text_1, text_2)[0]
#     print(encoded_layers)
#     print(encoded_layers.shape)
