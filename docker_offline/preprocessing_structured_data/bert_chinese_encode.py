from transformers import AutoTokenizer, AutoModel
import torch


# 本地模型路径或 HuggingFace 模型名称
source = '../model/BertBaseChinese/'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(source)
model = AutoModel.from_pretrained(source).to(torch_device)


def get_bert_encode_for_single(text):
    """
    使用bert-chinese编码中文文本
    """
    # 对输入文本进行编码
    # add_special_tokens=False 表示不添加 [CLS] 和 [SEP] 标记
    encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(torch_device)

    # 手动添加编码结果的 batch 维度
    tokens_tensor = encoded_input['input_ids']
    # print(tokens_tensor)  # tensor([[101, 872, 1962, 8024, 2356, 3362, 833, 102]])
    attention_mask = encoded_input['attention_mask']
    # print(attention_mask) # tensor([[1, 1, 1, 1, 1, 1, 1, 1]])

    # 使用模型获取编码
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor, attention_mask=attention_mask)  # 返回的是一个元组
    # 返回最后一层的隐藏状态，去掉 batch 维度
    return outputs.last_hidden_state.squeeze(0)


# if __name__ == '__main__':
#     text = "你好，周杰伦"
#     outputs = get_bert_encode_for_single(text)
#     print(outputs)
