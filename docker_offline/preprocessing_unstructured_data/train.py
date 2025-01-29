import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import BertTokenizer
from bilstm_crf import NER
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_batch_inputs(data, labels, tokenizer):
    # 函数需要返回一个按照内容长度从大到小排序过的，sentence 和 label, 还要返回 sentence 长度
    # 将批次数据的输入和标签值分开，并计算批次的输入长度
    data_inputs, data_length, data_labels = [], [], []
    for data_input, data_label in zip(data, labels):
        # 对输入句子进行编码
        data_input_encode = tokenizer.encode(data_input,
                                             return_tensors='pt',
                                             add_special_tokens=False)
        data_input_encode = data_input_encode.to(device)
        data_inputs.append(data_input_encode.squeeze())

        # 去除多余空格，计算句子长度
        data_input = ''.join(data_input.split())
        data_length.append(len(data_input))

        # 将标签转换为张量
        data_labels.append(torch.tensor(data_label, device=device))

    # 对一个批次的内容按照长度从大到小排序，符号表示降序
    sorted_index = np.argsort(-np.asarray(data_length))

    # 根据长度的索引进行排序
    sorted_inputs, sorted_labels, sorted_length = [], [], []
    for index in sorted_index:
        sorted_inputs.append(data_inputs[index])
        sorted_labels.append(data_labels[index])
        sorted_length.append(data_length[index])

    # 对张量进行填充，使其变成长度一样的张量
    pad_inputs = pad_sequence(sorted_inputs)

    return pad_inputs, sorted_labels, sorted_length


label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}


def save_train_history_metrics(train_history_list, valid_history_list, save_path_prefix):
    """
    保存训练和验证的 Recall、Precision、F1 曲线
    :param train_history_list: 训练阶段的历史指标记录（每个元素为 [Recall, Precision, F1]）
    :param valid_history_list: 验证阶段的历史指标记录（每个元素为 [Recall, Precision, F1]）
    :param save_path_prefix: 保存图片的文件名前缀
    """
    # 从历史记录中提取各指标
    train_recall = [epoch[0] for epoch in train_history_list]
    train_precision = [epoch[1] for epoch in train_history_list]
    train_f1 = [epoch[2] for epoch in train_history_list]

    valid_recall = [epoch[0] for epoch in valid_history_list]
    valid_precision = [epoch[1] for epoch in valid_history_list]
    valid_f1 = [epoch[2] for epoch in valid_history_list]

    # 映射指标到标签
    metrics = {
        "Recall": (train_recall, valid_recall),
        "Precision": (train_precision, valid_precision),
        "F1 Score": (train_f1, valid_f1),
    }

    epochs = range(1, len(train_history_list) + 1)

    # 动态绘制每个指标
    for metric_name, (train_values, valid_values) in metrics.items():
        plt.figure()

        # 绘制训练曲线
        plt.plot(epochs, train_values, label=f'Training {metric_name}', marker='o', linestyle='-')

        # 绘制验证曲线
        plt.plot(epochs, valid_values, label=f'Validation {metric_name}', marker='s', linestyle='--')

        plt.title(f'{metric_name} Curve')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)

        # 去重图例，确保只有两项
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())  # 去重后的图例

        plt.grid(True)
        plt.savefig(f'{save_path_prefix}_{metric_name.lower().replace(" ", "_")}.png')
        plt.close()

    print("训练和验证的 Recall、Precision、F1 曲线保存成功！")


def train():
    # 读取训练集
    dataset = load_from_disk('ner_data/bilstm_crf_data_aidoc')
    train_data = dataset['train']
    valid_data = dataset['valid']

    # tokenizer
    tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')

    # 构建模型
    model = NER(vocab_size=tokenizer.vocab_size, label_num=len(label_to_index)).to(device)

    # 批次大小
    batch_size = 16
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    # 训练轮数
    num_epoch = 700

    # train history
    train_history_list = []

    # valid history
    valid_history_list = []

    def start_train(data_inputs, data_labels, tokenizer):
        # 对数据进行填充补齐
        pad_inputs, sorted_labels, sorted_length = pad_batch_inputs(data_inputs, data_labels, tokenizer)

        # 计算损失
        loss = model(pad_inputs, sorted_labels, sorted_length)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计损失值
        nonlocal total_loss
        total_loss += loss.item()

    for epoch in range(num_epoch):
        # total_loss 用于统计一个 epoch 中的总损失值
        total_loss = 0.0

        train_data.map(start_train, input_columns=['data_inputs', 'data_labels'],
                       batched=True,
                       batch_size=batch_size,
                       fn_kwargs={'tokenizer': tokenizer},
                       desc='epoch: %d' % (epoch + 1))

        # 评估模型
        train_result = evaluate(model=model, tokenizer=tokenizer, data=train_data)
        train_history_list.append(train_result)

        valid_result = evaluate(model=model, tokenizer=tokenizer, data=valid_data)
        valid_history_list.append(valid_result)

        # 提取每个类别的指标
        train_recall, train_precision, train_f1 = train_result[0][0], train_result[0][1], train_result[0][2]
        valid_recall, valid_precision, valid_f1 = valid_result[0][0], valid_result[0][1], valid_result[0][2]

        # 打印总 Loss 和每个类别的 Recall, Precision, F1
        print(f'Epoch: {epoch + 1} | Total Loss: {total_loss:.3f} | '
              f'Train - Recall: {train_recall:.3f}, Precision: {train_precision:.3f}, F1: {train_f1:.3f} | '
              f'Valid - Recall: {valid_recall:.3f}, Precision: {valid_precision:.3f}, F1: {valid_f1:.3f}')

    # 保存训练和验证损失曲线
    save_train_history_metrics(
        train_history_list, valid_history_list,
        save_path_prefix='./img/bilstm_crf_metrics'
    )

    print(f"训练和验证损失曲线保存成功！")

    # 模型保存
    model.save_model('./model/BiLSTM-CRF-final.bin')
    print(f"模型保存成功！")


if __name__ == '__main__':
    train()
