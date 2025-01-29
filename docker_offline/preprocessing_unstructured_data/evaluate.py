import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from bilstm_crf import NER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model=None, tokenizer=None, data=None):
    """
    评估模型
    :param model: 如果是None，表示评估用的模型是已经训练好的，否则就是训练过程中传入的
    :param tokenizer: 如果是None，表示评估用的tokenizer是已经训练好的，否则就是训练过程中传入的
    :param data: 如果是None，表示评估用的数据是已经训练好的，否则就是训练过程中传入的
    :return: 返回评估指标
    """
    # 1 加载验证集，获取真实实体
    if data is None:
        data = load_from_disk('ner_data/bilstm_crf_data_aidoc')['valid']

    total_entities = get_true_entitie(data)

    model_entities = get_pred_entities(data, model, tokenizer)

    indicators = cal_prf(model_entities, total_entities)

    return indicators


def cal_prf(model_entities, total_entities):
    # 3 指标计算
    # 每个类别实体 精确率 召回率
    # 整个模型在所有数据上准确率
    indicators = []  # 保存计算的指标

    total_pred_correct = 0
    total_true_correct = 0
    for key in ['DIS', 'SYM']:  # total_entities.keys(): # key的取值这里就两个 DIS SYM
        true_entities = total_entities[key]
        true_entities_num = len(true_entities)  # TP+FN
        pred_entities = model_entities[key]  # model_entities就是通过模型预测得到的实体
        pred_entities_num = len(pred_entities)  # TP+FP

        # 计算TP，判断预测正确的数量
        pred_correct = 0  # TP
        pred_incorrect = 0  # FP
        for pred_entity in pred_entities:
            if pred_entity in true_entities:
                pred_correct += 1
                continue
            pred_incorrect += 1

        # 统计两轮循环中一共预测正确的数量和全部实体的数量
        total_pred_correct += pred_correct
        total_true_correct += true_entities_num

        # recall precision f1
        recall = pred_correct / true_entities_num if true_entities_num > 0 else 0
        precision = pred_correct / pred_entities_num if pred_entities_num > 0 else 0

        f1 = 0
        if recall != 0 or precision != 0:
            f1 = 2 * recall * precision / (recall + precision)

        print(key, '查全率: %.3f' % recall)
        print(key, '查准率: %.3f' % precision)
        print(key, 'f1: %.3f' % f1)
        print('-' * 50)

        indicators.append([recall, precision, f1])

    accuracy = total_pred_correct / total_true_correct if total_true_correct > 0 else 0
    print('准确率：%.3f' % accuracy)
    indicators.append(total_pred_correct / total_true_correct)

    return indicators


def get_pred_entities(data, model, tokenizer):
    # 2 使用模型对验证集进行预测
    if model is None:
        model_param = torch.load('model/BiLSTM-CRF-final-other.bin',
                                 map_location=torch.device('cpu'), weights_only=False)
        # 如果是通过GPU训练，在远程服务器运行，需要设置这个参数
        model = NER(**model_param['init']).to(device)  # 如果在远程服务器运行，需要注释掉cuda
        model.load_state_dict(model_param['state'])
    if tokenizer is None:
        tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')
    model_entities = {'DIS': [], 'SYM': []}

    def start_evaluate(data_inputs):
        # 对输入的文本的每个字进行编号
        model_inputs = tokenizer.encode(data_inputs, return_tensors='pt', add_special_tokens=False)[0]
        model_inputs = model_inputs.to(device)

        # 使用模型进行预测
        with torch.no_grad():
            label_list = model.predict(model_inputs)

        # 通过extract_decode()进行实体提取并保存
        extract_entities = extract_decode(label_list, ''.join(data_inputs.split()))

        nonlocal model_entities

        for key, value in extract_entities.items():
            model_entities[key].extend(value)

    data.map(start_evaluate, input_columns=['data_inputs'], batched=False)

    return model_entities


def get_true_entitie(data):
    # 检查数据集类型
    if not hasattr(data, "map"):
        raise TypeError("Expected a `datasets.Dataset` object, but got: {}".format(type(data)))

    # 统计不同类别实体的数量
    total_entities = {'DIS': [], 'SYM': []}

    def calculate_handler(data_inputs, data_labels):
        # 去掉输入数据中的空格
        data_inputs = ''.join(data_inputs.split())

        extract_entities = extract_decode(data_labels, data_inputs)

        nonlocal total_entities
        for key, value in extract_entities.items():
            total_entities[key].extend(value)

    data.map(calculate_handler, input_columns=['data_inputs', 'data_labels'])

    return total_entities


def extract_decode(data_labels, data_inputs):
    """
    根据输入的标签序列和文本，从文本中提取实体
    :param data_labels: 标签序列
    :param data_inputs: 文本
    :return:
    """
    label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    B_DIS, I_DIS = label_to_index['B-dis'], label_to_index['I-dis']
    B_SYM, I_SYM = label_to_index['B-sym'], label_to_index['I-sym']

    def extract_word(start_index, next_label):
        # 提取实体
        index, entity = start_index + 1, [data_inputs[start_index]]  # 注意 有可能不会进入下面的循环，所有需要给index赋值
        # entity = [data_inputs[start_index]]
        # 我肚子疼呀
        # Ｏ　Ｂｓｙｍ　Ｉｓｙｍ　Ｉｓｙｍ　Bsym

        for index in range(start_index + 1, len(data_labels)):  # 注意这里是data_labels长度
            if data_labels[index] != next_label:
                break
            entity.append(data_inputs[index])

        return index, ''.join(entity)

    extract_entities = {'DIS': [], 'SYM': []}
    index = 0
    next_label = {B_DIS: I_DIS, B_SYM: I_SYM}
    # print(next_label)

    word_class = {B_DIS: 'DIS', B_SYM: 'SYM'}

    while index < len(data_labels):
        label = data_labels[index]  # 注意 index不是字符串
        if label in next_label.keys():
            index, word = extract_word(index, next_label[label])
            extract_entities[word_class[label]].append(word)
            continue
        index += 1

    return extract_entities


if __name__ == '__main__':
    evaluate()
