import pandas as pd
from datasets import Dataset, DatasetDict


def encode_label():
    label_to_index = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}

    # 将 csv 数据转换成 Dataset 类型
    train_data = pd.read_csv('../ner_data/train.csv')
    valid_data = pd.read_csv('../ner_data/valid.csv')
    train_data = Dataset.from_pandas(train_data)
    valid_data = Dataset.from_pandas(valid_data)
    corpus_data = DatasetDict({'train': train_data, 'valid': valid_data})

    # 将标签数据转换为索引表示
    def data_handler(data_labels, data_inputs):

        data_label_ids = []
        for labels in data_labels:
            label_ids = []
            for label in labels.split():
                label_ids.append(label_to_index[label])
            data_label_ids.append(label_ids)

        return {'data_labels': data_label_ids, 'data_inputs': data_inputs}

    corpus_data = corpus_data.map(data_handler,
                                  input_columns=['data_labels', 'data_inputs'], batched=True)

    # 数据存储
    corpus_data.save_to_disk('ner_data/bilstm_crf_data_aidoc')


if __name__ == '__main__':
    encode_label()
