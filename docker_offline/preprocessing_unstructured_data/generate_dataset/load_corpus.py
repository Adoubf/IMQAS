import pandas as pd
import json


def load_corpus():
    """
    加载语料库，转成csv文件
    :return:
    """

    train_data_file_path = '../ner_data/train.txt'
    valid_data_file_path = '../ner_data/validate.txt'

    def process_text(path, type):
        data_inputs, data_labels = [], []
        for line in open(path, mode='r', encoding='utf-8'):
            # print(line)
            data = json.loads(line)
            # print(data)
            data_inputs.append(' '.join(data['text']))
            data_labels.append(' '.join(data['label']))

        data_df = pd.DataFrame()
        data_df['data_inputs'] = data_inputs
        data_df['data_labels'] = data_labels
        data_df.to_csv('ner_data/' + type + '.csv')
        print(type, '数据量:', len(data_df))

    process_text(train_data_file_path, 'train')
    process_text(valid_data_file_path, 'valid')


if __name__ == '__main__':
    load_corpus()
