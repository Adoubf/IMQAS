import json


def build_vocab():
    """
    处理json文件，读取key，并存入txt文件
    :return:
    """
    chat_to_id = json.load(open('../ner_data/char_to_id.json', mode='r', encoding='utf-8'))
    unique_words = list(chat_to_id.keys())[1:-1]
    unique_words.insert(0, '[UNK]')
    unique_words.insert(0, '[PAD]')

    # 把数据写入到文本中
    with open('../ner_data/bilstm_crf_vocab_aidoc.txt', 'w', encoding='utf-8') as file:
        for word in unique_words:
            file.write(word + '\n')


if __name__ == '__main__':
    build_vocab()
