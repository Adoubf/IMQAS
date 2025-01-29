import torch
from transformers import BertTokenizer
from bilstm_crf import NER
from evaluate import extract_decode
from tqdm import tqdm
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def entity_extract(text):

    # 构建分词器
    tokenizer = BertTokenizer(vocab_file='../ner_data/bilstm_crf_vocab_aidoc.txt')
    # 初始化模型
    model_param = torch.load('../model/BiLSTM-CRF-final.bin', map_location=torch.device('cpu'))
    model = NER(**model_param['init'])#.cuda(device)
    model.load_state_dict(model_param['state'])

    # 我们先按字将其分开，并在字之间添加空格，便于 Bert 分词器能够准确按字分割
    input_text = ' '.join(list(text))
    model_inputs = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt')[0]
    model_inputs = model_inputs.to(device)

    with torch.no_grad():
        outputs = model.predict(model_inputs)

    return extract_decode(outputs, ''.join(input_text.split()))


def batch_entity_extract(data_path):

    for fn in tqdm(os.listdir(data_path)):
        # 拼装全路径
        fullpath = os.path.join(data_path, fn)

        # 定义输出结果文件
        entities_file = open(os.path.join(prediction_result_path, fn.replace('txt', 'csv')), mode='w', encoding='utf8')

        with open(fullpath, mode='r', encoding='utf8') as f:
            # 读取文件内容
            text = ''
            for l in f.readlines():
                text += l
            # 调用单个预测模型，输出为目标劳累型实体文本列表
            entities = entity_extract(text)
            print(entities)
            # 写入识别结果文件
            entities_file.write("\n".join(entities))
    print('batch_predict Finished'.center(100, '-'))


if __name__ == '__main__':
    # prediction_result_path = '../structured/noreview/'
    # batch_entity_extract('../unstructured/norecognite')
    text = "本病是由DNA病毒的单纯疱疹病毒所致。人类单纯疱疹病毒分为两型，" \
              "即单纯疱疹病毒Ⅰ型（HSV-Ⅰ）和单纯疱疹病毒Ⅱ型（HSV-Ⅱ）。" \
              "Ⅰ型主要引起生殖器以外的皮肤黏膜（口腔黏膜）和器官（脑）的感染。" \
              "Ⅱ型主要引起生殖器部位皮肤黏膜感染。" \
              "病毒经呼吸道、口腔、生殖器黏膜以及破损皮肤进入体内，" \
              "潜居于人体正常黏膜、血液、唾液及感觉神经节细胞内。" \
              "当机体抵抗力下降时，如发热胃肠功能紊乱、月经、疲劳等时，" \
              "体内潜伏的HSV被激活而发病。"
    result = entity_extract(text)
    print(result)
