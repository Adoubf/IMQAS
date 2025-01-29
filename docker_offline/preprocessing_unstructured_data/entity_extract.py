import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from bilstm_crf import NER
from evaluate import extract_decode

# 如果无GPU可用，则加载时将模型映射到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
map_location = torch.device('cpu') if not torch.cuda.is_available() else None


def entity_extract(text):
    tokenizer = BertTokenizer(vocab_file='ner_data/bilstm_crf_vocab_aidoc.txt')
    # 显式地指定 map_location，保证无GPU可用时也能顺利加载
    model_param = torch.load('model/BiLSTM-CRF-final-other.bin', map_location=map_location)
    model = NER(**model_param['init']).to(device)
    model.load_state_dict(model_param['state'])

    input_text = ' '.join(list(text))
    model_inputs = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt')[0]
    model_inputs = model_inputs.to(device)

    with torch.no_grad():
        outputs = model.predict(model_inputs)

    return extract_decode(outputs, ''.join(input_text.split()))


def batch_entity_extract(data_path):
    for fn in tqdm(os.listdir(data_path)):
        fullpath = os.path.join(data_path, fn)
        out_fn = fn.replace('txt', 'csv')

        with open(fullpath, mode='r', encoding='utf8') as f, \
                open(os.path.join(prediction_result_path, out_fn), mode='w', encoding='utf8') as entities_file:
            text = f.readline()
            entities = entity_extract(text)
            print(entities)
            entities_file.write("\n".join(entities))

    print('batch_predict Finished'.center(100, '-'))


if __name__ == '__main__':
    prediction_result_path = '../output/unstructured/prediction_results'
    batch_entity_extract('../dataset/unstructured/norecognite')
