import os
from tqdm import tqdm
from rnn_model import RNN
import torch
from bert_chinese_encode import get_bert_encode_for_single

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
model_path = '../output/model/BERT_RNN.pth'
input_size = 768
hidden_size = 128
n_categories = 2

# 创建模型
rnn = RNN(input_size, hidden_size, n_categories).to(torch_device)

# 加载保存的字典
checkpoint = torch.load(model_path, map_location=torch_device, weights_only=True)

# 提取模型权重并加载
rnn.load_state_dict(checkpoint['model_state_dict'])

rnn.eval()  # 切换到评估模式，关闭 Dropout 等


def rnn_output(line_tensor):
    """
    RNN 模型输出函数，用于获取 RNN 模型的输出
    :param line_tensor:  输入文本张量
    :return:  RNN 模型的输出
    """
    # 初始化隐层张量
    hidden = rnn.initHidden().to(torch_device)
    # 与训练时相同，遍历输入文本的每一个字符
    for i in range(line_tensor.size()[1]):
        # 将其逐次输送给rnn模型
        output, hidden = rnn(line_tensor[0][i].unsqueeze(0), hidden)
    # 获得rnn模型最终的输出
    return output


def predict(input_line):
    """
    预测函数，用于预测输入文本是否为命名实体
    :param input_line: 输入文本
    :return: 1 代表是命名实体，0 代表不是命名实体
    """
    with torch.no_grad():
        # 将 input_line 编码为 BERT 张量
        line_tensor = get_bert_encode_for_single(input_line).unsqueeze(0).to(torch_device)
        output = rnn_output(line_tensor)

        # 检查 output 的维度，并调整 topk 操作
        # print("Output shape:", output.shape)
        if output.dim() == 1:  # 如果 output 是一维张量
            output = output.unsqueeze(0)  # 添加批次维度

        _, topi = output.topk(1, dim=1)  # 获取最大值索引
        return topi.item()


def batch_predict(input_path, output_path):
    """
    批量预测函数，用于批量预测输入路径下的所有文件
    :param input_path:  输入路径
    :param output_path:  输出路径
    :return:  None
    """

    # 待识别的命名实体组成的文件是以疾病名称为csv文件名，
    # 文件中的每一行是该疾病对应的症状命名实体
    # 读取路径下的每一个csv文件名，装入csv列表之中
    csv_list = os.listdir(input_path)
    # 遍历每一个csv文件
    for csv in tqdm(csv_list):
        try:
            # 以读的方式打开每一个csv文件
            with open(os.path.join(input_path, csv), "r", encoding="utf-8") as fr:
                input_lines = fr.readlines()
            # 再以写的方式打开输出路径的同名csv文件
            with open(os.path.join(output_path, csv), "w", encoding="utf-8") as fw:
                # 读取csv文件的每一行
                for input_line in input_lines:
                    # print(csv, input_line)

                    # 使用模型进行预测
                    res = predict(input_line)
                    if res:  # 结果是1，说明审核成功，把文本写入到文件中
                        # 去除多余的空白行
                        clean_line = input_line.strip()
                        if not clean_line:  # 跳过空行
                            continue
                        fw.write(input_line)
        except Exception as e:
            print(f"处理文件 {csv} 时出错: {e}")
    print(f"处理文件完成")


# if __name__ == '__main__':
#     input_path = "./dataset/structured/noreview/"
#     output_path = "./output/reviewed/"
#     batch_predict(input_path, output_path)

