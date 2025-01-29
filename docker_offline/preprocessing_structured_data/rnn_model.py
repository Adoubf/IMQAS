import torch
import torch.nn as nn


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        """
        :param input_size: 输入张量最后一个维度的大小
        :param hidden_size: 隐藏层张量最后一个维度的大小
        :param output_size: 输出张量最后一个维度的大小
        """
        super(RNN, self).__init__()

        # 将隐藏层的大小写成类的内部变量
        self.hidden_size = hidden_size

        # 构建第一个线性层，输入尺寸是input_size+hidden_size, 因为真正进入全连接层的张量是X(t)+h(t-1)
        # 输出尺寸是hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # tanh
        self.tanh = nn.Tanh()

        # 构建第二个线性层，输入尺寸是hidden_size
        # 输出尺寸是output_size
        self.i2o = nn.Linear(hidden_size, output_size)

        # 定义最终输出的softmax处理层
        self.softmax = nn.LogSoftmax(dim=-1)

        # 防止过拟合的 Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input1, hidden1):
        """
        :param input1: 相当与x(t)
        :param hidden1: 相当于h(t-1)
        :return:
        """
        # 首先要进行输入张量的拼接，将x(t)和h(t-1)拼接在一起
        combined = torch.cat((input1, hidden1), 1)

        # 让输入经过隐藏层的获得hidden
        hidden = self.i2h(combined)

        # tanh层
        hidden = self.tanh(hidden)
        # 防止过拟合
        hidden = self.dropout(hidden)
        # print('hidden.shape:', hidden.shape)
        # 让输入经过输出层获得output
        output = self.i2o(hidden)

        # 让output经过softmax层
        output = self.softmax(output)
        # 返回两个张量output, hidden
        return output, hidden

    def initHidden(self, batch_size=1):
        # 支持批量初始化，自动匹配设备
        return torch.zeros(batch_size, self.hidden_size, device=torch_device)


# # 测试RNN类
# input_size = 768    # 输入张量的最后一个维度的大小
# hidden_size = 128   # 隐藏层张量的最后一个维度的大小
# n_categories = 2    # 输出张量的最后一个维度的大小
# input = torch.rand(1, input_size, device=torch_device)  # 随机生成一个输入张量
# hidden = torch.rand(1, hidden_size, device=torch_device)    # 随机生成一个隐藏层张量
#
# # 实例化RNN类
# rnn = RNN(input_size, hidden_size, n_categories).to(torch_device)
# outputs, hidden = rnn(input, hidden)
#
# print("outputs:", outputs)
# print("hidden:", hidden)

