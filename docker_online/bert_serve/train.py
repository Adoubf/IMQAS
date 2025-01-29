import pandas as pd
from sklearn.utils import shuffle
from functools import reduce
from collections import Counter
from bert_chinese_encode import get_bert_encode
import torch
import torch.nn as nn
import time

# 加载微调网络
from finetuning_net import Net
import torch.optim as optim

max_len = 10
# 定义embedding_size, char_size
embedding_size = 768
char_size = 2 * max_len
# 实例化微调网络
net = Net(embedding_size, char_size)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义SGD优化方法
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train(train_data_labels):
    """
    description: 训练函数, 在这个过程中将更新模型参数, 并收集准确率和损失
    :param train_data_labels: 训练数据和标签的生成器对象
    :return: 整个训练过程的平均损失之和以及正确标签的累加数
    """
    # 定义训练过程的初始损失和准确率累加数
    train_running_loss = 0.0
    train_running_acc = 0.0
    # 循环遍历训练数据和标签生成器, 每个批次更新一次模型参数
    for train_tensor, train_labels in train_data_labels:
        # 初始化该批次的优化器
        optimizer.zero_grad()
        # 使用微调网络获得输出
        train_outputs = net(train_tensor)
        # 得到该批次下的平均损失
        train_loss = criterion(train_outputs, train_labels)
        # 将该批次的平均损失加到train_running_loss中
        train_running_loss += train_loss.item()
        # 损失反向传播
        train_loss.backward()
        # 优化器更新模型参数
        optimizer.step()
        # 将该批次中正确的标签数量进行累加, 以便之后计算准确率
        train_running_acc += (train_outputs.argmax(1) == train_labels).sum().item()
    return train_running_loss, train_running_acc


def valid(valid_data_labels):
    """
    description: 验证函数, 在这个过程中将验证模型的在新数据集上的标签, 收集损失和准确率
    :param valid_data_labels: 验证数据和标签的生成器对象
    :return: 整个验证过程的平均损失之和以及正确标签的累加数
    """
    # 定义训练过程的初始损失和准确率累加数
    valid_running_loss = 0.0
    valid_running_acc = 0.0
    # 循环遍历验证数据和标签生成器
    for valid_tensor, valid_labels in valid_data_labels:
        # 不自动更新梯度
        with torch.no_grad():
            # 使用微调网络获得输出
            valid_outputs = net(valid_tensor)
            # 得到该批次下的平均损失
            valid_loss = criterion(valid_outputs, valid_labels)
            # 将该批次的平均损失加到valid_running_loss中
            valid_running_loss += valid_loss.item()
            # 将该批次中正确的标签数量进行累加, 以便之后计算准确率
            valid_running_acc += (valid_outputs.argmax(1) == valid_labels).sum().item()
    return valid_running_loss, valid_running_acc


def data_loader(data_path, batch_size, split=0.2):
    data = pd.read_csv(data_path, header=None, sep='\t')

    # 打印整体数据集上的正负样本数量
    print("数据集的正负样本数量:")
    print(dict(Counter(data[0].values)))

    data = shuffle(data).reset_index(drop=True)

    split_point = int(len(data) * split)
    valid_data = data[:split_point]
    train_data = data[split_point:]

    if len(valid_data) < batch_size:
        raise ("Batch size or split not match")

    def _loader_generator(data):

        for batch in range(0, len(data), batch_size):
            batch_encoded = []
            batch_labels = []

            for item in data[batch: batch + batch_size].values.tolist():
                encoded = get_bert_encode(item[1], item[2])[0]
                batch_encoded.append(encoded)

                batch_labels.append([item[0]])

            encoded = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_encoded)
            labels = torch.tensor(reduce(lambda x, y: x + y, batch_labels))

            yield (encoded, labels)

    return _loader_generator(train_data), _loader_generator(valid_data), len(train_data), len(valid_data)


if __name__ == '__main__':
    # # 数据所在路径
    # data_path = "./train_data.csv"
    # # 定义batch_size大小
    # batch_size = 32
    #
    # train_data, valid_data, \
    # train_data_len, valid_data_len = data_loader(data_path, batch_size)
    #
    # print(next(train_data))
    # print(next(valid_data))
    # print("train_data_len:", train_data_len)
    # print("valid_data_len:", valid_data_len)

    # 定义训练轮数
    epochs = 20

    batch_size = 32

    data_path = "./train_data.csv"

    # 定义盛装每轮次的损失和准确率列表, 用于制图
    all_train_losses = []
    all_valid_losses = []
    all_train_acc = []
    all_valid_acc = []

    # 进行指定轮次的训练
    for epoch in range(epochs):
        # 打印轮次
        print("Epoch:", epoch + 1)
        # 通过数据加载器获得训练数据和验证数据生成器, 以及对应的样本数量
        train_data_labels, valid_data_labels, train_data_len, valid_data_len = data_loader(data_path, batch_size)
        # 调用训练函数进行训练
        train_running_loss, train_running_acc = train(train_data_labels)
        # 调用验证函数进行验证
        valid_running_loss, valid_running_acc = valid(valid_data_labels)
        # 计算每一轮的平均损失, train_running_loss和valid_running_loss是每个批次的平均损失之和
        # 因此将它们乘以batch_size就得到了该轮的总损失, 除以样本数即该轮次的平均损失
        train_average_loss = train_running_loss * batch_size / train_data_len
        valid_average_loss = valid_running_loss * batch_size / valid_data_len

        # train_running_acc和valid_running_acc是每个批次的正确标签累加和,
        # 因此只需除以对应样本总数即是该轮次的准确率
        train_average_acc = train_running_acc / train_data_len
        valid_average_acc = valid_running_acc / valid_data_len
        # 将该轮次的损失和准确率装进全局损失和准确率列表中, 以便制图
        all_train_losses.append(train_average_loss)
        all_valid_losses.append(valid_average_loss)
        all_train_acc.append(train_average_acc)
        all_valid_acc.append(valid_average_acc)
        # 打印该轮次下的训练损失和准确率以及验证损失和准确率
        print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
        print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)

    print('Finished Training')

    # 导入制图工具包
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    # 创建第一张画布
    plt.figure(0)

    # 绘制训练损失曲线
    plt.plot(all_train_losses, label="Train Loss")
    # 绘制验证损失曲线, 颜色为红色
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(1)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, epochs)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./loss.png")

    # 创建第二张画布
    plt.figure(1)

    # 绘制训练准确率曲线
    plt.plot(all_train_acc, label="Train Acc")

    # 绘制验证准确率曲线, 颜色为红色
    plt.plot(all_valid_acc, color="red", label="Valid Acc")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(1)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, epochs)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig("./acc.png")

    # 模型保存时间
    time_ = int(time.time())
    # 保存路径
    MODEL_PATH = './model/BERT_net_%d.pth' % time_
    # 保存模型参数
    torch.save(net.state_dict(), MODEL_PATH)
