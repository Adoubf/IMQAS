import math
import time
import torch.nn as nn
import pandas as pd
import torch
from rnn_model import RNN
from bert_chinese_encode import get_bert_encode_for_single
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 数据路径
train_data_path = "../dataset/train_data.csv"

# 读取数据并设置列名
train_data = pd.read_csv(train_data_path, header=None, sep="\t", names=["category", "line"])

# 数据集划分
train_set, valid_set = train_test_split(train_data, test_size=0.2, random_state=42)


def randomTrainingExample(data, device):
    """随机选取数据函数"""
    sample = data.sample(n=1).iloc[0]
    category = sample["category"]
    line = sample["line"]

    # 获取 BERT 编码的张量
    line_tensor = get_bert_encode_for_single(line).unsqueeze(0).to(device)
    category_tensor = torch.tensor([int(category)], device=device)

    return category_tensor, line_tensor


def train(category_tensor, line_tensor, model, criterion, optimizer, device):
    """模型训练函数"""
    hidden = model.initHidden(batch_size=1).to(device)
    optimizer.zero_grad()

    # 遍历line_tensor中的每个字的张量表示
    for i in range(line_tensor.size(1)):
        output, hidden = model(line_tensor[0][i].unsqueeze(0), hidden)  # 前向传播
    # 计算损失
    loss = criterion(output, category_tensor)
    loss.backward()   # 反向传播
    optimizer.step()    # 更新模型参数

    return output, loss.item()


def valid(category_tensor, line_tensor, model, criterion, device):
    """模型验证函数"""
    hidden = model.initHidden(batch_size=1).to(device)

    with torch.no_grad():
        for i in range(line_tensor.size(1)):
            output, hidden = model(line_tensor[0][i].unsqueeze(0), hidden)  # 前向传播

        loss = criterion(output, category_tensor)   # 计算损失

    return output, loss.item()


def timeSince(since):
    now = time.time()
    elapsed = now - since
    m = math.floor(elapsed / 60)
    s = elapsed - m * 60
    return f'{m}m {s:.2f}s'


def train_and_validate(model, train_set, valid_set, criterion, optimizer, device, n_iters=5000, plot_every=500):
    """训练和验证模型"""
    start_time = time.time()

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    train_loss_sum, train_correct = 0, 0
    valid_loss_sum, valid_correct = 0, 0

    for iter in tqdm(range(1, n_iters + 1)):
        # 获取训练和验证数据
        train_category_tensor, train_line_tensor = randomTrainingExample(train_set, device)
        valid_category_tensor, valid_line_tensor = randomTrainingExample(valid_set, device)

        # 模型训练和验证
        train_output, train_loss = train(train_category_tensor, train_line_tensor, model, criterion, optimizer, device)
        valid_output, valid_loss = valid(valid_category_tensor, valid_line_tensor, model, criterion, device)
        # 计算准确率
        train_loss_sum += train_loss
        train_correct += (train_output.argmax(1) == train_category_tensor).sum().item()
        valid_loss_sum += valid_loss
        valid_correct += (valid_output.argmax(1) == valid_category_tensor).sum().item()
        # 打印训练和验证信息
        if iter % plot_every == 0:
            train_losses.append(train_loss_sum / plot_every)
            valid_losses.append(valid_loss_sum / plot_every)

            train_accuracies.append(train_correct / plot_every)
            valid_accuracies.append(valid_correct / plot_every)

            print(f"\nIter: {iter} | Time: {timeSince(start_time)}")
            print(f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.4f}")
            print(f"Valid Loss: {valid_losses[-1]:.4f} | Valid Acc: {valid_accuracies[-1]:.4f}")

            train_loss_sum, train_correct = 0, 0
            valid_loss_sum, valid_correct = 0, 0

    return train_losses, valid_losses, train_accuracies, valid_accuracies


def save_figure(train_losses, valid_losses, train_accuracies, valid_accuracies, save_dir="./output/img"):
    """保存训练和验证的损失及准确率曲线图"""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss", color="red")
    plt.legend(loc='upper right')
    plt.title("Loss")
    plt.savefig(os.path.join(save_dir, "rnn_loss.png"))

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(valid_accuracies, label="Valid Accuracy", color="red")
    plt.legend(loc='upper left')
    plt.title("Accuracy")
    plt.savefig(os.path.join(save_dir, "rnn_accuracy.png"))


if __name__ == '__main__':
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001
    input_size = 768
    hidden_size = 128
    n_categories = 2
    n_iters = 50000
    plot_every = 1000
    # 创建模型
    rnn = RNN(input_size, hidden_size, n_categories).to(torch_device)
    # 损失函数和优化器
    criterion = nn.NLLLoss()
    # 使用随机梯度下降优化器
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    # 训练和验证模型
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_and_validate(
        rnn, train_set, valid_set, criterion, optimizer, torch_device, n_iters, plot_every
    )
    # 保存准确率和损失曲线图
    save_figure(train_losses, valid_losses, train_accuracies, valid_accuracies)

    # 保存模型
    save_dir = "../output/model"
    os.makedirs(save_dir, exist_ok=True)    # 创建保存模型的文件夹
    model_path = os.path.join(save_dir, "BERT_RNN.pth")   # 模型保存路径
    # 保存模型参数和优化器参数
    torch.save({'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)

    print(f"Model saved to {model_path}")   # 打印保存路径
