import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, char_size=20, embedding_size=768, dropout=0.2):
        super(Net, self).__init__()
        self.char_size = char_size
        self.embeding_size = embedding_size

        self.dropput = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(char_size*embedding_size, 8)
        self.fc2 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(-1, self.char_size*self.embeding_size)

        x = self.dropput(x)

        x = F.relu(self.fc1(x))

        x = self.dropput(x)

        x = F.relu(self.fc2(x))

        x = self.softmax(x)

        return x


if __name__ == '__main__':
    embedding_size = 768
    char_size = 20
    dropout = 0.2
    x = torch.randn(1, 20, 768)

    net = Net(char_size, embedding_size, dropout)
    print(net(x))
