import numpy as np


class HMM(object):
    def __init__(self, N, M, pi=None, A=None, B=None):
        self.N = N  # 盒子数量
        self.M = M  # 球颜色数量
        self.pi = pi  # 初始概率向量
        self.A = A  # 转移概率矩阵
        self.B = B  # 观测概率矩阵

    def get_data_with_distribute(self, dist):
        # 根据给定的概率分布，返回一个索引
        return np.random.choice(np.arange(len(dist)), p=dist)

    def generate(self, T: int):
        # T 要生成的数据的数量
        # 根据给定额参数生成观测序列
        # 根据初始概率分布，获取从哪个盒子取第一个球
        z = self.get_data_with_distribute(self.pi)  # 得到的是第一个盒子的编号
        # 从上一个盒子中根据观测概率选中一个球（颜色）
        x = self.get_data_with_distribute(self.B[z])  #x代表球的颜色，0红色 1白色
        result = [x]
        for _ in range(T - 1):
            z = self.get_data_with_distribute(self.A[z])  # 得到下一个盒子
            x = self.get_data_with_distribute(self.B[z])  # 从该盒子中随机选中一个颜色
            result.append(x)

        return result

    def forward_probability(self, X):
        # 根据给定的观测序列X，计算观测序列出现的概率
        alpha = self.pi * self.B[:, X[0]]

        for x in X[1:]:
            alpha = np.matmul(alpha, self.A) * self.B[:, x]

        return alpha.sum()


if __name__ == '__main__':
    pi = np.array([.25, .25, .25, .25])
    A = np.array([
        [0, 1, 0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])
    assert len(A) == len(pi)
    assert len(A) == len(B)
    hmm = HMM(B.shape[0], B.shape[1], pi, A, B)
    seq = hmm.generate(5)
    print(seq)  # 生成5个数据

    print(hmm.forward_probability(seq))
