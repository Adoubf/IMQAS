import numpy as np


def viterbi(text, states, pi, A, B):
    """
    维特比算法实现
    :param text: 观测序列，即待分词文本
    :param states: 状态列表
    :param pi: 初始概率向量
    :param A: 状态转移概率矩阵
    :param B: 观测概率矩阵
    :return: 最大概率, 预测状态序列
    """

    delta = [{}]  # 定义delta 存的是前向概率，截至到t时刻，前面o1-ot-1概率值
    psi = {}  # 定义psi 存t-1时刻的状态
    for state in states:
        delta[0][state] = pi[state] + B[state].get(text[0], 0)  # 初始化delta 在计算概率时求了log，所以这里是加法
        psi[state] = [state]  # 初始化psi
    # psi {'B':['B'], 'M':['M'], 'S':['S'], 'E':['E']}

    for t in range(1, len(text)):  # 按照公式进行递推
        delta.append({})
        newpsi = {}

        for state in states:
            # 第一个循环 state B
            # 计算delta(t-1)(j) + aji最大值，即从上一个时刻的每个状态到当前时刻概率最大值，
            # 并保存该概率值和上一个状态，因为计算概率时取了log,所以这里是相加
            (prob, state_sequence) = max([
                (delta[t - 1][state0] + A[state0].get(state, 0), state0) for state0 in states
            ])

            # 这里计算的是当前时刻的每种状态中概率最大值
            # 注意，B[state].get(text[t])有可能text[t]不在B[state]中，
            # 此时设置为-3.14e+100表示该字无法在B[state]中出现
            # 如果设置为0，则表示求log之前的概率值为1，意味着B[state]中只能出现这个字，显然是不对的。
            delta[t][state] = prob + B[state].get(text[t], -3.14e+100)
            # 保存路径
            newpsi[state] = psi[state_sequence] + [state]

        psi = newpsi

    # 判断最后一个字对应的4种状态哪个概率最大，即为要求解的序列
    (prob, state_sequence) = max([(delta[len(text) - 1][state], state) for state in states])

    return prob, psi[state_sequence]


if __name__ == '__main__':
    A = {
        'B': {'B': -0.26268660809250016, 'E': -1.4652633398537678},
        'E': {'B': -3.14e+100, 'E': -3.14e+100},
        'M': {'B': -2.3, 'E': -1.3},
        'S': {'B': -1.3, 'E': -2.3}
    }

    B = {
        'B': {'更': -0.510825623765990, '更高': -0.916290731874155},
        'E': {'更': -0.916290731874155, '更高': -0.510825623765990},
        'M': {'更': -0.510825623765990, '更高': -0.916290731874155},
        'S': {'更': -0.916290731874155, '更高': -0.510825623765990}
    }

    pi = {'B': -0.26268660809250016, 'E': -3.14e+100, 'M': -3.14e+100, 'S': -1.4652633398537678}

    seq = '更高地举起邓小平理论的伟大旗帜'
    states = ['B', 'M', 'E', 'S']
    prob, seq = viterbi(seq, states, pi, A, B)

    print(seq)  # 生成5个数据
    print(prob)