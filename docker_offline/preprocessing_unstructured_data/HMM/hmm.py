import numpy as np
import os
import pickle


class HMM(object):
    def __init__(self):

        self.pi = None
        self.B = None  # 观测矩阵/发射矩阵
        self.A = None  # 转移矩阵
        self.model_file = "../../output/model/hmm_model.pkl"
        self.state_list = ['B', 'M', 'E', 'S']  # 四个状态
        self.load_param = False

    def try_load_model(self, trained):
        if trained:
            with open(self.model_file, 'rb') as f:
                self.A = pickle.load(f)
                self.B = pickle.load(f)
                self.pi = pickle.load(f)
                self.load_param = True
        else:
            self.A = {}
            self.B = {}
            self.pi = {}
            self.load_param = False

    def train(self, path):
        # path 训练样本的路径
        self.try_load_model(False)

        def init_parameters():  # 初始化参数
            for state in self.state_list:
                self.A[state] = {s: 0.0 for s in self.state_list}
                # state-> B
                # A['B'] = {'B':0, 'M':0, 'E':0, 'S':0}
                self.B[state] = {}
                # B['B'] = {}
                self.pi[state] = 0.0
                # pi['B'] = 0.0
            # A = {'B':{'B':0, 'M':0, 'E':0, 'S':0},
            #      'M':{'B':0, 'M':0, 'E':0, 'S':0},
            #      'E':{'B':0, 'M':0, 'E':0, 'S':0},
            #      'S':{'B':0, 'M':0, 'E':0, 'S':0}}
            # B = {'B':{}, 'M': {}, 'E': {}, 'S': {}}
            # pi =  {'B':0, 'M': 0, 'E': 0, 'S': 0}

        def generate_state(text):  # 把分词之后的单词转成状态
            # 第一个词是 '迈向'
            state_sequence = []
            if len(text) == 1:
                state_sequence.append('S')
            else:
                state_sequence += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return state_sequence

        init_parameters()

        with open(path, encoding='utf8') as f:
            for line in f:  # 循环取出每一行文本
                line = line.strip()
                if not line:
                    continue
                # 迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）
                word_list = [i for i in line if i != ' ']  # 取出一行中的每个字，生成列表
                # word_list : ['迈', '向', ...]
                line_sequences = line.split()  # 一行文本按照空格进行分割，得到独立的单词
                # line_sequences : ['迈向', '充满', ...]
                line_states = []
                for seq in line_sequences:  # 取出一行中的每个单词
                    # 第一个词是 '迈向'
                    # 把每个字转成对应的状态
                    line_states.extend(generate_state(seq))  # 转成状态后追加在line_states之后

                # assert 是一个用于调试的语句，常用于测试程序的某些条件是否为真，如果该条件不为真，那么程序就会抛出异常。
                assert len(word_list) == len(line_states)  # 判断字的数量和状态数量是否一致，不一致则有问题

                # SSSBEBMMMESBEBE
                # word_list [更 高 地 举 起 邓 小 平 理 论 的 伟 大 旗 帜]
                # enumerate 是 Python 的一个内置函数，用于在遍历可迭代对象时，生成一个索引-值对的迭代器。
                for idx, state in enumerate(line_states):  # 按顺序取出每一个状态，以及对应的索引
                    if idx == 0:  # 如果是一行的第一个字
                        self.pi[state] += 1  # 则pi中统计的状态数加一
                        # 如果是第一个字，state不可能是M E
                    else:
                        # 假设idx=1，
                        # idx-1=0
                        self.A[line_states[idx - 1]][state] += 1  # 否则更新状态转移矩阵
                        # A = {'B':{'E':1}}
                    # 更新观测概率矩阵
                    self.B[line_states[idx]][word_list[idx]] = self.B[line_states[idx]].get(word_list[idx], 0) + 1.0
                    # idx = 0时
                    # B = {'B':{'迈':1}, 'E':{'向':1}}
                # 经过循环之后，得到一行中A B pi的值

            # 经过外层循环后，得到整个文件中 A B pi的值

            # 计算概率，如果句子较长许多个较小的数值连乘，容易造成下溢，对于这种情况我们常常使用log函数解决。
            # 计算pi中每个状态的初始概率，分母是pi中状态的总个数
            # print('self.pi:', self.pi)
            # 假设 pi = {'B':2000, 'S':1000, 'M':0, 'E':0}
            self.pi = {k: np.log(v / np.sum(list(self.pi.values()))) if v != 0 else -3.14e+100 for k, v in
                       self.pi.items()}
            # 计算状态转移概率矩阵，要注意每一行概率的和为1，即从某个状态向另外4个状态转移概率之和为1
            # print('self.A:', self.A)
            self.A = {k: {k1: np.log(v1 / np.sum(list(v.values()))) if v1 != 0 else -3.14e+100 for k1, v1 in v.items()}
                      for k, v in self.A.items()}
            # print('self.A:', self.A)
            # 计算观测概率矩阵中，每一行之和为1，即某一个状态到所有观测结果之和为1
            self.B = {k: {k1: np.log(v1 / np.sum(list(v.values()))) for k1, v1 in v.items()}
                      for k, v in self.B.items()}
            # print('self.B', self.B)

            # 保存模型
            with open(self.model_file, 'wb') as pkl:
                pickle.dump(self.A, pkl)
                pickle.dump(self.B, pkl)
                pickle.dump(self.pi, pkl)

            return self

    def viterbi(self, text, states, pi, A, B):
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

    def cut(self, text):
        if not self.load_param:
            self.try_load_model(os.path.exists(self.model_file))

        # 得到最大概率和分词序列
        #   更 高 地 举 起 邓 小 平 理 论 的 伟 大 旗 帜
        # B
        # M
        # S
        # E
        prob, pos_list = self.viterbi(text, self.state_list, self.pi, self.A, self.B)
        begin, next_idx = 0, 0

        # 根据分词序列进行划分
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next_idx = i + 1
            elif pos == 'S':
                yield char
                next_idx = i + 1
        if next_idx < len(text):
            yield text[next_idx:]

