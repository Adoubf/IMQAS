import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# 定义计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# tensor和model都需要设置成同样的device


class CRF(nn.Module):
    def __init__(self, label_num):
        # init(int label_num) {}
        # 自己写函数时自己明确传入参数的类型
        # python定义函数时也可指定类型，这个类型知识辅助 提示作用
        # 多练
        super(CRF, self).__init__()

        # 保存
        self.label_num = label_num

        # 初始化转移分数，模型需要学习的参数
        # +2原因，增加了START_TAG END_TAG
        self.transition_scores = nn.Parameter(torch.randn(self.label_num + 2, self.label_num + 2))

        # 设置start_tag end_tag
        self.START_TAG, self.END_TAG = self.label_num, self.label_num + 1

        # 定义填充的常量
        self.fill_value = -1000

        # 保证其他状态不会转到START 到END之后不会再转到其他状态
        self.transition_scores.data[:, self.START_TAG] = self.fill_value
        self.transition_scores.data[self.END_TAG, :] = self.fill_value

    # 计算单条路径的分数
    def _get_real_path_score(self, emission_score, sequence_label):
        # 保存sequence长度
        seq_len = len(sequence_label)

        # 计算发射分数
        # emission_score中每一行是一个字的发射分数
        real_emission_score = torch.sum(emission_score[list(range(seq_len)), sequence_label])

        # 计算转移路径分数
        # [1, 2, 3] sequence_label
        # 再真实标签前后增加 start end
        # device参数，指定张量所在的设别
        b_id = torch.tensor([self.START_TAG], dtype=torch.int32, device=device)
        e_id = torch.tensor([self.END_TAG], dtype=torch.int32, device=device)
        sequence_label_expand = torch.cat([b_id, sequence_label, e_id])
        # 结果类似 [5, 1, 2, 3, 6]

        # 获取转移路径
        pre_tag = sequence_label_expand[list(range(seq_len + 1))]
        # 结果类似 [5, 1, 2, 3]
        now_tag = sequence_label_expand[list(range(1, seq_len + 2))]
        # 结果类似 [1, 2, 3, 6]
        real_transition_score = torch.sum(self.transition_scores[
                                              pre_tag, now_tag])

        # 真实路径分数
        real_path_score = real_emission_score + real_transition_score

        return real_path_score

    def _log_sum_exp(self, score):
        # 根据公式计算路径的分数
        # 为了避免计算时溢出，每个元素先减去最大值，计算完成后，再把最大值加回来
        max_score, _ = torch.max(score, dim=0)
        max_score_expand = max_score.expand(score.shape)
        return max_score + torch.log(torch.sum(torch.exp(score - max_score_expand)))

    def _expand_emission_matrix(self, emission_score):
        # 对发射分数进行扩充，因为添加了start end两个标签
        # emission_score的形状
        # [字的个数, 5] 5代表的是 len [B-dis I-dis B-sym I-sym O]
        # 获取序列长度
        # 比如emission_score对应的是 我头疼 的发射分数矩阵
        # 是 3 * 5矩阵
        seq_length = emission_score.shape[0]

        # 增加start end这两个标签
        # b_s e_s 都是1 * 7向量
        b_s = torch.tensor([[self.fill_value] * self.label_num + [0, self.fill_value]],
                           device=device)
        e_s = torch.tensor([[self.fill_value] * self.label_num + [self.fill_value, 0]],
                           device=device)

        # 进行扩展， seq_length是字的个数
        expand_matrix = self.fill_value * torch.ones([seq_length, 2], dtype=torch.float32,
                                                     device=device)
        # 3 * 2

        emission_score_expand = torch.cat([emission_score, expand_matrix], dim=1)
        # 3 * 7

        emission_score_expand = torch.cat([b_s, emission_score_expand, e_s], dim=0)
        # 5 * 7

        return emission_score_expand

    def _get_total_path_score(self, emission_score):

        # 扩展发射分数矩阵
        emission_score_expand = self._expand_emission_matrix(emission_score)

        # 计算所有路径分数
        pre = emission_score_expand[0]  # pre代表的是累计到上一个时刻，每个状态之前的所有路径分数之和
        for obs in emission_score_expand[1:]:
            # 扩展pre的维度，把pre转置，横向广播一个维度
            pre_expand = pre.reshape(-1, 1).expand([self.label_num + 2, self.label_num + 2])
            # 扩展obs的维度，纵向添加一个维度
            obs_expand = obs.expand([self.label_num + 2, self.label_num + 2])
            # 按照矩阵计算的目录，计算上一个时刻的每种状态 到这个时刻的每种状态的组合方式全部包含在矩阵运算
            score = obs_expand + pre_expand + self.transition_scores

            # 计算分数
            # print('\nscore:', score)
            # print('\nscore.shape:', score.shape)
            pre = self._log_sum_exp(score)
            # 1 x 7 每一列代表的是上一个时刻的所有状态到这个时刻的某一个状态之和
        # for结束仍然得到一个pre 代表是最后一个时刻, 1 x 7 每一列代表的是上一个时刻的所有状态到这个时刻的某一个状态之和

        # 因为for循环执行完成后，pre最后一个时刻，每个状态之前的所有路径之和
        # 最终结果计算全部路径之和，因此还需要进行最后一步计算
        return self._log_sum_exp(pre)

    def forward(self, emission_scores, sequence_labels):
        # 计算损失值
        # 是一个批次的
        total = 0.0
        for emission_score, sequence_label in zip(emission_scores, sequence_labels):
            real_path_score = self._get_real_path_score(emission_score, sequence_label)
            total_path_score = self._get_total_path_score(emission_score)
            loss = total_path_score - real_path_score
            total += loss

        return total

    def predict(self, emission_score):
        # 扩展emission_score
        emission_score_expand = self._expand_emission_matrix(emission_score)

        # 记录每个时刻对应 每个状态对应的 最大分数，以及索引
        ids = torch.zeros(1, self.label_num + 2, dtype=torch.long, device=device)
        val = torch.zeros(1, self.label_num + 2, device=device)

        pre = emission_score_expand[0]

        for obs in emission_score_expand[1:]:
            # 对pre进行旋转
            pre_extend = pre.reshape(-1, 1).expand([self.label_num + 2, self.label_num + 2])
            obs_extend = obs.expand([self.label_num + 2, self.label_num + 2])

            # 累加，矩阵对用位置进行累加，得到的结果是上一个时刻的所有状态到这个时刻的所有状态可能转移方式
            score = obs_extend + pre_extend + self.transition_scores

            # 记录当前时刻最大的分值和索引
            value, index = score.max(dim=0)
            ids = torch.cat([ids, index.unsqueeze(0)], dim=0)
            val = torch.cat([val, value.unsqueeze(0)], dim=0)

            pre = value

        # 取出最后一个时刻的最大值
        index = torch.argmax(val[-1])
        best_path = [index]

        for i in reversed(ids[1:]):
            index = i[index].item()
            best_path.append(index)

        best_path = best_path[::-1][1:-1]

        return best_path


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, label_num):
        super(BiLSTM, self).__init__()

        # embeding
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256)

        # bilstm，没有BiLSTM对象，只有LSTM，bidirectional
        self.blstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            bidirectional=True,
            num_layers=1
        )

        # 线性层, 最终输出是发射概率矩阵
        self.linear = nn.Linear(in_features=1024, out_features=label_num)

    def forward(self, inputs, length):
        # 嵌入层，得到向量
        outputs_embed = self.embed(inputs)

        # 得到的每句话的结果会被补0
        outputs_packd = pack_padded_sequence(outputs_embed, length)

        # 把压缩后的结果输入到lstm中
        outputs_blstm, (hn, cn) = self.blstm(outputs_packd)

        # 把结果长度填充一致
        outputs_paded, outputs_lengths = pad_packed_sequence(outputs_blstm)

        # 调整形状，batch_size放在下标为0的维度
        outputs_paded = outputs_paded.transpose(0, 1)

        # 线性层
        outputs_logits = self.linear(outputs_paded)

        # 取出每句话真实长度发射概率矩阵
        outputs = []

        for outputs_logit, outputs_length in zip(outputs_logits, outputs_lengths):
            outputs.append(outputs_logit[:outputs_length])

        return outputs

    def predict(self, inputs):
        output_embed = self.embed(inputs)
        # print('output_embed.shape:', output_embed.shape)

        # 在batch size增加一个维度1
        output_embed = output_embed.unsqueeze(1)
        # print('output_embed.shape1:', output_embed.shape)

        output_blstm, (hn, cn) = self.blstm(output_embed)

        output_blstm = output_blstm.squeeze(1)

        output_linear = self.linear(output_blstm)

        return output_linear


class NER(nn.Module):
    # def __init__(self, vocab_size:int, label_num:int)->None:
    # 这里的int None就是对参数和返回值的类型的提示
    def __init__(self, vocab_size, label_num):
        super(NER, self).__init__()
        # vocab_size label_num
        # BiLSTM CRF两个模型
        self.vocab_size = vocab_size
        self.label_num = label_num

        self.bilstm = BiLSTM(vocab_size=self.vocab_size, label_num=self.label_num)

        self.crf = CRF(label_num=self.label_num)

    def forward(self, inputs, labels, length):
        # 前向计算
        # bilstm的forward函数返回 发射分数矩阵
        emission_scores = self.bilstm(inputs, length)
        # 得到一个批次的损失值
        batch_loss = self.crf(emission_scores, labels)

        return batch_loss

    def predict(self, inputs):
        # 预测
        # 得到输入句子的发射分数矩阵
        # print('inputs.shape:', inputs.shape)
        emission_scores = self.bilstm.predict(inputs)
        logits = self.crf.predict(emission_scores)

        return logits

    def save_model(self, save_path):
        save_info = {
            'init': {'vocab_size': self.vocab_size, 'label_num': self.label_num},
            'state': self.state_dict()
        }
        torch.save(save_info, save_path)


# 测试
# if __name__ == '__main__':
#     char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
#                   "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}
#
#     # 参数2:标签码表对照
#     tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
#     bilstm = BiLSTM(vocab_size=len(char_to_id),
#                     label_num=len(tag_to_id), )
#     print(bilstm)
