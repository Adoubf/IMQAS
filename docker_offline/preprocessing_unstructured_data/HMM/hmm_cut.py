import os
from hmm import HMM


def load_article(fname):
    with open(fname, encoding='utf-8') as file:
        article = []
        for line in file:
            # 去除空格，以及换行符
            article.append(line.strip())
    return article


def to_region(segmentation):
    """
    将分词结果转换为区间
    :param segmentation: 已经分词的句子例如 "研究 生命 起源论"
    :return: 区间 例如 [(1,2),(3,4),(5,7)]
    """
    import re
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start, end))
        start = end
    return region


def prf(gold, pred):
    """
    计算 P R F1
    :param gold: 标准答案
    :param pred: 分词结果
    :return: P R F1
              预测值
             正      负
     真  正  TP      FN
     实
     值  负  FP      TN

    """

    #  研究 生命 起源论 [(1,2),(3,4),(5,7)] 真实值
    #  研究生 命 起源论 [(1,3),(4,4),(5,7)] 预测值
    A, B = set(to_region(gold)), set(to_region(pred))
    A_size = len(A)
    B_size = len(B)
    A_cap_B_size = len(A & B)  # 求交集，对照混淆矩阵进行思考
    p, r = A_cap_B_size / B_size, A_cap_B_size / A_size
    return p, r, 2 * p * r / (p + r)


def test():
    hmm = HMM()
    if not os.path.exists(hmm.model_file):
        hmm.train("./dataset/HMMTrainSet.txt")
        print("模型不存在，正在训练模型...")
    else:
        print("模型已经存在，正在加载模型...")
        hmm.try_load_model(True)
    print(list(hmm.cut('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')))
    print(list(hmm.cut('改判被告人死刑立即执行')))
    article1 = load_article('./dataset/test1_org.txt')
    article2 = load_article('./dataset/test2_org.txt')
    print(list(hmm.cut(article1[0])))
    print(list(hmm.cut(article2[0])))

    pred = "  ".join(list(hmm.cut(article1[0])))
    gold = load_article('./dataset/test1_cut.txt')[0]
    print(gold)
    print("精确率:%.5f, 召回率:%.5f, F1:%.5f" % prf(gold, pred))


if __name__ == '__main__':
    hmm = HMM()
    if not os.path.exists(hmm.model_file):
        hmm.train("./dataset/HMMTrainSet.txt")
        print("模型不存在，正在训练模型...")
    else:
        print("模型已经存在，正在加载模型...")
        hmm.try_load_model(True)

    article1 = load_article('./dataset/test1_org.txt')
    article2 = load_article('./dataset/test2_org.txt')
    pred = "  ".join(list(hmm.cut(article1[0])))
    gold = load_article('./dataset/test1_cut.txt')[0]
    print("精确率:%.5f, 召回率:%.5f, F1:%.5f" % prf(gold, pred))
