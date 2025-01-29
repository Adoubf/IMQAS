# 服务框架使用Flask
# 导入必备的工具
from flask import Flask
from flask import request

app = Flask(__name__)
# 导入发送http请求的requests工具
import requests
# 导入操作redis数据库的工具
import redis
# 导入加载json文件的工具
import json
# 导入操作neo4j数据库的工具
from neo4j import GraphDatabase
# 从配置文件中导入需要的配置
# NEO4J的连接配置
from config import NEO4J_CONFIG
# REDIS的连接配置
from config import REDIS_CONFIG, ex_time, reply_path, model_serve_url, TIMEOUT
from zhipu_ai_chat import zhipu_ai_chat

# 建立REDIS连接池
pool = redis.ConnectionPool(**REDIS_CONFIG)

# 初始化NEO4J驱动对象
_driver = GraphDatabase.driver(**NEO4J_CONFIG)


# 根据用户输入的症状进行疾病名称的查询
def query_neo4j(text):
    """
    :param text: 症状名称
    :return: 对应的疾病名称
    """
    with _driver.session() as session:
        cypher = "MATCH(a:Symptom) WHERE(%r contains a.name) WITH" \
                 " a MATCH(a)-[r:dis_to_sym]-(b:Disease) RETURN b.name LIMIT 5" % text
        record = session.run(cypher)
        result = list(map(lambda x: x[0], record))

    return result


# 实现处理不同分支函数
class Handler(object):
    def __init__(self, uid, text, r, reply):  # 类内部的函数，第一个参数是self
        self.uid = uid
        self.r = r
        self.text = text
        self.reply = reply

    def non_first_sentence(self, previous):
        # 处理用户非第一次输入
        try:
            data = {'text1': previous, 'text2': self.text}
            result = requests.post(model_serve_url, data=data, timeout=TIMEOUT)
            if not result.text: return zhipu_ai_chat(self.text)
            # print("non_first_sentence, zhipu_ai_chat")
            # return zhipu_ai_chat(self.text)
        except Exception as e:
            print("模型异常", e)
            return zhipu_ai_chat(self.text)

        s = query_neo4j(self.text)
        print(s)
        if not s: return zhipu_ai_chat(self.text)

        old_disease = self.r.hget(str(self.uid), "previous_d")
        if old_disease:
            new_disease = list(set(s) | set(old_disease))
            res = list(set(s) - set(old_disease))  # 在第一个集合中存在且不在第二个集合中存在的元素
        else:
            res = list(set(s))
            new_disease = res

        self.r.hset(str(self.uid), "previous_d", str(new_disease))
        self.r.expire(str(self.uid), ex_time)

        if not res:
            return self.reply['4']
        else:
            res = '，'.join(res)
            return self.reply['2'] % res

    def first_sentence(self):
        # 处理用户第一次输入
        s = query_neo4j(self.text)

        if not s:
            return zhipu_ai_chat(self.text)

        self.r.hset(str(self.uid), 'previous_d', str(s))
        self.r.expire(str(self.uid), ex_time)

        res = "，".join(s)

        return self.reply['2'] % res


# 实现主要逻辑服务
@app.route("/v1/main_serve/", methods=["POST"])
def main_serve():
    # 接收werobot发送的请求
    uid = request.form['uid']
    text = request.form['text']
    # 从redis中获取一个活跃的连接
    r = redis.StrictRedis(connection_pool=pool)

    # 根据用户的uid获取是否存在上一句话
    previous = r.hget(str(uid), 'previous')
    print("main_serve previous:", previous)

    # 设置当前的输入作为上一句
    r.hset(str(uid), 'previous', text)

    # 获取模板
    reply = json.load(open(reply_path, 'r'))

    # 构造handler，根据是否是第一次请求实现逻辑
    handler = Handler(uid, text, r, reply)

    # 根据previous是否存在，判断是否是第一句
    if previous:
        return handler.non_first_sentence(previous)
    else:
        return handler.first_sentence()

# 测试query_neo4j函数
# if __name__ == '__main__':
#     text = "我最近腹痛!"
#     print(query_neo4j(text))
