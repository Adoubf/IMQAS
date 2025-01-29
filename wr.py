import werobot
import requests
from docker_online.wechat.config import werobot_config, wechat_public_account

url = "http://0.0.0.0:5000/v1/main_serve/"

# 访问主要逻辑服务时的超时时长
TIMEOUT = 3

robot = werobot.WeRoBot(token=wechat_public_account["token"])


# 处理请求入口
@robot.handler
def doctor(message, sesssion):
    try:
        uid = message.source
        try:
            if sesssion.get(uid, None) != '1':
                sesssion[uid] = '1'
                return "您好, 我是智能客服小艾, 有什么需要帮忙的吗?"

            # 此时用户不是第一次发言
            text = message.content
        except:
            return "您好, 我是智能客服小艾, 有什么需要帮忙的吗?"

        data = {'uid': uid, 'text': text}
        # 向主要逻辑服务发送post请求，实现接口时规定是post请求
        res = requests.post(url, data=data, timeout=TIMEOUT)

        return res.text

    except Exception as e:
        print(e)
        return "机器人客服正在休息，请稍后再试..."


robot.config["HOST"] = werobot_config["HOST"]
robot.config["PORT"] = werobot_config["PORT"]   # 微信公众号的端口是80
robot.run()
