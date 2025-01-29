import werobot
from zhipuai import ZhipuAI
from config import werobot_config, wechat_public_account, zhipu_config


# 初始化智普
client = ZhipuAI(api_key=zhipu_config["API_KEY"])

# 创建 WeRoBot 实例（仅需要 token，未使用客服接口）
robot = werobot.WeRoBot(token=wechat_public_account["token"])


# 设置所有请求（包含文本、语音、图片等消息）入口
@robot.handler
def doctor(message, session):
    try:
        # 获得用户 uid
        uid = message.source
        try:
            # 检查 session，判断该用户是否第一次发言
            if session.get(uid, None) != "1":
                session[uid] = "1"
                return '您好，我是智能客服小艾，有什么需要帮忙的吗?'
            # 获取用户发言内容
            text = message.content
            print("用户发言内容：", text)
        except Exception as e:
            print("获取用户发言内容异常：", e)
            return '您好，我是智能客服小艾，有什么需要帮忙的吗?'

        # 使用智普 API 处理用户请求
        response = client.chat.completions.create(
            model=zhipu_config["model"],
            messages=[
                zhipu_config["massages"],
                {"role": "user", "content": text}
            ],
            top_p=zhipu_config["top_p"],
            temperature=zhipu_config["temperature"],
            max_tokens=zhipu_config["max_tokens"]
        )
        print("智普 API 回答内容：\n", response.choices[0].message.content)

        # 返回智普 API 的回答内容
        return response.choices[0].message.content

    except Exception as e:
        print("出现异常：", e)
        return "对不起，机器人客服正在休息..."


# 配置服务器监听地址和端口
robot.config["HOST"] = werobot_config["HOST"]
robot.config["PORT"] = werobot_config["PORT"]

if __name__ == "__main__":
    print("服务器启动中，监听端口 80...")
    robot.run()
