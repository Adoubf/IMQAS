from zhipuai import ZhipuAI
from config import zhipu_config

# 初始化智普
client = ZhipuAI(api_key=zhipu_config["API_KEY"])


def zhipu_ai_chat(text):
    try:
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
        # print("智普 API 回答内容：", response.choices[0].message.content)

    except Exception as e:
        print("出现异常：", e)
        return "对不起，机器人客服正在休息..."

    return response.choices[0].message.content


# if __name__ == '__main__':
#     print(zhipu_ai_chat("头疼"))
