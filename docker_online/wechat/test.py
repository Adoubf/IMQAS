import time
import werobot
from werobot.replies import TextReply
from zhipuai import ZhipuAI
from config import werobot_config, wechat_public_account, zhipu_config

# 初始化智普
client = ZhipuAI(api_key=zhipu_config["API_KEY"])

# 创建 WeRoBot 实例（仅需要 token，未使用客服接口）
robot = werobot.WeRoBot(token=wechat_public_account["token"])

# 每段被动回复不宜过长，避免微信消息接口对长度的限制
MAX_CHUNK_SIZE = 500


def split_long_text(text, max_length=MAX_CHUNK_SIZE):
    """将长文本按指定长度分割成多段"""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


@robot.handler
def doctor(message, session):
    """
    统一处理用户请求。根据当前上下文(session)判断是否是第一次问问题，
    还是在查看分段回答的后续部分。
    """
    user_input = message.content.strip()
    uid = message.source

    # 如果用户输入“下一段”或类似关键字，表示想看剩余的内容
    if user_input in ["下一段", "more", "继续", "下一个", "next"]:
        # 检查会话中是否已经有待发送的内容
        if 'full_answer' in session and 'current_index' in session:
            full_answer = session['full_answer']
            current_index = session['current_index']
            if current_index < len(full_answer):
                chunk = full_answer[current_index]
                session['current_index'] = current_index + 1

                # 如果还有剩余的段，就提示用户输入下一段关键字继续
                if session['current_index'] < len(full_answer):
                    reply_content = f"{chunk}\n\n如果想继续查看，请回复“下一段”或“more”。"
                else:
                    reply_content = f"{chunk}\n\n已经是最后一段啦。"

                return TextReply(message=message, content=reply_content)
            else:
                return "没有更多内容了。"
        else:
            return "目前没有可查看的内容，或者会话已过期。"
    else:
        # 这是新的问题，或者用户输入的不是“下一段”
        question = user_input

        # 调用智普API，可以先做个简单的保护，比如超时5秒就返回简短答复
        # 这里示例中直接调用，如果耗时太久仍会有超时风险。
        # 建议：可换更快模型，或缩短回答。
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=zhipu_config["model"],
                messages=[
                    zhipu_config["massages"],
                    {"role": "user", "content": question}
                ],
                top_p=zhipu_config["top_p"],
                temperature=zhipu_config["temperature"],
                max_tokens=zhipu_config["max_tokens"]
            )
        except Exception as e:
            print(f"调用智普API异常: {e}")
            return "抱歉，生成回答出现异常，请稍后再试。"

        end_time = time.time()
        print(f"调用智普API耗时: {end_time - start_time:.2f}秒")

        if response and response.choices:
            full_answer_text = response.choices[0].message.content.strip()
        else:
            full_answer_text = "抱歉，我暂时无法理解您的问题。"

        # 将完整答案先分段存储
        chunks = split_long_text(full_answer_text, MAX_CHUNK_SIZE)
        session['full_answer'] = chunks
        session['current_index'] = 0
        session['question'] = question

        # 返回第一段的概述/简短回答，或者干脆只返回一个“摘要”，再让用户通过 “下一段” 获取剩余
        # 这里做法：先把第一段当做简短答复，如果只有1段，就直接显示全部
        if len(chunks) == 0:
            return "抱歉，似乎没有生成任何内容。"
        else:
            # 第一段
            first_chunk = chunks[0]
            session['current_index'] = 1  # 标记已经发送了第一段

            if len(chunks) > 1:
                # 多于1段，提示用户继续
                reply_content = (
                    f"{first_chunk}\n\n"
                    "如果想查看剩余内容，请回复“下一段”或“more”。"
                )
            else:
                # 只有一段，直接全部返回即可
                reply_content = first_chunk

            return TextReply(message=message, content=reply_content)


# 配置服务器监听地址和端口
robot.config["HOST"] = werobot_config["HOST"]
robot.config["PORT"] = werobot_config["PORT"]

if __name__ == "__main__":
    print("服务器启动中，监听端口 80...")
    robot.run()
