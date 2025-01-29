werobot_config = {
    "HOST": "0.0.0.0",
    "PORT": 80
}

wechat_public_account = {
    "app_id": "wx7c0b4f91529a8d3f",
    "app_secret": "8bd626b44f49e4cd0e96e49ff034a410",
    "token": "C8pj03DdsVpvfUQeOxEEX9cWYf4",
}

zhipu_config = {
    "API_KEY": "8d403a993bb49c41d23adc94dc62dcb2.dK9co2tjDcKdK1ym",
    "model": "glm-4-flash",
    "massages": {
        "role": "system",
        "content": "你是一个专业的医疗健康AI顾问，专注于提供基于科学证据的准确医疗咨询和健康建议。\n"
                "- **目标**：你的任务是以专业、清晰的方式解答用户的健康问题，帮助他们了解潜在的病症、治疗选择以及预防措施。\n"
                "- **语气**：保持温和、尊重且清晰，避免医疗术语的过度使用，确保用户能够理解。\n"
                "- **限制**：你不能提供确诊或取代医生的建议，但可以基于已知知识和用户提供的信息进行详细解释和指导。\n"
                "- **输出格式**：以结构化和易于理解的方式提供信息,不能超过300个汉字，包括症状可能的原因、需要注意的信号以及建议的下一步行动（如咨询专业医生）。"
    },
    "max_tokens": 600,
    "temperature": 0.7,
    "top_p": 0.7
}
