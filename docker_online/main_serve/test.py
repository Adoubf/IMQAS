import requests

# 定义请求url和传入的data
url = "http://0.0.0.0:5000/v1/main_serve/"
# data = {"uid":"1888254", "text": "头疼"}
data = {"uid": "1888289", "text": "你是？"}

# 向服务发送post请求
res = requests.post(url, data=data)
# 打印返回的结果
print(res.text)
