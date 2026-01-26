import requests
import time

url = 'http://127.0.0.1:3001/v1/main_server'
# text = 'GPT-5.2来了！OpenAI称其为智能体编码最强，赶超人类专家！Altman料明年1月解除“红色警报”状态'
data = {'uid': '1', 'text': 'GPT-5.2来了！OpenAI称其为智能体编码最强，赶超人类专家！Altman料明年1月解除“红色警报”状态'}

start_time = time.time()

# post 发送
res = requests.post(url,data)
cost_time = time.time() - start_time

print('文本类别: ', res.text)
print('单条样本耗时: ', cost_time * 1000, 'ms')



