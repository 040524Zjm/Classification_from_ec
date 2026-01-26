import requests
import time

url = 'http://127.0.0.1:3001/v1/main_server'
data = {
    'uid': 'AI-6-202204',
    'text': input('请输入文本：')
}
start_time = time.time()

result = requests.post(url=url, data=data)
cost_time = time.time() - start_time

print(f'输入文本：{data["text"]}')
print(f'输出结果：{result.text}')
print(f'单条样本预测耗时：{cost_time * 1000}ms')
