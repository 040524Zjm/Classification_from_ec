import time
from crypt import methods

import jieba
import fasttext

from flask import Flask
from flask import request
app = Flask(__name__)

import requests

# 加载自定义的停用词表

# 提供已经训练好的模型路径
model_save_path = './news_fasttext_1765420862.163383.bin'
# 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
model = fasttext.load_model(model_save_path)
print('模型加载完成!')
# 设定投满分项目的服务的路由和请求方法
@app.route('/v1/main_server', methods=['POST'])
def main_server():
    # 接收来自请求方发送的服务字段
    text = request.form['text']
    # text = '智元第5000台通用具身机器人量产下线，具身机器人从技术验证全面迈入规模商用时代'
    # 对请求文本进行处理, 因为前面加载的是基于分词的模型, 所以这里也要对text进行分词操作
    text = ' '.join(jieba.lcut(text))
    # 执行模型的预测
    result = model.predict(text)
    # print(result) #(('__label__science',), array([0.26969656]))
    # print(result[0][0][9:])
    return result[0][0][9:]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
    # main_setver()