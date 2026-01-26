
import torch
from importlib import import_module
import numpy as np
from predict import inference, id2name
from flask import Flask, request

CLS = '[CLS]'
# 设置
model_name = 'bert'
x = import_module('model.' + model_name)
config = x.Config()
# 随机种子
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

# 加载模型
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))

# 创建Flask应用
app = Flask(__name__)
# # 根路由
# @app.route('/')
# def home():
#     return 'BERT分类服务正在运行'
# 定义路由，接收POST请求并进行推理
@app.route('/v1/main_server', methods=['POST'])
def main_server():
    try:
        # 从POST请求中获取用户ID和文本数据
        uid = request.form['uid']
        text = request.form['text']
        # 调用推理函数获取预测结果
        cls_id = inference(model, text, config)
        return id2name(cls_id, config)
    except Exception as e:
        print(f'Error processing request: {str(e)}')
        return 'Error processing request', 500

if __name__ == '__main__':
    app.run('0.0.0.0', port=3001)

