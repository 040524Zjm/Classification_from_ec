# CPU
import torch
from model.bert import Config, Model
from utils.utils import build_dataset, build_iterator
from importlib import import_module
from train_eval import test
import numpy as np

# 加载模型配置
config = Config()
print(config.device)
# 设置随机种子
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
# 迭代器获取，生成
print('Loading data for Bert Model...')
train_data, dev_data, test_data =build_dataset(config)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
# 实例化模型加载参数
model = Model(config)
# 加载模型权重
weight = torch.load(config.save_path, map_location=config.device)
# 模型架构和权重组合在一起
model.load_state_dict(weight)
# 量化
model.eval()
# macOS ARM64（M1/M2/M3）芯片对 fbgemm（x86 适配）支持差，需改用 qnnpack 引擎。
torch.backends.quantized.engine = 'qnnpack'
# quantized_model = torchao
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print(quantized_model)
# 测试模型结果
test(quantized_model, test_iter, config)
# 保存量化模型权重
torch.save(quantized_model, config.save_path2)
