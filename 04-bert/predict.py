# 模型预测
from importlib import import_module

import torch

CLS = '[CLS]'
# 定义模型推理函数
def inference(model, text, config, pad_size=32):
    """

    :param model: 模型
    :param text: 要被预测的
    :param config: 配置
    :param pad_size: 默认32，大于切分，小于掩码
    :return: 类型id
    """
    # 预处理
    content = config.tokenizer.tokenize(text)
    content = [CLS] + content
    seq_len = len(content)
    token_ids = config.tokenizer.convert_tokens_to_ids(content)
    if seq_len < pad_size:
        mask = [1] * seq_len + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)# 小于填充0
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size

    # 处理后的转换pytorch tensor
    x = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)
    # 增加一维度，表示batch_size = 1
    x = x.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)
    data = (x, seq_len, mask)
    # 模型推理
    output = model(data)
    # 获取类型id
    cls_id = torch.max(output.data, 1)[1]
    return cls_id

# 也可以直接获取类型名称
def id2name(cls_id, config):
    name = config.class_list[cls_id]
    return name


# 加载模型，获取数据
if __name__ == '__main__':
    # 加载训练好的模型
    X = import_module('model.bert')
    config = X.Config()
    model = X.Model(config)
    model.load_state_dict(config.save_path, map_location=config.device)
    # 获取文本数据
    text = 'GPT-5.2来了！OpenAI称其为智能体编码最强，赶超人类专家！Altman料明年1月解除“红色警报”状态'
    # 推理函数
    cls_id = inference(model, text, config)
    name = id2name(cls_id, config)