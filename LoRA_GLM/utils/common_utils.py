import torch
import torch.nn as nn
from glm_config import *
import copy
pc = ProjectConfig()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def second2time(seconds: int):
    """
    将秒转换成时分秒。
    :param seconds: _description_
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def save_model(
        model,
        cur_save_dir: str
    ):
    """
    存储模型
    :param model:
    :param cur_save_dir: path
    :return:
    """
    if pc.use_lora:  # merge lora params with origin model
        merged_model = copy.deepcopy(model) # 未改变原模型，复制了新的保存
        # 如果直接保存，只保存的是adapter也就是lora模型的参数
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(cur_save_dir)
        # 多模态一般分开保存
    else:
        model.save_pretrained(cur_save_dir)

