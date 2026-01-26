import time

import torch
from sympy.physics.units import amount
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0])) # nelement张量x的总元素个数, 处以批量大小
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)
# print(model)
# print('*' * 80)
#
# # print(model.parameters())# 可迭代的参数
#
# module = model.conv1
# print(list(module.named_parameters()))
# print('*' * 80)
# print(list(module.named_buffers())) # 缓存的
# print('*' * 80)
#
# # prune.random_unstructured(module, name='weight', amount=0.3)
# prune.l1_unstructured(module, name='weight', amount=0.3) # 绝对值小的剪掉了？
# print(list(module.named_parameters())) # weight不见了，后面新出现了weight_orig
# print('*' * 80)
# print(list(module.named_buffers())) # 变化了 weight_mask
# print('*' * 80)
# print(module.weight)
# print('*' * 80)
#
# # 在此之后必须remove永久删除
# prune.remove(module, name='weight')
# print(list(module.named_parameters())) # weight回来了，但是排在后面，且已经被剪枝了
# print('*' * 80)
# print(list(module.named_buffers())) # 变空
# print('*' * 80)
# print(module.weight) # 被剪枝了
# print('*' * 80)



# # 多参数剪枝,多头
# print(model.state_dict().keys()) # 名字和权重
# print('*' * 80)
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#     elif isinstance(module, torch.nn.Linear):
#         prune.ln_structured(module, name='weight', amount=0.4, n=2, dim=0)
#
# print(dict(model.named_buffers()).keys())
# print('*' * 80)
#
# print(model.state_dict().keys())
# # buffer里出现了mask，原本的weight消失了，变成了origin在后面
# prune.remove(model, )


# # 全局剪枝
# print(model.state_dict().keys())
# print(list(model.named_buffers()))
# print('*' * 100)
#
# parameters_to_prune = (
#     (model.conv1,'weight'),
#     (model.conv2,'weight'),
#     (model.fc1,'weight'),
#     (model.fc2,'weight'),
#     (model.fc3,'weight'),
# )
#
# prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
#
# print(model.state_dict().keys())



# 自定义剪枝
class myself_pruning_method(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    # 内部实现compute_mask函数，完成程序员自己定义的剪枝规则，本质上是如何去mask掉权重参数
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # 每隔一个参数就遮掩掉一个
        mask.view(-1)[::2] = 0
        return mask


def myself_unstructured_pruning(module, name):
    myself_pruning_method.apply(module,name)
    return module

# model = LeNet().to(device=device)

start = time.time()

myself_unstructured_pruning(model.fc3, name='bias')

print(model.fc3.bias_mask)

duration = time.time() - start
print(duration * 1000, 'ms')








