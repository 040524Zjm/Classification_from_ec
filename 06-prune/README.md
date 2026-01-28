### 项目说明：06-prune（PyTorch 剪枝示例）

本项目演示了如何使用 PyTorch 的剪枝（`torch.nn.utils.prune`）功能对神经网络进行稀疏化处理，  
包括：**单层剪枝、多层/全局剪枝** 以及 **自定义剪枝策略**。示例网络采用经典的 LeNet 结构。

---

### 一、模型结构与基础设置（`01-demo.py`）

- **设备选择**
  - `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
  - 支持自动选择 GPU 或 CPU。

- **LeNet 模型定义**
  - 两个卷积层 + 三个全连接层：
    - `conv1 = nn.Conv2d(1, 6, 3)`
    - `conv2 = nn.Conv2d(6, 16, 3)`
    - `fc1 = nn.Linear(16 * 5 * 5, 120)`
    - `fc2 = nn.Linear(120, 84)`
    - `fc3 = nn.Linear(84, 10)`
  - 前向传播：
    - 卷积 → ReLU → 2x2 最大池化。
    - 展平后经过三层线性 + ReLU（最后一层无激活，输出 logits）。

示例代码片段（节选自 `01-demo.py`）：

```python
import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

- **模型实例化**
  - `model = LeNet().to(device=device)`。

> 该脚本主要关注剪枝逻辑，本身未集成完整的训练数据加载与训练流程，可根据需要自行补充。

---

### 二、PyTorch 内置剪枝方法示例（代码中注释说明）

> 以下内容在 `01-demo.py` 中以注释形式展示，可按需逐段取消注释进行实验。

#### 1. 单层 L1 非结构化剪枝

- 对某一层（如 `model.conv1`）的 `weight` 参数进行剪枝：
  - `prune.l1_unstructured(module, name='weight', amount=0.3)`：
    - 按 L1 范数对权重排序，剪掉绝对值较小的 30%。
  - 剪枝后：
    - 原 `weight` 参数被替换为 `weight_orig`。
    - 新增 buffer：`weight_mask`。

- 可通过：
  - `list(module.named_parameters())` 查看 `weight_orig` 等参数。
  - `list(module.named_buffers())` 查看 `weight_mask` 等 buffer。

- 若希望**将剪枝结构固化**到模型中（去掉 mask 和 orig）：
  - 调用 `prune.remove(module, name='weight')`。

示例代码片段（典型单层剪枝流程）：

```python
import torch.nn.utils.prune as prune

module = model.conv1
print(list(module.named_parameters()))
print(list(module.named_buffers()))

prune.l1_unstructured(module, name='weight', amount=0.3)
print(list(module.named_parameters()))  # 出现 weight_orig
print(list(module.named_buffers()))     # 出现 weight_mask

prune.remove(module, name='weight')    # 将剪枝固化
```

#### 2. 多层结构化/非结构化剪枝

- 遍历模型的所有子模块：
  - 对 `Conv2d` 层做非结构化剪枝：
    - `prune.l1_unstructured(module, name='weight', amount=0.2)`。
  - 对 `Linear` 层做结构化剪枝（按通道 / 维度）：
    - `prune.ln_structured(module, name='weight', amount=0.4, n=2, dim=0)`。

- 剪枝后：
  - 在 `state_dict` 中可看到所有被剪枝过的层都新增了 mask / orig 等信息。

#### 3. 全局剪枝

- 先构建待剪枝参数列表：

```python
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1,  'weight'),
    (model.fc2,  'weight'),
    (model.fc3,  'weight'),
)
```

- 调用：
  - `prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)`：
    - 在所有指定权重上**统一排序**，全局剪去 20% 的权重。

---

### 三、自定义剪枝规则示例

本项目的亮点是通过继承 `prune.BasePruningMethod` 实现自定义剪枝策略。

#### 1. 自定义剪枝类：`myself_pruning_method`

- 核心要点：
  - `PRUNING_TYPE = 'unstructured'`：定义为非结构化剪枝。
  - 重写 `compute_mask(self, t, default_mask)`：
    - `t`：待剪枝的权重张量。
    - `default_mask`：与 `t` 同形状的初始 mask（一般为全 1）。
    - 示例策略：`mask.view(-1)[::2] = 0`，即**每隔一个参数剪掉一个**。
  - 返回更新后的 `mask`。

示例代码片段（节选自 `01-demo.py` 的自定义剪枝实现）：

```python
class myself_pruning_method(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

def myself_unstructured_pruning(module, name):
    myself_pruning_method.apply(module, name)
    return module
```

#### 2. 应用自定义剪枝：`myself_unstructured_pruning`

- 封装函数：
  - `myself_pruning_method.apply(module, name)`。
  - 对指定 `module` 的某个参数（如 `bias`）进行自定义剪枝。

- 示例使用（脚本主逻辑）：
  - 记录开始时间。
  - 对 `model.fc3` 的 `bias` 应用自定义剪枝：
    - `myself_unstructured_pruning(model.fc3, name='bias')`。
  - 打印 `model.fc3.bias_mask` 查看剪枝后的 mask。
  - 打印耗时。

示例代码片段（应用自定义剪枝并查看结果）：

```python
import time

start = time.time()
myself_unstructured_pruning(model.fc3, name='bias')
print(model.fc3.bias_mask)
duration = time.time() - start
print(duration * 1000, 'ms')
```

---

### 四、使用与扩展建议

- **如何运行当前示例**

```bash
cd 06-prune
python 01-demo.py
```

> 可根据需要取消/添加注释，观察不同剪枝策略下参数与 mask 的变化。

- **在实际项目中的应用建议**
  - 将本示例中的剪枝逻辑嵌入到已有训练完成的模型中（例如 CNN/RNN/Transformer 等）。
  - 在剪枝后：
    - 可以对模型进行再训练（Fine-tuning），恢复部分精度。
    - 可将剪枝后的权重导出为部署模型，配合稀疏矩阵库或硬件加速库使用。

- **进一步实验方向**
  - 设计更精细的自定义 `compute_mask` 逻辑（如基于重要性评分、梯度等）。
  - 探索结构化剪枝（按通道 / filter 维度）对模型速度与精度的综合影响。

