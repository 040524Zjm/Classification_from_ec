### 项目说明：05-bert_distil（BERT→TextCNN 知识蒸馏）

本项目在中文新闻分类任务上实现了一个典型的**知识蒸馏（Knowledge Distillation）**流程：  
使用 BERT 作为 Teacher 模型，TextCNN 作为 Student 模型，通过蒸馏让轻量的 CNN 逼近 BERT 的分类能力。

整体流程拆分为两大阶段：
1. **Teacher 训练（train_bert）**：训练或加载 BERT 分类模型。
2. **Student 蒸馏训练（train_kd）**：利用 BERT 的输出作为软标签训练 TextCNN。

---

### 一、目录与核心文件概览

- `models/bert.py`：
  - BERT 分类模型的 `Config` 与 `Model` 定义。
- `models/textCNN.py`：
  - TextCNN 学生模型的 `Config` 与 `Model` 定义。
- `utils/utils.py`：
  - `build_dataset`：为 BERT 构建数据集。
  - `build_dataset_CNN`：为 TextCNN 构建数据集，并返回词表。
  - `build_iterator`：构建 batch 迭代器。
  - `get_time_dif`：统计训练/评估时间。
- `train_eval.py`：
  - `train`：普通训练流程（用于 BERT 或 CNN 单独训练）。
  - `evaluate`：统一评估函数。
  - `test`：加载最优模型，在测试集上输出最终指标。
  - `train_kd`：知识蒸馏训练流程（BERT→TextCNN）。
- `run.py`：
  - 项目入口，根据命令行参数选择 `train_bert` 或 `train_kd` 任务。
- `save_model/`：
  - 保存学生/教师等训练好的模型权重（如 `textCNN.pt`）。

---

### 二、数据构建流程（`utils/utils.py`）

#### 1. BERT 数据构建：`build_dataset(config)`

- 典型输入文件：
  - `data/` 下的 `train.txt`、`dev.txt`、`test.txt` 等。
  - `class.txt` 中保存类别名称。
- 步骤要点：
  - 读取每行 `text\tlabelID`。
  - 使用 BERT tokenizer 将文本转为 token ID 序列，添加 `[CLS]` / `[SEP]` 标记。
  - 截断 / Padding 至固定长度，生成 `input_ids`、`attention_mask` 等。
  - 按 `Config.batch_size` 构建迭代器：`build_iterator(train_data, config)`。

示例代码片段（典型的 BERT 数据编码逻辑，与 `utils/utils.py` 一致）：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(config.bert_path)

def encode(text, label_id):
    token = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=config.pad_size,
        return_tensors='pt'
    )
    input_ids = token['input_ids'].squeeze(0)
    attention_mask = token['attention_mask'].squeeze(0)
    token_type_ids = token['token_type_ids'].squeeze(0)
    return (input_ids, attention_mask, token_type_ids, label_id)
```

#### 2. TextCNN 数据构建：`build_dataset_CNN(cnn_config)`

- 区别在于：
  - 使用的是词级 / 词表 ID 的形式（通常不依赖 BERT tokenizer）。
  - 返回 4 个值：
    - `vocab`：词表（用于设置 `cnn_config.n_vocab`）。
    - `cnn_train_data`、`cnn_dev_data`、`cnn_test_data`：分别对应训练、验证、测试集。
  - 通过 `build_iterator` 为 TextCNN 构造批次迭代器。

示例代码片段（典型 CNN 数据构建逻辑）：

```python
def build_dataset_CNN(config):
    # 读取数据并构建词表 vocab
    # 将句子映射为词 ID 序列，pad 到固定长度
    # 返回 vocab, train_data, dev_data, test_data
    ...

vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_CNN(cnn_config)
cnn_config.n_vocab = len(vocab)
cnn_train_iter = build_iterator(cnn_train_data, cnn_config)
```

---

### 三、普通训练流程（`train_eval.py` 中的 `train`）

> 该函数可用于单独训练 BERT 或 TextCNN，本项目主要在 `train_bert` 任务中使用。

- **损失与优化**
  - 使用标准交叉熵损失：`F.cross_entropy(outputs, labels)`。
  - 使用 `AdamW` 优化器，并按是否应用权重衰减拆分参数。

- **训练循环**
  - 遍历 `config.num_epochs`：
    - 前向计算：`outputs = model(trains)`。
    - 计算损失、反向传播、参数更新。
    - 每隔固定 batch 调用 `evaluate` 在验证集上评估：
      - 若验证损失更小，则 `torch.save(model.state_dict(), config.save_path)`。

示例代码片段（节选自 `train_eval.py` 的普通训练循环）：

```python
def train(config, model, train_iter, dev_iter, test_iter):
    loss_fn = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        for i, (trains, labels) in enumerate(train_iter):
            model.zero_grad()
            outputs = model(trains)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
```

- **评估与测试**
  - `evaluate`：在验证/测试集上统计 `loss`、`acc`，并根据 `test` 标记决定是否输出分类报告与混淆矩阵。
  - `test`：加载保存的最优权重，在测试集上输出最终指标。

---

### 四、知识蒸馏训练流程（`train_kd`）

`train_kd` 将 BERT 的输出（logits）作为软标签，联合真实标签训练 TextCNN，核心步骤如下：

#### 1. 准备 Teacher 与 Student 及其数据

- 在 `run.py` 中，当 `--task train_kd` 时：
  - 加载 BERT 模型模块：`bert_module = import_module('models.bert')`。
  - 加载 TextCNN 模块：`cnn_module = import_module('models.textCNN')`。
  - 构建 BERT 数据：
    - `bert_train_data, _, _ = build_dataset(bert_config)`。
    - `bert_train_iter = build_iterator(bert_train_data, bert_config)`。
  - 构建 CNN 数据：
    - `vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_CNN(cnn_config)`。
    - 调整 `cnn_config.n_vocab = len(vocab)`。
    - 分别构建训练/验证/测试迭代器。
  - 实例化模型：
    - `bert_model = bert_module.Model(bert_config).to(bert_config.device)`。
    - `cnn_model = cnn_module.Model(cnn_config).to(cnn_config.device)`。
  - 通常 BERT 会加载预训练的 checkpoint（Teacher 已经在任务上训好）。

#### 2. 获取 Teacher 输出：`fetch_teacher_output`

- 在 `train_eval.py` 顶部定义：
  - 将 `bert_model` 切换到 `eval` 模式。
  - 遍历 `bert_train_iter`：
    - 对每个 batch 前向传播：`predict = t_model(x)`。
    - 将 logits 存入 `output_list` 返回。

> 注意：`teacher_outputs[i]` 将与 CNN 训练时第 `i` 个 batch 对齐，因此通常需要保证 BERT 与 CNN 的训练数据批次数目一致或对应关系清晰。

#### 3. 蒸馏损失函数：`loss_fn_kd`

- 使用 KL 散度 + 交叉熵的组合：
  - 温度 `T`（通常 > 1，例如 2、4）。
  - 系数 `alpha`（控制软标签损失与硬标签损失的权重）。
- 具体计算：
  - 学生输出：`output_student = F.log_softmax(outputs / T, dim=1)`。
  - 老师输出：`output_teacher = F.softmax(teacher_outputs / T, dim=1)`。
  - 软标签损失：`soft_loss = KLDivLoss(output_student, output_teacher)`。
  - 硬标签损失：`hard_loss = cross_entropy(outputs, labels)`。
  - 总损失：`KD_loss = alpha * soft_loss * T * T + (1 - alpha) * hard_loss`。

示例代码片段（节选自 `train_eval.py` 的 KD 损失定义与使用）：

```python
criterion = nn.KLDivLoss()

def loss_fn_kd(outputs, labels, teacher_outputs):
    T = 2
    alpha = 0.8

    output_student = F.log_softmax(outputs / T, dim=1)
    output_teacher = F.softmax(teacher_outputs / T, dim=1)

    soft_loss = criterion(output_student, output_teacher)
    hard_loss = F.cross_entropy(outputs, labels)
    KD_loss = alpha * soft_loss * T * T + (1 - alpha) * hard_loss
    return KD_loss
```

#### 4. CNN 训练循环（带蒸馏）

- 与普通 `train` 类似，但损失换成 `loss_fn_kd`：
  - 取当前 batch 的 `teacher_outputs[i]` 与 CNN 输出一起计算 KD 损失。
  - 反向传播、更新 CNN 参数。
  - 定期在验证集上评估，并保存最优 CNN 权重。
  - 最后在 `cnn_test_iter` 上调用 `test` 获取最终性能指标。

示例代码片段（节选自 `train_eval.py` 的 `train_kd` 主循环）：

```python
teacher_outputs = fetch_teacher_output(bert_model, bert_train_iter)

for epoch in range(cnn_config.num_epochs):
    for i, (trains, labels) in enumerate(cnn_train_iter):
        cnn_model.zero_grad()
        outputs = cnn_model(trains)
        loss = loss_fn_kd(outputs, labels, teacher_outputs[i])
        loss.backward()
        optimizer.step()
```

---

### 五、入口脚本使用方式（`run.py`）

在 `05-bert_distil` 目录下：

```bash
cd 05-bert_distil

# 1. 仅训练 BERT（teacher）模型
python run.py --task train_bert

# 2. 在已有 BERT 模型的基础上，训练 TextCNN（student）进行知识蒸馏
python run.py --task train_kd
```

> 请确保在 `train_kd` 前，BERT Teacher 已经训练好并保存好权重，且 `bert_config.save_path` 指向该模型文件。

---

### 六、实验与扩展建议

- 尝试修改：
  - 蒸馏温度 `T`、权重 `alpha`；
  - CNN 模型宽度 / 深度、卷积核尺寸等。
- 对比：
  - 单独训练的 TextCNN 与蒸馏后的 TextCNN 精度差异；
  - BERT Teacher 与 Student 在精度、推理速度、模型大小上的权衡。
- 进一步可以：
  - 尝试多任务蒸馏或中间层特征蒸馏；
  - 使用更复杂的学生模型（如 BiLSTM、Transformer encoder）进行对比实验。

