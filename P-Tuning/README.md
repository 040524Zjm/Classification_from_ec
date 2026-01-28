# P-Tuning 项目实现流程

## 项目概述

P-Tuning 项目使用 **P-Tuning** 技术对 BERT 模型进行参数高效的微调。P-Tuning 通过在输入序列中插入可训练的连续 prompt embeddings 来引导模型进行下游任务，无需修改模型结构，只需训练少量参数。

## 技术原理

### P-Tuning 核心思想
- **连续 Prompt**：使用可训练的连续向量（而非离散的 token）作为 prompt
- **参数效率**：只训练 prompt embeddings，冻结预训练模型参数
- **位置灵活**：Prompt tokens 可以插入到输入序列的任意位置（通常放在 [CLS] 之后）

### 实现方式
- 使用 `[unused1]`, `[unused2]`, ... 等未使用的 token 作为 prompt token 的占位符
- 这些 token 的 embeddings 在训练过程中会被更新
- 通过 MLM (Masked Language Modeling) 任务进行训练

## 项目结构

```
P-Tuning/
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── ptune_config.py          # 配置文件
├── data_handle/
│   ├── data_loader.py       # 数据加载器
│   └── data_preprocess.py   # 数据预处理
├── utils/
│   ├── common_utils.py      # 工具函数
│   ├── metirc_utils.py      # 评估指标
│   └── verbalizer.py        # 标签映射器
└── data/
    ├── train.txt            # 训练数据（格式：label\ttext）
    ├── dev.txt              # 验证数据
    └── verbalizer.txt       # 标签映射文件
```

## 实现流程

### 1. 数据准备

#### 1.1 数据格式
训练数据采用制表符分隔的格式：
```
娱乐	嗨放派怎么停播了
体育	世界杯为何迟迟不见宣传
```

#### 1.2 Verbalizer 配置
`verbalizer.txt` 文件定义了标签到 token 的映射关系：
```
娱乐: 娱乐, 综艺, 节目
体育: 体育, 运动, 比赛
...
```

#### 1.3 数据预处理 (`data_preprocess.py`)

**处理流程**：
1. **解析数据**：从 `label\ttext` 格式中提取标签和文本
2. **插入 MASK tokens**：在 [CLS] 之后插入 `max_label_len` 个 [MASK] tokens
3. **插入 Prompt tokens**：在序列开头插入 `p_embedding_num` 个 `[unused1]`, `[unused2]`, ... tokens
4. **记录位置**：记录 MASK tokens 的位置和对应的标签 token IDs

**关键代码逻辑**：
```python
# 1. 生成 MASK Tokens
mask_tokens = ['[MASK]'] * max_label_len
mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)

# 2. 构建 prompt tokens
p_tokens = ["[unused{}]".format(i + 1) for i in range(p_embedding_num)]
p_tokens_ids = tokenizer.convert_tokens_to_ids(p_tokens)

# 3. 插入 MASK -> [CLS][MASK][MASK]文本...[SEP]
tmp_input_ids = tmp_input_ids[:start_mask_position] + mask_ids + tmp_input_ids[start_mask_position:]

# 4. 插入 prompt -> [unused1][unused2]...[CLS][MASK]...[SEP]
input_ids = p_tokens_ids + input_ids

# 5. 记录 MASK 位置
mask_positions = [len(p_tokens_ids) + start_mask_position + i for i in range(max_label_len)]
```

**输出格式**：
```python
{
    'input_ids': [[unused1, unused2, ..., CLS, MASK, MASK, text, ..., SEP], ...],
    'mask_positions': [[5, 6], ...],  # MASK token 的位置
    'mask_labels': [[183, 234], ...]  # 标签对应的 token IDs
}
```

### 2. 模型配置 (`ptune_config.py`)

主要配置参数：
- `pre_model`: 预训练 BERT 模型路径
- `p_embedding_num`: Prompt token 的数量（默认 6）
- `max_label_len`: 标签最大长度（默认 2）
- `max_seq_len`: 最大序列长度（512）
- `batch_size`: 批次大小
- `learning_rate`: 学习率（5e-5）
- `epochs`: 训练轮数

### 3. 模型初始化 (`train.py`)

#### 3.1 加载模型
```python
model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
```

#### 3.2 Verbalizer 初始化
```python
verbalizer = Verbalizer(
    verbalizer_file=pc.verbalizer,
    tokenizer=tokenizer,
    max_label_len=pc.max_label_len
)
```

Verbalizer 的作用：
- 将标签文本映射到多个可能的 token（子标签）
- 在推理时，将预测的 token IDs 映射回主标签

### 4. 训练流程

#### 4.1 优化器配置
- 使用 **AdamW** 优化器
- 对 `bias` 和 `LayerNorm.weight` 不应用权重衰减

#### 4.2 学习率调度
- 使用线性学习率调度器
- 设置预热步数

#### 4.3 训练循环

**前向传播**：
```python
logits = model(
    input_ids=batch['input_ids'].to(device),
    attention_mask=batch['attention_mask'].to(device)
).logits
```

**损失计算**：
```python
# 获取真实标签对应的子标签 token IDs
mask_labels = batch['mask_labels'].numpy().tolist()
sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
sub_labels = [ele['token_ids'] for ele in sub_labels]

# 计算 MLM 损失（只在 MASK 位置计算）
loss = mlm_loss(
    logits,
    batch['mask_positions'].to(device),
    sub_labels,
    criterion,
    pc.device
)
```

**MLM 损失计算逻辑**：
- 提取 MASK 位置的 logits
- 计算这些位置与真实标签 token IDs 的交叉熵损失
- 支持多 token 标签（通过 verbalizer 找到所有可能的子标签）

#### 4.4 模型评估

**评估流程**：
1. 获取 MASK 位置的预测 token IDs
2. 通过 Verbalizer 将 token IDs 映射回主标签
3. 计算准确率、精确率、召回率、F1 分数

```python
# 获取预测结果
predictions = convert_logits_to_ids(logits, batch['mask_positions'])

# 映射到主标签
predictions = verbalizer.batch_find_main_label(predictions)
predictions = [ele['label'] for ele in predictions]

# 计算指标
metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
eval_metric = metric.compute()
```

#### 4.5 模型保存
```python
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

### 5. 推理流程 (`inference.py`)

#### 5.1 模型加载
```python
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.to(device).eval()
```

#### 5.2 输入处理
```python
# 构建输入（不包含标签）
tokenized_output = convert_example(
    examples={'text': contents},
    tokenizer=tokenizer,
    max_seq_len=128,
    max_label_len=max_label_len,
    p_embedding_num=p_embedding_num,
    train_mode=False,  # 推理模式
    return_tensor=True
)
```

#### 5.3 预测
```python
logits = model(
    input_ids=tokenized_output['input_ids'].to(device),
    attention_mask=tokenized_output['attention_mask'].to(device)
).logits

# 获取 MASK 位置的预测
predictions = convert_logits_to_ids(logits, tokenized_output['mask_positions'])

# 映射到主标签
predictions = verbalizer.batch_find_main_label(predictions)
predictions = [ele['label'] for ele in predictions]
```

## 关键技术点

### 1. Prompt Token 的选择
- 使用 `[unused1]`, `[unused2]`, ... 等未使用的 token
- 这些 token 的 embeddings 在训练时会被更新
- 数量由 `p_embedding_num` 控制（通常 6-20 个）

### 2. Verbalizer 机制
- **子标签映射**：一个主标签可以对应多个 token（子标签）
- **标签映射**：将预测的 token IDs 映射回主标签
- **提高鲁棒性**：即使预测的 token 不完全匹配，也能通过子标签映射找到正确的主标签

### 3. MLM 损失计算
- 只在 MASK 位置计算损失
- 支持多 token 标签（通过 verbalizer 找到所有可能的子标签）
- 使用交叉熵损失函数

### 4. 位置灵活性
- Prompt tokens 可以插入到任意位置
- 本项目将 prompt 放在序列开头，MASK 放在 [CLS] 之后

## 使用示例

### 训练
```bash
python train.py
```

### 推理
```python
python inference.py
```

## 注意事项

1. **数据格式**：训练数据必须是 `label\ttext` 格式
2. **Verbalizer 配置**：确保 `verbalizer.txt` 中包含了所有标签及其对应的 token
3. **Prompt 数量**：`p_embedding_num` 影响模型效果，需要根据任务调整
4. **标签长度**：`max_label_len` 需要根据实际标签的最大 token 数设置
5. **模型兼容性**：某些模型（如 RoBERTa）不需要 `token_type_ids`，代码已做兼容处理

## 配置说明

主要配置项在 `ptune_config.py` 中：
- `p_embedding_num`: Prompt token 数量，影响参数量和效果（建议 6-20）
- `max_label_len`: 标签最大 token 长度
- `max_seq_len`: 最大序列长度（BERT 最大 512）
- `verbalizer`: Verbalizer 文件路径，定义了标签到 token 的映射关系

## 与 PET 的区别

- **P-Tuning**：使用连续的可训练 embeddings 作为 prompt
- **PET**：使用硬模板（hard template）定义 prompt 的结构，但同样使用连续 embeddings
