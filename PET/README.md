# PET 项目实现流程

## 项目概述

PET 项目使用 **Pattern-Exploiting Training (PET)** 技术对 BERT 模型进行参数高效的微调。PET 是一种基于模板的提示学习方法，通过设计人工模板（hard template）来将分类任务转换为完形填空任务，然后使用 MLM 进行训练。

## 技术原理

### PET 核心思想
- **硬模板（Hard Template）**：人工定义包含 [MASK] token 的模板，将分类任务转换为完形填空
- **模板示例**：`"这是一条{MASK}评论：{textA}。"`
- **MLM 训练**：在 [MASK] 位置预测标签对应的 token
- **参数效率**：只训练模型参数（包括 prompt embeddings），无需修改模型结构

### 与 P-Tuning 的区别
- **PET**：使用硬模板定义 prompt 结构，模板中包含具体的文本和 [MASK]
- **P-Tuning**：使用连续的 prompt embeddings，不包含具体的文本模板

## 项目结构

```
PET/
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── pet_config.py            # 配置文件
├── data_handle/
│   ├── data_loader.py       # 数据加载器
│   ├── data_preprocess.py   # 数据预处理
│   └── template.py          # 硬模板处理
├── utils/
│   ├── common_utils.py      # 工具函数
│   ├── metric_utils.py      # 评估指标
│   └── verbalizer.py        # 标签映射器
└── data/
    ├── train.txt            # 训练数据（格式：label\ttext）
    ├── dev.txt              # 验证数据
    ├── prompt.txt           # 模板定义文件
    └── verbalizer.txt       # 标签映射文件
```

## 实现流程

### 1. 数据准备

#### 1.1 数据格式
训练数据采用制表符分隔的格式：
```
手机	这个手机也太卡了。
体育	世界杯为何迟迟不见宣传
```

#### 1.2 模板定义 (`prompt.txt`)
定义硬模板，例如：
```
这是一条{MASK}评论：{textA}。
```

模板说明：
- `{MASK}`: 将被替换为多个 [MASK] tokens（数量由 `max_label_len` 决定）
- `{textA}`: 将被替换为实际的文本内容

#### 1.3 Verbalizer 配置
`verbalizer.txt` 文件定义了标签到 token 的映射关系：
```
手机: 手机, 数码, 电子
体育: 体育, 运动, 比赛
...
```

### 2. 硬模板处理 (`template.py`)

#### 2.1 模板解析
`HardTemplate` 类负责解析和处理硬模板：

```python
class HardTemplate:
    def prompt_analysis(self):
        # 解析模板，提取自定义字段
        # prompt -> "这是一条{MASK}评论：{textA}。"
        # inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
        # custom_tokens -> {'textA', 'MASK'}
```

#### 2.2 模板应用
```python
def __call__(self, inputs_dict, tokenizer, mask_length, max_seq_len):
    # 将模板中的 {MASK} 替换为多个 [MASK] tokens
    # 将 {textA} 替换为实际文本
    # 进行 tokenization
    # 记录 [MASK] token 的位置
```

**处理流程**：
1. 解析模板，识别 `{MASK}` 和 `{textA}` 等自定义字段
2. 将 `{MASK}` 替换为 `mask_length` 个 `[MASK]` tokens
3. 将 `{textA}` 替换为实际文本内容
4. 使用 tokenizer 进行编码
5. 记录所有 [MASK] token 的位置

**示例**：
```python
# 输入
inputs_dict = {
    'textA': '这个手机也太卡了。',
    'MASK': '[MASK]'
}
mask_length = 2

# 输出
# 文本: "这是一条[MASK][MASK]评论：这个手机也太卡了。"
# input_ids: [CLS, 这, 是, 一, 条, [MASK], [MASK], 评, 论, ：, 这个, 手机, ..., SEP]
# mask_position: [5, 6]  # [MASK] token 的位置
```

### 3. 数据预处理 (`data_preprocess.py`)

#### 3.1 处理流程
```python
def convert_example(examples, tokenizer, hard_template, max_seq_len, max_label_len, train_mode):
    # 1. 解析数据：label\ttext
    if train_mode:
        label, content = example.strip().split('\t')
    else:
        content = example.strip()
    
    # 2. 构建输入字典
    inputs_dict = {
        'textA': content,
        'MASK': '[MASK]'
    }
    
    # 3. 应用硬模板
    encoded_inputs = hard_template(
        inputs_dict=inputs_dict,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        mask_length=max_label_len
    )
    
    # 4. 处理标签（训练模式）
    if train_mode:
        label_encoded = tokenizer(text=[label])
        label_encoded = label_encoded['input_ids'][0][1:-1]  # 去掉 [CLS] 和 [SEP]
        label_encoded = label_encoded[:max_label_len]
        label_encoded += [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))
```

#### 3.2 输出格式
```python
{
    'input_ids': [[CLS, 这, 是, 一, 条, MASK, MASK, 评, 论, ：, text, ..., SEP], ...],
    'token_type_ids': [[0, 0, ...], ...],
    'attention_mask': [[1, 1, ...], ...],
    'mask_positions': [[5, 6], ...],  # [MASK] token 的位置
    'mask_labels': [[2372, 3442], ...]  # 标签对应的 token IDs
}
```

### 4. 模型配置 (`pet_config.py`)

主要配置参数：
- `pre_model`: 预训练 BERT 模型路径
- `prompt_file`: 模板文件路径
- `verbalizer`: Verbalizer 文件路径
- `max_label_len`: 标签最大长度（默认 2）
- `max_seq_len`: 最大序列长度（512）
- `batch_size`: 批次大小
- `learning_rate`: 学习率（5e-5）
- `epochs`: 训练轮数

### 5. 模型初始化 (`train.py`)

#### 5.1 加载模型
```python
model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
```

#### 5.2 初始化组件
```python
# Verbalizer：标签映射器
verbalizer = Verbalizer(
    verbalizer_file=pc.verbalizer,
    tokenizer=tokenizer,
    max_label_len=pc.max_label_len
)

# HardTemplate：硬模板处理器
prompt = open(pc.prompt_file, 'r', encoding='utf8').readlines()[0].strip()
hard_template = HardTemplate(prompt=prompt)
```

### 6. 训练流程

#### 6.1 优化器配置
- 使用 **AdamW** 优化器
- 对 `bias` 和 `LayerNorm.weight` 不应用权重衰减

#### 6.2 学习率调度
- 使用线性学习率调度器
- 设置预热步数

#### 6.3 训练循环

**前向传播**：
```python
logits = model(
    input_ids=batch['input_ids'].to(device),
    token_type_ids=batch['token_type_ids'].to(device),
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

#### 6.4 模型评估

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

#### 6.5 模型保存
```python
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

### 7. 推理流程 (`inference.py`)

#### 7.1 模型加载
```python
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.to(device).eval()
```

#### 7.2 初始化组件
```python
verbalizer = Verbalizer(
    verbalizer_file='data/verbalizer.txt',
    tokenizer=tokenizer,
    max_label_len=max_label_len
)

prompt = open('data/prompt.txt', 'r', encoding='utf8').readlines()[0].strip()
hard_template = HardTemplate(prompt=prompt)
```

#### 7.3 输入处理
```python
# 构建输入（不包含标签）
examples = {'text': contents}
tokenized_output = convert_example(
    examples=examples,
    tokenizer=tokenizer,
    hard_template=hard_template,
    max_seq_len=128,
    max_label_len=max_label_len,
    train_mode=False,  # 推理模式
    return_tensor=True
)
```

#### 7.4 预测
```python
logits = model(
    input_ids=tokenized_output['input_ids'].to(device),
    token_type_ids=tokenized_output['token_type_ids'].to(device),
    attention_mask=tokenized_output['attention_mask'].to(device)
).logits

# 获取 MASK 位置的预测
predictions = convert_logits_to_ids(logits, tokenized_output['mask_positions'])

# 映射到主标签
predictions = verbalizer.batch_find_main_label(predictions)
predictions = [ele['label'] for ele in predictions]
```

## 关键技术点

### 1. 硬模板设计
- **模板结构**：包含固定文本和可替换字段
- **字段类型**：
  - `{MASK}`: 自动替换为多个 [MASK] tokens
  - `{textA}`, `{textB}`: 替换为实际文本内容
- **灵活性**：可以根据任务设计不同的模板

### 2. Verbalizer 机制
- **子标签映射**：一个主标签可以对应多个 token（子标签）
- **标签映射**：将预测的 token IDs 映射回主标签
- **提高鲁棒性**：即使预测的 token 不完全匹配，也能通过子标签映射找到正确的主标签

### 3. MLM 损失计算
- 只在 MASK 位置计算损失
- 支持多 token 标签（通过 verbalizer 找到所有可能的子标签）
- 使用交叉熵损失函数

### 4. 模板解析
- 自动识别模板中的自定义字段
- 支持多个自定义字段（如 `{textA}`, `{textB}`, `{MASK}`）
- 灵活处理字段位置

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
2. **模板设计**：模板需要根据任务特点设计，影响模型效果
3. **Verbalizer 配置**：确保 `verbalizer.txt` 中包含了所有标签及其对应的 token
4. **标签长度**：`max_label_len` 需要根据实际标签的最大 token 数设置
5. **模板文件**：确保 `prompt.txt` 中的模板格式正确

## 配置说明

主要配置项在 `pet_config.py` 中：
- `prompt_file`: 模板文件路径，定义了任务的 prompt 结构
- `verbalizer`: Verbalizer 文件路径，定义了标签到 token 的映射关系
- `max_label_len`: 标签最大 token 长度
- `max_seq_len`: 最大序列长度（BERT 最大 512）

## 模板设计建议

1. **简洁明了**：模板应该简洁，避免过于复杂
2. **语义清晰**：模板应该让模型能够理解任务意图
3. **位置合理**：MASK 的位置应该符合语言习惯
4. **示例模板**：
   - 分类任务：`"这是一条{MASK}评论：{textA}。"`
   - 情感分析：`"这句话的情感是{MASK}：{textA}"`
   - 关系抽取：`"{textA}和{textB}的关系是{MASK}"`

## 与 P-Tuning 的对比

| 特性 | PET | P-Tuning |
|------|-----|---------|
| Prompt 类型 | 硬模板（包含具体文本） | 连续 embeddings |
| 模板定义 | 需要人工设计模板 | 自动学习 |
| 可解释性 | 高（模板可读） | 低（连续向量） |
| 灵活性 | 中等（需要设计模板） | 高（自动学习） |
| 适用场景 | 需要可解释性的场景 | 需要自动学习的场景 |
