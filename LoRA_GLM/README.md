# LoRA_GLM 项目实现流程

## 项目概述

LoRA_GLM 项目使用 **LoRA (Low-Rank Adaptation)** 技术对 GLM 模型进行高效微调。LoRA 是一种参数高效的微调方法，通过低秩矩阵分解来减少可训练参数数量，从而降低显存占用和训练时间。

## 技术原理

### LoRA 核心思想
- **低秩分解**：将权重更新矩阵分解为两个低秩矩阵的乘积（A × B），其中 A 的维度为 `[d, r]`，B 的维度为 `[r, d]`，r << d
- **参数效率**：只训练低秩矩阵，冻结原始模型参数，大幅减少可训练参数
- **可合并**：训练完成后可以将 LoRA 权重合并回原始模型

## 项目结构

```
LoRA_GLM/
├── train.py                 # 训练脚本
├── inference.py             # 推理脚本
├── glm_config.py            # 配置文件
├── data_handle/
│   ├── data_loader.py       # 数据加载器
│   └── data_preprocess.py   # 数据预处理
├── utils/
│   └── common_utils.py      # 工具函数
└── data/
    ├── mixed_train_dataset.jsonl
    └── mixed_dev_dataset.jsonl
```

## 实现流程

### 1. 数据准备

#### 1.1 数据格式
数据采用 JSONL 格式，每行一个样本：
```json
{"context": "年基准利率4.35%。从实际看...", "target": "2017年银行贷款基准利率"}
```

#### 1.2 数据预处理 (`data_preprocess.py`)
- **输入格式**：`{"context": "...", "target": "..."}`
- **处理流程**：
  1. 将 `context` 编码为 token IDs（作为 prompt）
  2. 将 `target` 编码为 token IDs（作为目标答案）
  3. 构建输入序列：`[context] + [gMASK] + [sop] + [target] + [eos]`(其中对context和target的最大长度做处理，多余的pad填充)
  4. 构建标签序列：`[-100] * context_length + [target_ids]`（只计算 target 部分的损失）
  5. 进行 padding 到最大长度


### 2. 模型配置 (`glm_config.py`)

主要配置参数：
- `pre_model`: 预训练模型路径（ChatGLM2-6B）
- `use_lora`: 是否使用 LoRA（True）
- `lora_rank`: 低秩矩阵的秩（默认 8）
- `batch_size`: 批次大小
- `learning_rate`: 学习率（3e-5）
- `epochs`: 训练轮数
- `max_source_seq_len`: 源序列最大长度
- `max_target_seq_len`: 目标序列最大长度
- `weight_decay`: 权重衰减
- `warmup_ratio`: 预热比例
- `logging_steps`: 日志记录间隔
- `save_freq`: 保存间隔

- 等其他参数。
### 3. 模型初始化 (`train.py`)

#### 3.1 加载预训练模型
```python
model = AutoModel.from_pretrained(
    pc.pre_model,
    config=config,
    trust_remote_code=True
)
```

#### 3.2 模型优化设置
- **梯度检查点**：启用 `gradient_checkpointing_enable()` 减少显存占用
- **输入梯度**：启用 `enable_input_require_grads()` 允许输入梯度计算
- **缓存禁用**：设置 `use_cache = False` 减少内存
- **半精度训练**：使用 `model.half()` 启用半精度训练以节省显存

#### 3.3 LoRA 配置
```python
peft_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    inference_mode=False,
    r=pc.lora_rank,          # 低秩矩阵维度
    lora_alpha=32,           # 缩放系数
    lora_dropout=0.1,
)
model = peft.get_peft_model(model, peft_config)
```

#### 3.4 输出层处理
```python
model.lm_head = CastOutputToFloat(model.lm_head)  # 确保输出为 float32
```

### 4. 训练流程

#### 4.1 优化器配置
- 使用 **AdamW** 优化器
- 对 `bias` 和 `LayerNorm.weight` 不应用权重衰减
- 其他参数应用权重衰减

#### 4.2 学习率调度
- 使用线性学习率调度器
- 设置预热步数（warmup steps）
- 根据总训练步数动态调整学习率

#### 4.3 训练循环
```python
for epoch in range(1, pc.epochs + 1):
    for batch in train_dataloader:
        # 混合精度训练（如果使用 LoRA）
        with autocast():
            loss = model(
                input_ids=batch['input_ids'].to(device),
                labels=batch['labels'].to(device)
            ).loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
```

#### 4.4 模型评估
- 定期在验证集上评估模型
- 计算验证损失
- 保存最佳模型（验证损失最低）

#### 4.5 模型保存
```python
# LoRA 模型需要合并后保存
merged_model = copy.deepcopy(model)
merged_model = merged_model.merge_and_unload()  # 合并 LoRA 权重
merged_model.save_pretrained(cur_save_dir)
```

### 5. 推理流程 (`inference.py`)

#### 5.1 模型加载
```python
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = model.half().to(device)  # 使用半精度推理
```

#### 5.2 输入构建
```python
input_text = f'Instruction: {instruction}\n'
if sentence:
    input_text += f'Input: {sentence}\n'
input_text += 'Answer: '
```

#### 5.3 生成输出
```python
out = model.generate(
    input_ids=batch["input_ids"].to(device),
    max_new_tokens=max_new_tokens,
    temperature=0
)
answer = tokenizer.decode(out[0]).split('Answer: ')[-1]
```

## 关键技术点

### 1. LoRA 的优势
- **参数效率**：只训练少量参数（通常 < 1% 的原始参数）
- **显存友好**：大幅降低显存占用
- **易于部署**：可以合并权重，无需额外推理开销

### 2. 混合精度训练
- 使用 `autocast()` 进行混合精度训练
- 在保持数值精度的同时提高训练速度

### 3. 梯度检查点
- 牺牲部分计算时间换取显存节省
- 适合显存受限的场景

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

1. **数据格式**：确保数据为 JSONL 格式，包含 `context` 和 `target` 字段
2. **显存管理**：如果显存不足，可以减小 `batch_size` 或 `lora_rank`
3. **模型路径**：确保预训练模型路径正确
4. **LoRA 合并**：推理时可以选择合并 LoRA 权重或直接使用 LoRA 适配器

## 配置说明

主要配置项在 `glm_config.py` 中：
- `use_lora`: 控制是否使用 LoRA（True/False）
- `lora_rank`: LoRA 的秩，影响参数量和效果（建议 4-16）
- `use_ptuning`: 是否使用 P-Tuning（本项目为 False）
- `pre_seq_len`: P-Tuning 的前缀长度（仅当 use_ptuning=True 时有效）
