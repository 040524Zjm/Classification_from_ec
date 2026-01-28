### 项目说明：04-bert（BERT 中文新闻分类）

本项目使用预训练 BERT 模型进行中文新闻分类，整体流程为：**构造数据集与迭代器 → 加载 BERT 配置与模型 → 训练或加载已训练权重 → 在测试集上评估**。  
该目录下的代码可实现完整的 BERT fine-tuning 以及推理评估。

---

### 一、目录结构与核心文件

- `data/bert_pretrain/`：预训练 BERT 模型相关文件
  - `bert_config.json`、`pytorch_model.bin`、`vocab.txt` 等。
- `data/data1/`：本任务数据
  - `train.txt`、`dev.txt`、`test.txt`：样本格式通常为 `文本\t标签ID`。
  - `class.txt`：类别名称列表。
  - `vocab.pkl`：构建好的词表 / 映射等（由 `utils` 相关脚本生成）。
- `model/bert.py`：BERT 分类模型定义
  - 内含 `Config`（配置类）与 `Model`（模型类），封装了预训练 BERT 的加载及分类头。
- `utils/utils.py`：
  - `build_dataset`：构建训练/验证/测试数据集。
  - `build_iterator`：基于 `DataLoader` 构建批次迭代器。
  - `get_time_dif`：时间统计工具。
- `train_eval.py`：
  - `train`：训练流程（包含优化器、验证与模型保存）。
  - `evaluate`：在验证或测试集上评估模型。
  - `test`：加载最优模型，在测试集上给出最终指标。
- `run.py`：
  - 入口脚本，协调配置加载、数据构建、模型初始化与训练 / 测试。
- `save_model/bert.pt`：
  - 已训练好的模型权重文件。

---

### 二、数据构建流程（`utils/utils.py`）

- **读取原始数据**
  - 从 `data/data1/train.txt`、`dev.txt`、`test.txt` 中读取样本。
  - 每一行包括文本和标签 ID，标签与 `class.txt` 中的类别名称相对应。

- **文本编码与标记化**
  - 使用 BERT 的 `vocab.txt` 和 tokenizer 对文本进行分词、转 ID。
  - 通常会：
    - 添加 `[CLS]` / `[SEP]` 标记。
    - 截断或 Padding 到 `Config.pad_size`。

- **数据格式**
  - `build_dataset(config)` 返回：
    - `train_data`、`test_data`、`dev_data`（列表或 Dataset 对象）。
  - `build_iterator(data, config)` 将上述数据转换为可迭代的 batch：
    - 每个 batch 通常包含 `input_ids`、`attention_mask`、`token_type_ids`、`labels` 等张量。

---

### 三、训练与验证流程（`train_eval.py`）

#### 1. 训练函数 `train(model, train_iter, dev_iter, config)`

- **损失与优化器**
  - 使用 `nn.CrossEntropyLoss` 作为分类损失。
  - 使用 `AdamW` 优化器，并对带权重衰减和不带权重衰减的参数做分组：
    - `no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']`。
  - 学习率、epoch 等超参数从 `Config` 中读取。

- **训练循环**
  - 遍历 `config.num_epochs` 个 epoch。
  - 在每个 batch：
    - 前向计算：`output = model(trains)`。
    - 计算损失 `loss`，反向传播 `loss.backward()`，更新参数 `optimizer.step()`。
  - 每隔若干 batch（示例中为 100）进行一次验证：
    - 计算当前 batch 的训练准确率。
    - 调用 `evaluate` 在验证集上计算 `dev_acc` 与 `dev_loss`。
    - 若 `dev_loss` 优于历史最优，则保存模型参数至 `config.save_path`。

#### 2. 验证与测试函数

- `evaluate(config, model, data_iter, test=False)`：
  - 切换到 `model.eval()` 模式。
  - 无梯度地遍历数据集，累积损失并收集预测结果。
  - 计算准确率：
    - 若 `test=False`：返回 `acc` 与平均 `loss`。
    - 若 `test=True`：额外返回分类报告和混淆矩阵。

- `test(model, test_iter, config)`：
  - 加载最优权重：`model.load_state_dict(torch.load(config.save_path, ...))`。
  - 调用 `evaluate(..., test=True)`：
    - 打印最终测试集上的 `loss`、`acc`。
    - 打印每类的 `precision/recall/f1`。
    - 打印混淆矩阵与耗时。

---

### 四、入口脚本运行流程（`run.py`）

- **模型与配置加载**
  - 通过 `import_module('model.bert')` 动态导入 BERT 模型。
  - 实例化配置：`config = X.Config()`。
  - 设置随机种子（`numpy`/`torch`/`cuda`），保证结果可复现。

- **构建数据迭代器**
  - `train_data, test_data, dev_data = build_dataset(config)`。
  - `train_iter = build_iterator(train_data, config)` 等。

- **模型实例化与训练 / 测试**
  - `model = X.Model(config)` 实例化 BERT 分类模型。
  - 两种典型使用方式：
    - **训练模式**：调用 `train(model, train_iter, dev_iter, config)`（目前示例中被注释，可按需放开）。
    - **测试模式**：从 `save_model/bert.pt` 加载已训练参数后，在测试集上调用 `test(...)`。

示例运行命令：

```bash
cd 04-bert
python run.py
```

> 如需重新训练，可在 `run.py` 中取消 `train(...)` 调用的注释，并确保 `config.save_path` 指向希望保存的模型文件路径。

---

### 五、可扩展与调优方向

- 调整 `Config` 中的超参数：学习率、batch size、最大序列长度、epoch 数等。
- 尝试冻结部分 BERT 层，仅训练顶部分类层，以降低训练资源消耗。
- 在 `train_eval.py` 中加入学习率调度器（如 `Warmup` + `LinearDecay`）。
- 增加早停策略（Early Stopping），在验证集指标不再提升时提前停止训练。

