# 新闻文本分类项目

一个全面的中文新闻文本分类项目，实现了从传统机器学习到深度学习、从预训练模型到LoRA、P-Tuning、PET的多种分类方法。

## 📋 项目简介

本项目实现了多种文本分类方法，用于对中文新闻进行10分类任务。项目涵盖了传统机器学习、深度学习、模型优化以及大模型微调等多种技术路线，为新闻分类任务提供了完整的解决方案。

### 分类类别

项目包含以下10个新闻类别：
- finance（财经）
- realty（房产）
- stocks（股票）
- education（教育）
- science（科技）
- society（社会）
- politics（时政）
- sports（体育）
- game（游戏）
- entertainment（娱乐）

## 🗂️ 项目结构

```
News666/
├── 01-data/              # 数据文件
│   ├── train.txt         # 训练集
│   ├── dev.txt           # 验证集
│   ├── test.txt          # 测试集
│   └── class.txt         # 类别文件
│
├── 02-rf/                # 随机森林方法
│   ├── rf.py             # 随机森林分类实现
│   ├── news.py           # 数据预处理
│   └── dev.csv           # 处理后的数据
│
├── 03-fasttest/          # FastText方法
│   ├── 01-preprocess.py  # 数据预处理
│   ├── 02-fasttext.py    # FastText训练
│   ├── 03-fasttext_2.py  # FastText变体
│   ├── 05-fasttext_3.py  # FastText优化
│   ├── server.py         # Flask服务部署
│   └── client.py         # 客户端测试
│
├── 04-bert/              # BERT方法
│   ├── run.py            # 主运行脚本
│   ├── train_eval.py     # 训练和评估
│   ├── predict.py        # 预测脚本
│   ├── run_quantization.py  # 模型量化
│   ├── server.py         # 服务部署
│   ├── model/
│   │   └── bert.py       # BERT模型定义
│   ├── utils/
│   │   └── utils.py      # 工具函数
│   ├── data/
│   │   └── bert_pretrain/  # BERT预训练模型
│   └── save_model/       # 模型保存目录
│
├── 05-bert_distil/       # BERT蒸馏方法
│   ├── run.py            # 主运行脚本
│   ├── train_eval.py     # 训练和评估
│   ├── models/
│   │   ├── bert.py       # BERT模型
│   │   └── textCNN.py    # TextCNN模型
│   └── utils/
│       └── utils.py      # 工具函数
│
├── 06-prune/             # 模型剪枝
│   └── 01-demo.py        # 剪枝示例
│
├── LoRA_GLM/             # LoRA微调GLM
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 推理脚本
│   ├── glm_config.py     # 配置文件
│   ├── data_handle/      # 数据处理
│   └── checkpoints/      # 模型检查点
│
├── P-Tuning/             # P-Tuning方法
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 推理脚本
│   ├── ptune_config.py   # 配置文件
│   ├── data_handle/      # 数据处理
│   ├── utils/            # 工具函数
│   └── checkpoints/      # 模型检查点
│
├── PET/                  # Pattern-Exploiting Training
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 推理脚本
│   ├── pet_config.py     # 配置文件
│   ├── data_handle/      # 数据处理
│   └── utils/            # 工具函数
│
├── requirements0.txt     # 基础依赖
└── requirements1.txt     # 完整依赖
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.13+
- CUDA（可选，用于GPU加速）

### 安装依赖

#### 基础环境（推荐）
```bash
pip install -r requirements0.txt
```

#### 完整环境（包含所有功能）
```bash
pip install -r requirements1.txt
```

### 数据准备

1. 将训练数据放置在 `01-data/` 目录下
2. 数据格式：每行一条数据，格式为 `文本内容\t标签`
3. 确保 `class.txt` 包含所有类别名称

## 📖 使用方法

### 1. 随机森林（RF）

```bash
cd 02-rf
python rf.py
```

**特点：**
- 使用TF-IDF特征提取
- 准确率约 81.87%
- 训练速度快，适合快速原型

### 2. FastText

```bash
cd 03-fasttest
python 02-fasttext.py
```

**特点：**
- 基于词向量的快速分类
- 准确率约 84.41%
- 支持中文分词
- 提供Flask服务接口

**启动服务：**
```bash
python server.py
# 服务运行在 http://0.0.0.0:3001
```

### 3. BERT

```bash
cd 04-bert
python run.py
```

**特点：**
- 基于预训练BERT模型
- 准确率约 93.14%
- 支持模型量化优化
- 提供完整的训练和评估流程

**配置说明：**
- 修改 `model/bert.py` 中的 `Config` 类来调整超参数
- 预训练模型路径在配置中指定

### 4. BERT蒸馏

```bash
cd 05-bert_distil
python run.py
```

**特点：**
- 使用知识蒸馏技术
- 模型更小，推理更快
- 保持较高准确率

### 5. P-Tuning

```bash
cd P-Tuning
python train.py
```

**特点：**
- 基于提示学习的参数高效微调
- 只训练少量参数
- 适合资源受限场景

**配置：**
- 修改 `ptune_config.py` 中的参数
- 支持P-Tuning和P-Tuning v2

### 6. PET（Pattern-Exploiting Training）

```bash
cd PET
python train.py
```

**特点：**
- 基于模板的提示学习
- 使用verbalizer进行标签映射
- 参数高效

### 7. LoRA微调GLM

```bash
cd LoRA_GLM
python train.py
```

**特点：**
- 使用LoRA技术微调GLM大模型
- 低秩矩阵分解，参数高效
- 支持ChatGLM2等模型

**配置：**
- 修改 `glm_config.py` 中的模型路径和参数
- 支持LoRA和P-Tuning两种模式

## 📊 模型性能对比

| 方法 | 准确率 | 特点 |
|------|--------|------|
| 随机森林 | ~81.87% | 快速，轻量级 |
| FastText | ~84.41% | 快速，适合短文本 |
| BERT | ~93.14% | 高准确率，资源消耗大 |
| BERT蒸馏 | - | 模型小，推理快 |
| P-Tuning | - | 参数高效 |
| PET | - | 提示学习 |
| LoRA-GLM | - | 大模型微调 |

*注：部分方法的准确率需要根据实际训练结果填写*

## 🔧 配置说明

### BERT配置
- `learning_rate`: 学习率（默认：5e-5）
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `max_seq_len`: 最大序列长度（默认：512）

### P-Tuning配置
- `p_embedding_num`: 提示嵌入数量
- `max_label_len`: 最大标签长度
- `warmup_ratio`: 学习率预热比例

### LoRA配置
- `lora_rank`: LoRA秩（默认：8）
- `use_lora`: 是否使用LoRA
- `use_ptuning`: 是否使用P-Tuning

## 🌐 服务部署

### FastText服务
```bash
cd 03-fasttest
python server.py
```

### BERT服务
```bash
cd 04-bert
python server.py
```

服务接口：
- **URL**: `http://localhost:3001/v1/main_server`
- **Method**: POST
- **参数**: `text` (文本内容)
- **返回**: 分类结果

## 📝 数据格式

### 训练数据格式
```
文本内容\t标签
```

示例：
```
今日股市大涨，投资者信心增强\tstocks
教育部发布新政策\teducation
```

### 类别文件格式
每行一个类别名称：
```
finance
realty
stocks
...
```

## 🛠️ 开发说明

### 添加新方法
1. 在项目根目录创建新的文件夹
2. 实现训练、评估和推理脚本
3. 更新本README文档

### 代码规范
- 使用Python 3.7+语法
- 遵循PEP 8代码规范
- 添加必要的注释和文档字符串

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题或建议，请通过Issue联系。

---

**注意：** 
- 使用BERT等预训练模型时，请确保已下载相应的预训练权重
- 部分路径配置为绝对路径，使用时请根据实际情况修改
- GPU加速需要安装CUDA和对应的PyTorch版本
