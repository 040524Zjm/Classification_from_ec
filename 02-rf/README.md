### 项目说明：02-rf（TF-IDF + 随机森林文本分类）

本项目基于传统机器学习方法完成新闻文本分类，核心流程为：**原始 TSV 数据 → 文本分词与特征工程 → TF‑IDF 向量化 → 随机森林训练与评估**。

---

### 一、数据准备与预处理

- **数据来源**
  - 使用 `../01-data/train.txt` 作为原始训练数据，格式为：`句子 \t 标签ID`。
  - 类别名称保存在 `../01-data/class.txt`。

- **分词与字段构造（`news.py`）**
  - 使用 `pandas` 读取 `train.txt`：
    - 列名设为：`sentences`（文本）、`labels`（标签ID）。
  - 统计标签分布、句长均值和方差等基础统计信息。
  - 使用 `jieba` 对句子进行分词：
    - 通过 `cut_sentence` 函数切分为词列表。
    - 将分词结果拼接为以空格分隔的字符串，保存到新字段 `words`。
    - 代码中对 `words` 字段进行了长度截断（示例中使用了 `[:30]`），用于控制特征长度。
  - 将处理好的数据保存为 `dev.csv`，其中包含至少两列：
    - `words`：分好词的文本字符串。
    - `labels`：对应的类别 ID。

---

### 二、特征工程与模型训练（`rf.py`）

- **读取预处理后的数据**
  - 使用 `pandas.read_csv('dev.csv')` 读取 `news.py` 生成的文件。
  - 提取：
    - `data = df['words'].values` 作为文本输入。
    - `label = df['labels']` 作为监督标签。

- **TF‑IDF 特征提取**
  - 使用 `sklearn.feature_extraction.text.TfidfVectorizer`：
    - `transform = TfidfVectorizer()`
    - `feature_vector = transform.fit_transform(data)`
  - 输出为稀疏矩阵，形状为 `[样本数, 词表大小]`。

- **数据集划分**
  - 使用 `train_test_split` 按比例划分训练集与测试集：
    - `test_size=0.2`（20% 作为测试集）。
    - `random_state=22` 保证结果可复现。

- **模型构建与训练**
  - 使用 `RandomForestClassifier` 作为分类模型：
    - `rf = RandomForestClassifier()`
  - 在训练集上拟合：
    - `rf.fit(x_train, y_train)`。

- **模型预测与评估**
  - 在测试集上预测：`y_predict = rf.predict(x_test)`。
  - 使用多种指标评估模型：
    - `accuracy_score(y_test, y_predict)`
    - （代码中预留了 `precision_score / recall_score / f1_score`，可按需打开）。

---

### 三、运行步骤示例

1. **确保已完成数据预处理**
   - 运行 `news.py`，生成 `dev.csv`：
   - 若脚本中有注释掉的部分（如分词方式或截断长度），可根据需要调整后再运行。

2. **训练并评估随机森林模型**
   - 在 `02-rf` 目录下执行：

```bash
cd 02-rf
python news.py   # 生成 dev.csv（如有需要）
python rf.py     # 训练随机森林并输出准确率等指标
```

---

### 四、可扩展与改进方向

- **特征工程**
  - 调整 `TfidfVectorizer` 参数（如 `ngram_range`、`min_df` 等）。
  - 改变文本截断方式（按字符长度 / 词数量）。

- **模型选择**
  - 替换为其他传统模型（如 `LinearSVC`、`LogisticRegression` 等）。
  - 调整随机森林的超参数（树数量、深度等）。

- **评估方式**
  - 补充宏平均 / 微平均 F1 分数，绘制混淆矩阵等。

