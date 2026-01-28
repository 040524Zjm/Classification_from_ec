### 项目说明：03-fasttest（FastText 中文新闻分类）

本项目基于 Facebook 的 FastText 库完成中文新闻分类，整体流程为：**原始 TSV 数据 → 按字/按词构造 FastText 训练语料 → FastText 监督训练 → 测试集评估与模型保存**。  
本目录中主要脚本配合 `../01-data` 中的原始数据使用。

---

### 一、数据准备与预处理

#### 1. 类别映射与基础设置

- `../01-data/class.txt` 中保存类别名称，每行一个类别。
- 若需要将标签 ID 转成类别名，再拼接成 FastText 所需的 `__label__类别名 文本` 形式，会使用到：
  - 将 `class.txt` 读入为 `id2name` 或 `id2label` 字典。

#### 2. 按“字”为单位生成 FastText 数据（`01-preprocess.py`）

- 核心思路：
  - 从 `../01-data/train.txt / dev.txt / test.txt` 中逐行读取样本，每行格式为：`文本\t标签ID`。
  - 使用 `id2name` 字典将标签 ID 转换为类别名称。
  - 构造 FastText 所需标签格式：`__label__<类别名>`。
  - 将原始句子按“字”切分：`new_text = ' '.join(list(text))`。
  - 拼接为：`__label__类别名 + ' ' + new_text`。
  - 分别写入：
    - `train_fast.txt`（训练集）
    - `dev_fast.txt`（验证集）
    - `test_fast.txt`（测试集）

示例代码片段（节选自 `01-preprocess.py`，以 test 为例）：

```python
id2name = {}
idx = 0
with open('../01-data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        class_name = line.strip('\n').strip()
        id2name[idx] = class_name
        idx += 1

train_data = []
with open('../01-data/test.txt', 'r', encoding='utf-8') as f2:
    for line in f2.readlines():
        line = line.strip('\n').strip()
        text, label = line.split('\t')
        label_id = int(label)
        label_name = id2name[label_id]
        new_label = '__label__' + label_name
        new_text = ' '.join(list(text))
        new_data = new_label + ' ' + new_text
        train_data.append(new_data)

with open('test_fast.txt', 'w', encoding='utf-8') as f3:
    for data in train_data:
        f3.write(data + '\n')
```

> 说明：`01-preprocess.py` 中针对 train/dev/test 的部分有注释控制，可按需要打开或注释来生成不同文件。

#### 3. 按“词”为单位生成 FastText 数据（`04-preprocess1.py`）

- 使用 `jieba` 进行中文分词：
  - 读取同样的 `train.txt / dev.txt / test.txt`。
  - 使用 `jieba.cut(sentence)` 将句子切成词序列。
  - 用空格拼接得到 `sent_char`，然后同样前面加上 `__label__类别名`。
  - 分别输出到：
    - `train_fast1.txt`
    - `dev_fast1.txt`
    - `test_fast1.txt`

示例代码片段（节选自 `04-preprocess1.py`）：

```python
import jieba

id2label = {}
idx = 0
with open('../01-data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        line = line.strip('\n').strip()
        id2label[idx] = line
        idx += 1

def cut_text(before_path, after_path):
    datas = []
    with open(before_path, 'r', encoding='utf-8') as bf:
        for line in bf.readlines():
            line = line.strip('\n').strip()
            sentence, label = line.split('\t')
            label_id = int(label)
            label_name = id2label[label_id]
            new_label = '__label__' + label_name
            sent_char = ' '.join(jieba.cut(sentence))
            new_sentence = new_label + ' ' + sent_char
            datas.append(new_sentence)
    with open(after_path, 'w', encoding='utf-8') as af:
        for data in datas:
            af.write(data + '\n')
    print('数据处理完成!')
```

> 通过对比 `train_fast*.txt` 与 `train_fast1*.txt` 的实验，可以对比“字级别”与“词级别”输入对 FastText 效果的影响。

---

### 二、FastText 模型训练与评估

#### 1. 简单监督训练（`02-fasttext.py`）

- 使用 `fasttext.train_supervised` 直接训练分类模型：
  - 训练输入：`train_fast.txt`（字级别方案）。
  - 测试输入：`test_fast.txt`。
  - 接口示例：
    - `model = fasttext.train_supervised('./train_fast.txt', 2)`  
      （第二个参数指定线程数等，具体请参考源码或 FastText 文档）。
  - 训练完成后：
    - 打印词表大小：`len(model.words)`。
    - 打印标签：`model.labels`。
    - 使用 `model.test('./test_fast.txt')` 在测试集上评估性能（返回样本数、准确率等）。

示例代码片段（节选自 `02-fasttext.py`）：

```python
import fasttext

model = fasttext.train_supervised('./train_fast.txt', 2)
print(f'词的数量：{len(model.words)}')
print(f'标签值:{model.labels}')

result = model.test('./test_fast.txt')
print(f'准确率:{result}')
```

#### 2. 自动调参 + 模型保存（`03-fasttext_2.py`）

- 使用 FastText 自动调参功能：
  - 训练输入：`train_fast.txt`。
  - 验证集：`dev_fast.txt`（通过 `autotuneValidationFile` 指定）。
  - 核心参数：
    - `autotuneDuration=6`：自动调参持续时间（秒）。
    - `wordNgrams=2`：使用 bi-gram。
    - `verbose=3`：输出详细训练日志。
- 训练完成后：
  - 调用 `model.test(test_data_path)` 在 `test_fast.txt` 上评估。
  - 使用当前时间戳拼接模型名，比如 `news_fasttext_*.bin`，并通过：
    - `model.save_model(model_save_path)` 保存二进制模型。

示例代码片段（节选自 `03-fasttext_2.py`）：

```python
import fasttext
import time

train_data_path = './train_fast.txt'
dev_data_path = './dev_fast.txt'
test_data_path = './test_fast.txt'

model = fasttext.train_supervised(
    input=train_data_path,
    autotuneValidationFile=dev_data_path,
    autotuneDuration=6,
    wordNgrams=2,
    verbose=3
)

result = model.test(test_data_path)
print(result)

time1 = time.time()
model_save_path = './news_fasttext_{}.bin'.format(time1)
model.save_model(model_save_path)
```

#### 3. 词级别输入下的自动调参（`05-fasttext_3.py`）

- 与 `03-fasttext_2.py` 类似，但使用的是**通过 `04-preprocess1.py` 生成的词级别数据**：
  - `train_fast1.txt`、`dev_fast1.txt`、`test_fast1.txt`。
  - 保持 `autotuneDuration`、`wordNgrams` 等参数设置，比较最终测试集结果。
  - 通过注释可见实验结论：字级与词级的差异并不大（在当前设置和硬件下）。

---

### 三、推荐运行顺序

在 `03-fasttest` 目录下：

```bash
cd 03-fasttest

# 1. 生成按“字”划分的 FastText 训练/验证/测试文件
python 01-preprocess.py   # 根据脚本中注释打开相应部分

# 2. 使用字级别数据进行训练、测试
python 02-fasttext.py     # 简单训练/测试
python 03-fasttext_2.py   # 自动调参 + 模型保存

# 3. 生成按“词”划分的 FastText 训练/验证/测试文件
python 04-preprocess1.py  # 根据 main 中的调用生成 *fast1.txt

# 4. 使用词级别数据进行训练、测试
python 05-fasttext_3.py
```

---

### 四、项目要点与可扩展方向

- **标签格式约定**
  - FastText 要求标签形如：`__label__<类名>`，注意与原始 `labelID` 的映射。

- **分词策略对比**
  - 本项目显式对比了“字级别输入”和“词级别输入”，便于理解粒度变化对性能的影响。

- **进一步改进**
  - 调整 `wordNgrams`、`lr`、`epoch` 等超参数。
  - 使用更长的自动调参时间 `autotuneDuration`，获取更优参数组合。
  - 结合子词（subword）特征，提高对 OOV 词的鲁棒性。

