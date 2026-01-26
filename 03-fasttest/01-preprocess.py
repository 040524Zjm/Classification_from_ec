# id2name
id2name = {}
id = 0
with open('../01-data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        class_name = line.strip('\n').strip()
        id2name[id] = class_name
        id += 1
        # 字分词

# print(id2name)
# 用来存储训练集数据
train_data = []
# 打开数据集文件，进行处理
# with open('../01-data/train.txt', 'r', encoding='utf-8') as f2:
#     # 获取每一行数据
#     for line in f2.readlines():
#         # 获取文本和对应的标签信息
#         line = line.strip('\n').strip()
#         # 1: 首先处理标签部分：获取标签id,并获取标签名称
#         text, label = line.split('\t')
#         # 构建fasttext需要的训练集数据
#         label_id = int(label)
#         label_name = id2name[label_id]
#         new_label = '__label__' + label_name
#         # 2: 然后处理文本部分, 可以按字划分, 也可以按词划分
#         new_text = ' '.join(list(text))
#         # 3: 将文本和标签组合成fasttext规定的格式
#         new_data = new_label + ' ' + new_text
#         # 4: 将数据添加到list中
#         train_data.append(new_data)
# # print(train_data[:5])
# with open('train_fast.txt', 'w', encoding='utf-8') as f3:
#     for data in train_data:
#         f3.write(data + '\n')
# print('数据处理完成!')


# with open('../01-data/dev.txt', 'r', encoding='utf-8') as f2:
#     # 获取每一行数据
#     for line in f2.readlines():
#         # 获取文本和对应的标签信息
#         line = line.strip('\n').strip()
#         # 1: 首先处理标签部分：获取标签id,并获取标签名称
#         text, label = line.split('\t')
#         # 构建fasttext需要的训练集数据
#         label_id = int(label)
#         label_name = id2name[label_id]
#         new_label = '__label__' + label_name
#         # 2: 然后处理文本部分, 可以按字划分, 也可以按词划分
#         new_text = ' '.join(list(text))
#         # 3: 将文本和标签组合成fasttext规定的格式
#         new_data = new_label + ' ' + new_text
#         # 4: 将数据添加到list中
#         train_data.append(new_data)
# # print(train_data[:5])
# with open('dev_fast.txt', 'w', encoding='utf-8') as f3:
#     for data in train_data:
#         f3.write(data + '\n')
# print('数据处理完成!')

with open('../01-data/test.txt', 'r', encoding='utf-8') as f2:
    # 获取每一行数据
    for line in f2.readlines():
        # 获取文本和对应的标签信息
        line = line.strip('\n').strip()
        # 1: 首先处理标签部分：获取标签id,并获取标签名称
        text, label = line.split('\t')
        # 构建fasttext需要的训练集数据
        label_id = int(label)
        label_name = id2name[label_id]
        new_label = '__label__' + label_name
        # 2: 然后处理文本部分, 可以按字划分, 也可以按词划分
        new_text = ' '.join(list(text))
        # 3: 将文本和标签组合成fasttext规定的格式
        new_data = new_label + ' ' + new_text
        # 4: 将数据添加到list中
        train_data.append(new_data)
# print(train_data[:5])
with open('test_fast.txt', 'w', encoding='utf-8') as f3:
    for data in train_data:
        f3.write(data + '\n')
print('数据处理完成!')







