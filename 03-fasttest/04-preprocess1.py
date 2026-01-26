import jieba
# 词分词
id2label = {}
idx = 0
with open('../01-data/class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        line = line.strip('\n').strip()
        id2label[idx] = line
        idx += 1
# print(id2label)

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
    # print(datas[:10])
    with open(after_path, 'w', encoding='utf-8') as af:
        for data in datas:
            af.write(data + '\n')
    print('数据处理完成!')






if __name__ == '__main__':
    # cut_text('../01-data/train.txt', 'train_fast1.txt')
    # cut_text('../01-data/dev.txt', 'dev_fast1.txt')
    # cut_text('../01-data/test.txt', 'test_fast1.txt')
    pass