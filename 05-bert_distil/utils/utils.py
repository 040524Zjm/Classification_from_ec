from tqdm import tqdm
from transformers.models.deprecated.realm.tokenization_realm import load_vocab
import os
import pickle as pkl
from datetime import timedelta
import time
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 目录问题
from models.textCNN import Config, Model



UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'
MAX_VOCAB_SIZE = 10000
# 词表
def build_vocab(file_path, tokenizer, max_size, min_freq):
    """

    :param file_path:文件路径
    :param tokenizer:分词函数
    :param max_size:词表最大
    :param min_freq:词汇表中词语最小出现频率
    :return: vocab_dic:字典，词汇映射到索引
    """
    vocab_dic = {} # 键单词，值单词出现次数
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split('\t')[0]
            # 分词器分词
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1 # get()安全写法，如果word不在字典中，则返回0，而如果有数字，那就继续加1
        # 根据词汇频率对词汇表排序，选择频率较高的单词
        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1] >= min_freq], # dic.item() -> (key, value)  _ 是元组
            key=lambda x: x[1],
            reverse=True
        )[:max_size] # 排序后取到max_size范围内的单词
        # 选定单词构建为字典，键单词，值单词索引
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)} # enumerate ->(索引，元素)迭代一个个出，默认0开始
        # 特殊符号
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1, CLS: len(vocab_dic) + 2})
    return vocab_dic


# 针对textCNN
def build_dataset_CNN(config):
    # 字符级分词器
    tokenizer = lambda x: [y for y in x]
    # 词汇表文件
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f'Vocab size: {len(vocab)}')

    # 加载数据集的辅助函数
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)

                # 填充截断
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                # 词转换为id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK))) # 双重兜底，如果不存在word用已知UNK代替

                # 数据添加到contents列表
                contents.append((words_line, int(label), seq_len)) # wordline是id序列，
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test

def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = ['[cls]'] + token
                seq_len = len(token)
                mask = []

                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += [0] * (pad_size - len(token))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return train, dev, test

class DatasetIterator(object):
    def __init__(self, batches, batch_size, device, model_name):
        self.batches = batches
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device
        self.n_batches = len(batches) // batch_size
        self.residue = False # 记录 batch 是否使用完全
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0 # 当前批次索引

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # 如果是Bert
        if self.model_name == 'bert':
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        # textCNN
        if self.model_name == 'textCNN':
            return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches: # 无残留且索引等于总批次数
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration # raise StopIteration 停止迭代
        else:
            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device, config.model_name)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif))) # 整数秒






if __name__ == '__main__':
    config = Config()
    vocab, train, dev, test = build_dataset_CNN(config)
    # print(list(vocab.items())[-10:-1])
    train_iter = build_iterator(train, config)
    for data in train_iter:
        print(data)
        break




