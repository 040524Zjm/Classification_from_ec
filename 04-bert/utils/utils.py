import torch
from tqdm import tqdm
import time
from datetime import timedelta

# 保证直接运行该脚本时也能找到同级的 model 包
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.bert import Config

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
    train, dev, test = build_dataset(config)
    # print(dev)
    iter = build_iterator(dev, config)
    for data in iter:
        print(data) # (x, seq_len, mask), y
        break
