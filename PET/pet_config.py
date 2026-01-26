# coding:utf-8
import torch
import sys
# print(sys.path)

class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.pre_model = r'/Users/zjm/PycharmProjects/News666/04-bert/data/bert_pretrain'
        self.train_path = r'/Users/zjm/PycharmProjects/News666/PET/data/train.txt'
        self.dev_path = r'/Users/zjm/PycharmProjects/News666/PET/data/dev.txt'
        self.prompt_file = r'/Users/zjm/PycharmProjects/News666/PET/data/prompt.txt'
        self.verbalizer = r'/Users/zjm/PycharmProjects/News666/PET/data/verbalizer.txt'
        self.max_seq_len = 512
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.max_label_len = 2
        self.epochs = 50
        self.logging_steps = 1 # 每多少步记录一次
        self.valid_steps = 10 # 每多少步验证一次
        self.save_dir = r'./checkpoints'

if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.device)

