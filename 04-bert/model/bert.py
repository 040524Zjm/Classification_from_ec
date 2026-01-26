import os
import torch
from torch.onnx.ops import attention
from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn


class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert" # 模型名称
        self.data_path = "/Users/zjm/PycharmProjects/News666/04-bert/data/data1/" #数据集的根路径
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path + "/class.txt").readlines()]  # 类别名单

        self.save_path = "/Users/zjm/PycharmProjects/News666/04-bert/save_model" #模型训练结果保存路径
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果

        self.save_path2 = "/Users/zjm/PycharmProjects/News666/04-bert/save_model1" # 量化模型存储结果路径
        if not os.path.exists(self.save_path2):
            os.mkdir(self.save_path2)
        self.save_path2 += "/" + self.model_name + "_quantized.pt"  # 量化模型存储结果

        # 模型训练+预测的时候, 放开下一行代码, 在GPU上运行.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu
        # 模型量化的时候, 放开下一行代码, 在CPU上运行.
        # self.device = 'cpu'

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 1  # epoch数2
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "/Users/zjm/PycharmProjects/News666/04-bert/data/bert_pretrain" # 预训练BERT模型的路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json') # BERT模型的配置
        self.hidden_size = 768 # BERT模型的隐藏层大小768


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.out = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self,x):
        _, pool = self.bert(x[0], attention_mask=x[2], return_dict=False)
        logits = self.out(pool)
        return logits

if __name__ == '__main__':
    config = Config()
    print(config.device)
    model = Model(config)
    # print(model)
    print(list(model.named_parameters()))


