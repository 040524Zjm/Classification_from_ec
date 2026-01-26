import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Config(object):
    def __init__(self):
        self.model_name = "textCNN"
        self.data_path = "/Users/zjm/PycharmProjects/News666/04-bert/data/data1/"
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path+"class.txt", encoding="utf-8").readlines()]
        self.vocab_path = self.data_path + "vocab.pkl"  # 词表
        self.save_path = "/Users/zjm/PycharmProjects/News666/05-bert_distil/save_model"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = 300  # 字向量维度
        self.filter_sizes = (2, 3, 4) # 卷积核的大小
        self.num_filters = 256 # 卷积核的数量


class Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab -1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        ) # Conv2d 参数：输入通道数，输出通道数(卷积核个数)，卷积核大小(2，3，4) 最终2，3，4各个卷积核的输出通道数都是256
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self,x,conv):
        x = F.relu(conv(x)).squeeze(3) # (128, 1, 32, 300) conv(x) -> (128, 256, 高度卷积后2、3、4都不同, 300宽度变1)rulu后不变 去掉1的维度。(128,256,height)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # (128,256,height) -> 池化之后，高度也没了 (128,256)
        return x

    def forward(self,x):
        # (x, seq_len), y -> x
        out = self.embedding(x[0]) # (128, 32, 300) -> (batch, pad_size, embed)
        out = out.unsqueeze(1) # (128, 32, 300) -> (128, 1, 32, 300) # 128批、1通道、32长度、300维度
        # 卷积和池化，再拼接
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs],1) # 拼接1维度 (128,256) 拼接总共3个conv高度 -> (128,768)
        # 池化且拼接后才能dropout
        out = self.dropout(out)
        # 全连接层输出
        out = self.fc(out) # (128,768) -> (128,2)
        return out




if __name__ == '__main__':
    config = Config()

