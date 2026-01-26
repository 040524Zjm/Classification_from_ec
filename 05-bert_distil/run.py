import numpy as np
import torch
from fontTools.config import Config

from train_eval import train_kd, train
from importlib import import_module
import argparse
from utils.utils import build_dataset, build_dataset_CNN, build_iterator

# 解析命令行参数
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--task', type=str, default='train_kd', help='choose a task: train_bert, or train_kd')
args = parser.parse_args()

"""
source activate NLP_demo -> 可运行的环境
cd /Users/zjm/PycharmProjects/News666/05-bert_distil

"""

if __name__ == '__main__':
    # 任务类型来分配模型
    if args.task == 'train_bert':
        model_name = 'bert'
        x = import_module('models.' + model_name)
        config = x.Config()
        # 初始化
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        # 构建数据集
        print('Loading data for Bert Model...')
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        # 模型实例化与训练
        model = x.Model(config).to(config.device) # 实例化模型且把模型移动到设备
        train(config, model, train_iter, dev_iter, test_iter)

    if args.task == 'train_kd':
        # 加载bert模型
        model_name = 'bert'
        bert_module = import_module('models.' + model_name)
        bert_config = bert_module.Config()
        # 加载cnn
        model_name = 'textCNN'
        cnn_module = import_module('models.' + model_name)
        cnn_config = cnn_module.Config()
        # 初始化
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        # 构建bert数据集，只需要训练结果作为软目标
        bert_train_data, _, _ = build_dataset(bert_config)
        bert_train_iter = build_iterator(bert_train_data, bert_config)
        # 构建cnn数据集
        vocab, cnn_train_dada, cnn_dev_data, cnn_test_data = build_dataset_CNN(cnn_config)
        cnn_train_iter = build_iterator(cnn_train_dada, cnn_config)
        cnn_dev_iter = build_iterator(cnn_dev_data, cnn_config)
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)
        # 加载训练好的teacher模型
        bert_model = bert_module.Model(bert_config).to(bert_config.device)
        # 加载student
        cnn_model = cnn_module.Model(cnn_config).to(cnn_config.device)
        print("Teacher and student models loaded, start training")
        train_kd(cnn_config, bert_model, cnn_model, bert_train_iter, cnn_train_iter, cnn_dev_iter, cnn_test_iter)


"""
Test Loss:  0.63,  Test Acc: 90.43%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.9208    0.8840    0.9020      1000
       realty     0.9225    0.9280    0.9252      1000
       stocks     0.8498    0.8490    0.8494      1000
    education     0.9555    0.9450    0.9502      1000
      science     0.8600    0.8600    0.8600      1000
      society     0.8683    0.9100    0.8887      1000
     politics     0.8858    0.8840    0.8849      1000
       sports     0.9494    0.9560    0.9527      1000
         game     0.9317    0.9010    0.9161      1000
entertainment     0.9025    0.9260    0.9141      1000

     accuracy                         0.9043     10000
    macro avg     0.9046    0.9043    0.9043     10000
 weighted avg     0.9046    0.9043    0.9043     10000

Confusion Matrix...
[[884  11  56   1   9  14  12   6   4   3]
 [ 11 928  14   2   3  18   4   5   1  14]
 [ 46  22 849   0  34   5  37   3   2   2]
 [  1   2   2 945   5  16  14   1   1  13]
 [  1   6  37   4 860  19  19   6  27  21]
 [  4  16   1  19  11 910  17   2   5  15]
 [  8  12  28   9  16  34 884   2   1   6]
 [  2   4   3   1   3  11   4 956   3  13]
 [  0   0   7   4  51   6   5  13 901  13]
 [  3   5   2   4   8  15   2  13  22 926]]
Time usage: 0:00:02

"""









