# 主函数
from importlib import import_module

import torch

from utils.utils import build_dataset, build_iterator, get_time_dif
from train_eval import train,test
import numpy as np

if __name__ == '__main__':
    model_name = 'bert'
    if model_name == 'bert':
        X = import_module('model.bert')
        config = X.Config()
        print(f'device:{config.device}')
        # 设置随机种子
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        # 数据集获取
        train_data, test_data, dev_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        test_iter = build_iterator(test_data, config)
        dev_iter = build_iterator(dev_data, config)
        # 模型实例化
        model = X.Model(config)
        # 模型训练
        # train(model, train_iter, dev_iter, config)
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))
        # 模型验证
        test(model, test_iter, config)
        """
        /Users/zjm/miniconda3/envs/NLP_demo/bin/python /Users/zjm/PycharmProjects/News666/04-bert/run.py 
cpu
180000it [00:11, 15802.62it/s]
10000it [00:00, 16778.07it/s]
10000it [00:00, 16700.60it/s]
Test loss:  0.21 Test acc: 93.14%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.8986    0.9220    0.9102      1000
       realty     0.9522    0.9170    0.9343      1000
       stocks     0.8924    0.8540    0.8728      1000
    education     0.9670    0.9670    0.9670      1000
      science     0.8687    0.9000    0.8841      1000
      society     0.9402    0.9270    0.9335      1000
     politics     0.9134    0.9490    0.9308      1000
       sports     0.9919    0.9750    0.9834      1000
         game     0.9419    0.9410    0.9415      1000
entertainment     0.9515    0.9620    0.9567      1000

     accuracy                         0.9314     10000
    macro avg     0.9318    0.9314    0.9314     10000
 weighted avg     0.9318    0.9314    0.9314     10000

Confusion Matrix...
[[922   8  39   3   4   8  10   0   1   5]
 [ 12 917  29   3  13   9  11   2   1   3]
 [ 74  12 854   1  36   1  21   1   0   0]
 [  2   0   0 967   8   9   4   0   2   8]
 [  9   4  18   4 900  16   8   0  35   6]
 [  3   7   2  14  10 927  27   2   2   6]
 [  2   5  11   5  13  11 949   0   0   4]
 [  1   5   2   1   1   0   2 975   0  13]
 [  0   1   2   0  45   2   5   0 941   4]
 [  1   4   0   2   6   3   2   3  17 962]]
Time usage: 0:01:19

进程已结束，退出代码为 0

        """
    else:
        pass

