from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils.utils import get_time_dif
# from transformers.optimization import AdamW
from torch.optim import AdamW # 导入方式换了
from tqdm import tqdm
import math
import logging


def train(model, train_iter, dev_iter, config):
    # 记录起始时间
    start_time = time.time()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss() # nn的交叉熵和F交叉熵有不同。
    # 优化器
    param_optimizer = list(model.named_parameters())
    # named_parameters()返回的是一个迭代器，迭代器返回的是一个元组，元组中第一个元素是参数名，第二个元素是参数值。
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],# 两个for循环嵌套，外层循环得到params和n，内层循环判断是否有,
            'weight_decay': 0.01
        },
        {
            'params' : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    dev_best_loss = float('inf')
    # 模型切换训练模式
    model.train()
    # 遍历epoch
    for epoch in range(config.num_epochs):
        total_batch = 0
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        # 遍历batch
        for i, (trains,labels) in enumerate(tqdm(train_iter)):# enumerate()函数返回的是索引和值。还有个get()场景不同
            # 前向传播
            output = model(trains)
            # 损失
            loss = loss_fn(output, labels)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # step更新
            optimizer.step()
            # 验证保存
            if total_batch % 100 == 0 and total_batch != 0:
                true = labels.data.cpu() # cpu()作用是，把GPU数据转移到CPU，这样python，numpy等才能运作，且此方法不会改变gpu上的张量，会返回一个新的。
                predict = torch.max(output.data,1)[1].cpu() # [batch, 10] ->10, [name,index] -> index
                train_acc = metrics.accuracy_score(true, predict)
                # 评估验证集效果
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 如果验证集损失更低，保存模型参数
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*' # improve用来记录是否提升。
                else:
                    improve = ''
                # 时间差
                time_dif = get_time_dif(start_time)
                # 输出训练和验证集的结果
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 评估完成置于训练模式，更新参数
                model.train()
            total_batch += 1
# 验证函数
def evaluate(config, model, data_iter, test=False):
    model.eval()
    # dev_acc, dev_loss
    loss_total = 0
    # 预测结果
    predict_all = np.array([], dtype=int)
    # label信息
    labels_all = np.array([], dtype=int)
    # 不梯度计算
    with torch.no_grad():
        # 遍历数据集
        for texts, labels in data_iter:
            # 数据送入网络
            outputs = model(texts)
            # 损失函数
            loss = F.cross_entropy(outputs, labels)
            # 损失和
            loss_total += loss.item()
            # label
            labels = labels.data.cpu().numpy()
            # 获取预测结果
            predict = torch.max(outputs.data, 1)[1].cpu().numpy() # [batch, 10] ->10, [name,index] -> index
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    # 是不是test
    if test:
        # 如果是测试集，计算分类报告、混淆矩阵
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    else:
        return acc, loss_total / len(data_iter)


def test(model, test_iter, config):
    # model.eval()  # 量化推理要关闭
    start_time = time.time()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    msg = 'Test loss: {0:>5.2f} Test acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)







