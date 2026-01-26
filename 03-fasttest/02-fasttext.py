import fasttext

# model = fasttext.train_supervised('./dev_fast.txt', 2)
# print(f'词的数量：{len(model.words)}')
# print(f'标签值:{model.labels}')
#
# result = model.test('./test_fast.txt')
# print(f'准确率:{result}') # 0.7949

model = fasttext.train_supervised('./train_fast.txt', 2)
print(f'词的数量：{len(model.words)}')
print(f'标签值:{model.labels}')

result = model.test('./test_fast.txt')
print(f'准确率:{result}') # 0.8441










