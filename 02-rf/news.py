import pandas as pd
import numpy as np
import jieba
from collections import Counter

df = pd.read_csv('../01-data/train.txt', sep='\t', names=['sentences', 'labels'])
# print(df)
# print(len(df))
count = Counter(df['labels'])
# print(count)#18000
# print(len(count))#10
total = 0
for i,v in count.items():
    total += v
# print(total)

# for i,v in count.items():
#     print(f'{i}:{v / total * 100}%')

# print('***************************************')
df['sentence_len'] = df['sentences'].apply(lambda x: len(x))
# print(df.head(10))
length_mean = np.mean(df['sentence_len'])
length_std = np.std(df['sentence_len'])
# print('***************************************')
# print(f'mean:{length_mean:.2f}, std:{length_std:.2f}')
def cut_sentence(sentence):
    return list(jieba.cut(sentence))
# df['words'] = df['sentences'].apply(cut_sentence)
# print(df.head(10))
df['words'] = df['sentences'].apply(lambda s: ' '.join(cut_sentence(s))[:30])
for i in range(10):
    print(len(df.iloc[i]['words']))
print('***************************************')
# df['words'] = df['words'].apply(lambda s: ' '.join(s.split())[:30])
# for i in range(10):
#     print(len(df.iloc[i]['words']))
df.to_csv('dev.csv')





