# 导入
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score



# 数据获取
df = pd.read_csv('dev.csv')
data = df['words'].values
# print(data)
# 特征工程
transform = TfidfVectorizer()
feature_vector = transform.fit_transform(data)
# print(feature_vector.toarray().shape)

# 数据划分
label = df['labels']
x_train, x_test, y_train, y_test = train_test_split(feature_vector, label, test_size=0.2, random_state=22)
# 模型构建
rf = RandomForestClassifier()
# 模型训练
rf.fit(x_train, y_train)
# 模型预测
y_predict = rf.predict(x_test)
# print(y_predict)
# 模型评估
print(accuracy_score(y_test, y_predict)) # 0.8186944444444444

# print(precision_score(y_test, y_predict, average='micro'))
# print(recall_score(y_test, y_predict, average='micro'))
# print(f1_score(y_test, y_predict, average='micro'))
