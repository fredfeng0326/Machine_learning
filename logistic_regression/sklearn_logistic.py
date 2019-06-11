import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
X = np.array(X.values)
y = np.array(y.values)
# y = y.ravel()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试

classifier = LogisticRegression()  # 使用类，参数全是默认的
classifier.fit(X_train, y_train)  # 训练数据来学习，不需要返回值

# 分类测试集，这将返回一个测试结果的数组
y_pred = classifier.predict(X_train)
print(y_pred)
# 准确率 0.7875
print(accuracy_score(y_train, y_pred))
