import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

# mat数据格式是Matlab的数据存储的标准格式。 一般在python 中都用scipy.io.loadmat 来打开
mat = sio.loadmat('./data/ex6data1.mat')
# print(mat.keys())

# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])

data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

# print(data)
"""
       X1      X2  y
0  1.9643  4.5957  1
1  2.2753  3.8589  1
2  2.9781  4.5651  1
3  2.9320  3.5519  1
4  3.5772  2.8560  1
"""


fig, ax = plt.subplots(figsize=(8,6))
"""
fig  --  Figure(800x600)
ax -- AxesSubplot(0.125,0.11;0.775x0.77)
"""
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='copper')
ax.set_title('Raw data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

# 其中 C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1', 'X2']], data['y'])
score1 = svc1.score(data[['X1', 'X2']], data['y'])
# 0.9803921568627451

# 计算样本点到分割超平面的函数距离
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])
"""
0     0.799288
1     0.382056
2     1.374693
3     0.520373
4     0.334441
5     0.869356
6     0.686276
7     1.609823
8     0.832599
9     1.163944
"""

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

# with large C, you try to overfit the data, so the left hand side edge case now is categorized right

svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc100.fit(data[['X1', 'X2']], data['y'])
score100 = svc100.score(data[['X1', 'X2']], data['y'])
# print(score100)
# 0.9019607843137255

data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM100 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()

# print(data.head())
"""
       X1      X2  y  SVM1 Confidence  SVM100 Confidence
0  1.9643  4.5957  1         0.880305           4.057321
1  2.2753  3.8589  1         0.431562           2.145418
2  2.9781  4.5651  1         1.432925           5.164162
3  2.9320  3.5519  1         0.545438           1.973529
4  3.5772  2.8560  1         0.322202           0.584310
"""