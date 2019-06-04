import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# #Return the first n rows. n default 5
# print(data.head())

# 看看数据长什么样子
# print(data.describe())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)

# 现在我们来做一些变量初始化。
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
y = data.iloc[:, cols - 1:cols]  # y是所有行，最后一列

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

computeCost(X, y, theta)
