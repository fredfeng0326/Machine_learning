import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Copy from ComputeCost.py
"""
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)

# 现在我们来做一些变量初始化。
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
y = data.iloc[:, cols - 1:cols]  # y是所有行，最后一列

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


"""
批量梯度下降
"""


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    # 返回来一个给定形状和类型的用0填充的数组 np.zeros()
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            # 组和矩阵对应位置相乘，输出与相乘数组 / 矩阵的大小一致
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)
print(cost)
# [[-3.24140214  1.1272942 ]]  4.5159555
