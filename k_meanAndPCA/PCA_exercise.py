# PCA是在数据集中找到“主成分”或最大方差方向的线性变换。 它可以用于降维。 在本练习中，我们首先负责实现PCA并将其应用于一个简单的二维数据集，以了解它是如何工作的。 我们从加载和可视化数据集开始。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat('data/ex7data1.mat')
# print(data)
# {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Mon Nov 14 22:41:44 2011', '__version__': '1.0', '__globals__': [], 'X': array([[3.38156267, 3.38911268],
#        [4.52787538, 5.8541781 ],
#        [2.65568187, 4.41199472],
#        [2.76523467, 3.71541365],
#        [2.84656011, 4.17550645],
#        [3.89067196, 6.48838087],
#         ...
#       ])}

X = data['X']
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V


U, S, V = pca(X)


# 现在我们有主成分（矩阵U），我们可以用这些来将原始数据投影到一个较低维的空间中。 对于这个任务，我们将实现一个计算投影并且仅选择顶部K个分量的函数，有效地减少了维数。
def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


Z = project_data(X, U, 1)


def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)


X_recovered = recover_data(Z, U, 1)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()
