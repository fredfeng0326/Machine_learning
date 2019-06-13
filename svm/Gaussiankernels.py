import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio


# kernek function 高斯核函数
def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

test_g = gaussian_kernel(x1, x2, sigma)
# print(test_g)
# 0.32465246735834974

mat = sio.loadmat('./data/ex6data2.mat')
# print(mat.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

# print(data.head())
"""
         X1        X2  y
0  0.107143  0.603070  1
1  0.093318  0.649854  1
2  0.097926  0.705409  1
3  0.155530  0.784357  1
4  0.210829  0.866228  1
"""

# print(data.shape)
# (863, 3)

sns.set(context="notebook", style="white", palette=sns.diverging_palette(240, 10, n=2),color_codes=False)
sns.lmplot('X1', 'X2', hue='y', data=data,
           size=5,
           fit_reg=False,
           scatter_kws={"s": 10}
          )
plt.show()

# try built-in Gaussian Kernel of sklearn
svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
# print(svc)
"""
gamma越大，支持向量越少，gamma值越小，支持向量越多。
SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
"""

svc.fit(data[['X1', 'X2']], data['y'])
score_g = svc.score(data[['X1', 'X2']], data['y'])
# print(score_g)
# 0.9698725376593279

predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]
print(predict_prob)
"""
Compute probabilities of possible outcomes for samples in X.
X : array-like, shape (n_samples, n_features)
For kernel=”precomputed”, the expected shape of X is [n_samples_test, n_samples_train]
"""

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')
plt.show()
