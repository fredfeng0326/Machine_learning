from sklearn import svm
# from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('./data/ex6data3.mat')
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y', 'yval', 'Xval'])


training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')

"""
training
           X1        X2  y
0   -0.158986  0.423977  1
1   -0.347926  0.470760  1
2   -0.504608  0.353801  1
3   -0.596774  0.114035  1

cv
           X1        X2  y
0   -0.353062 -0.673902  0
1   -0.227126  0.447320  1
2    0.092898 -0.753524  0
3    0.148243 -0.718473  0
"""

"""
manual grid search for  𝐶  and  𝜎
"""
candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

combination = [(C, gamma) for C in candidate for gamma in candidate]
# 9*9 81 种组合

search = []

for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))


best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

# print(best_score, best_param)
# 0.965 (0.3, 100)

best_svc = svm.SVC(C=100, gamma=0.3)
best_svc.fit(training[['X1', 'X2']], training['y'])
ypred = best_svc.predict(cv[['X1', 'X2']])

print(metrics.classification_report(cv['y'], ypred))