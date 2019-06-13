from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio

mat_tr = sio.loadmat('data/spamTrain.mat')
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
"""
X.shape, y.shape
((4000, 1899), (4000,))
"""


mat_test = sio.loadmat('data/spamTest.mat')
# dict_keys(['__header__', '__version__', '__globals__', 'Xtest', 'ytest'])

test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
"""
test_X.shape, test_y.shape
((1000, 1899), (1000,))
"""

# fit SVM model

svc = svm.SVC()
svc.fit(X, y)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
"""

pred = svc.predict(test_X)
# print(metrics.classification_report(test_y, pred))
"""
              precision    recall  f1-score   support

           0       0.94      0.99      0.97       692
           1       0.98      0.87      0.92       308

    accuracy                           0.95      1000
   macro avg       0.96      0.93      0.94      1000
weighted avg       0.95      0.95      0.95      1000
"""


logit = LogisticRegression()
logit.fit(X, y)

pred = logit.predict(test_X)
# print(metrics.classification_report(test_y, pred))
"""
              precision    recall  f1-score   support

           0       1.00      0.99      1.00       692
           1       0.99      0.99      0.99       308

    accuracy                           0.99      1000
   macro avg       0.99      0.99      0.99      1000
weighted avg       0.99      0.99      0.99      1000
"""