import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), np.array([1,2,3,4,5])
print(X)
print(y)
print(len(X))
print(type(X))
print(len(y))
print(type(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)