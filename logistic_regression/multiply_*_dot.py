import numpy as np

"""
1. np.multiply()  数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
"""
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = np.multiply(A, B)
print(C)
"""
[ 4 10 18]
"""
A = np.arange(1, 5).reshape(2, 2)
B = np.arange(0, 4).reshape(2, 2)
C = np.multiply(A, B)
print(C)
"""
[[ 0  2]
 [ 6 12]]
"""
