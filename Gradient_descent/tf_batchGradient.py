import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Copy from ComputeCost.py
"""
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

x_data = data['Population'].values
y_data = data['Profit'].values

# 打印出这些点y
plt.plot(x_data, y_data, 'r*', label="Original data")  # 红色星形的点
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()



"""
2.构建线性模型
"""

W = tf.Variable(tf.random_uniform([1], -0.1, 1.0))  # 初始化 Weight
b = tf.Variable(tf.zeros([1]))  # 初始化 bias
y = W * x_data + b

"""
3.定义 loss/cost function, 对tensor 所有维度计算 ((y- y_data)^2)之和/N
"""

loss = tf.reduce_mean(tf.square(y - y_data))

"""
4.用梯度下降优化器来优化loss function
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 学习率
train = optimizer.minimize(loss)

"""
5.创建会话
"""
sess = tf.Session()
# 初始数据流图
init = tf.global_variables_initializer()
sess.run(init)

# 训练2000步
for step in range(2000):
    # 优化每一步
    sess.run(train)
    # 打印出每一步的损失,权重，和偏差
    print("第 {} 步的 损失={}, 权重={}, 偏差={}".format(step + 1, sess.run(loss), sess.run(W), sess.run(b)))

# 图像 2 ：绘制所有的点并且绘制出最佳拟合的直线
plt.plot(x_data, y_data, 'r*', label="Original data")  # 红色星形的点
plt.title("Linear Regression using Gradient Descent")  # 标题，表示 "梯度下降解决线性回归"
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted line")  # 拟合的线
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 关闭会话
sess.close()

# 2000步 结果
"""
# 2000步 结果 第 2000 步的 损失=8.953943252563477, 权重=[1.1927482], 偏差=[-3.8929393]
"""