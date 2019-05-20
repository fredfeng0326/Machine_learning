# -*- coding: UTF-8 -*-

"""
用梯度下降的优化方法来快速解决线性回归问题
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
1.构建数据
"""
points_num = 100
vectors = []
# 用numpy 的 正太随机分布函数生成 100 个点,这些点的x,y坐标值对应线性方程 y = 0.1x + 0.2

for i in range(points_num):
    x1 = np.random.normal(0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]  # 真实的x坐标
y_data = [v[1] for v in vectors]  # 真实的y坐标

# 打印出这些点
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)  # 学习率
train = optimizer.minimize(loss)

"""
5.创建会话
"""
sess = tf.Session()
# 初始数据流图
init = tf.global_variables_initializer()
sess.run(init)

# 训练20步
for step in range(20):
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