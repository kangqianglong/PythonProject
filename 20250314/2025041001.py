import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# 生成一些随机数据用于线性回归
# 生成100个随机的x值，范围在0到1之间
x_data = np.random.rand(100).astype(np.float32)
# 根据线性关系y = 2*x + 1生成对应的y值，再加上一些噪声
y_data = 2 * x_data + 1 + 0.1 * np.random.randn(100).astype(np.float32)

# 定义模型参数，随机初始化权重和偏置
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# 定义模型的预测函数
def forward(x):
    return W * x + b

# 定义损失函数（均方误差）
def loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# 定义优化器，使用梯度下降法，学习率为0.5
optimizer = tf.optimizers.SGD(0.5)


# 训练模型
for step in range(100):
    with tf.GradientTape() as tape:
        y_pred = forward(x_data)
        loss_value = loss(y_pred, y_data)
    gradients = tape.gradient(loss_value, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    if step % 10 == 0:
        print('Step:', step, 'Loss:', loss_value.numpy())

print("训练后的权重:", W.numpy(), "训练后的偏置:", b.numpy())
