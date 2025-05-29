import tensorflow as tf

# 创建两个常量张量
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 矩阵乘法运算
c = tf.matmul(a, b)

# 直接打印结果
print(c.numpy())  # 使用 .numpy() 将张量转换为 NumPy 数组