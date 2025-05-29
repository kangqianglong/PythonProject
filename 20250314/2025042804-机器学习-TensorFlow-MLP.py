import tensorflow as tf
from tensorflow.keras import layers, models

# 构建 MLP 模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 假设输入是 28x28 的图像，将其展平为一维向量
    layers.Dense(128, activation='relu'),  # 第一个隐藏层，128 个神经元，使用 ReLU 激活函数
    layers.Dense(64, activation='relu'),  # 第二个隐藏层，64 个神经元，使用 ReLU 激活函数
    layers.Dense(10, activation='softmax')  # 输出层，10 个神经元，对应 10 个类别，使用 Softmax 激活函数
])