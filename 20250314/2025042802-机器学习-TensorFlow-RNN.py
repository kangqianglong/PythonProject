import tensorflow as tf
from tensorflow.keras import layers, models


# 构建简单 RNN 模型
def build_simple_rnn(input_shape, num_classes):
    model = models.Sequential([
        layers.SimpleRNN(64, input_shape=input_shape),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# 示例使用
input_shape = (10, 20)  # 假设输入序列长度为 10，每个时间步特征维度为 20
num_classes = 2
model = build_simple_rnn(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
