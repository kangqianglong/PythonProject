import tensorflow as tf
from tensorflow.keras import layers, models


# 构建 LSTM 模型
def build_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(1)
    ])
    return model


# 示例使用
input_shape = (10, 1)  # 假设输入序列长度为 10，每个时间步特征维度为 1
model = build_lstm(input_shape)
model.compile(optimizer='adam', loss='mse')
model.summary()
