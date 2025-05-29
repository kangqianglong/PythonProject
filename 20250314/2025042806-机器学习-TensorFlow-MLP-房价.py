import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 数据预处理：标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建 MLP 模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # 输入层神经元个数根据特征数量确定
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # 输出层只有一个神经元，因为是回归问题
])

# 编译模型
model.compile(optimizer='adam',
              loss='mean_squared_error',  # 回归问题常用均方误差作为损失函数
              metrics=['mean_absolute_error'])  # 可选择平均绝对误差作为评估指标

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# 在测试集上评估模型
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"测试集上的均方误差: {test_loss}")
print(f"测试集上的平均绝对误差: {test_mae}")

# 进行预测
predictions = model.predict(X_test)
print("部分预测结果:", predictions[:5])