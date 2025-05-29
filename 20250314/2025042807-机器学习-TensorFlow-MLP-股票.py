import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import baostock as bs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# ==================== 数据获取（Baostock） ====================
# 登录Baostock（无需注册）
lg = bs.login()
if lg.error_code != '0':
    print(f"登录失败，错误码：{lg.error_code}，错误信息：{lg.error_msg}")
else:
    print("登录成功")

# 定义股票代码（A股格式：sz.000651、sh.603259）
stock_codes = ["sz.000651", "sz.300498", "sh.603259"]  # 格力电器、温氏股份、药明康德
start_date = "2010-01-01"
end_date = "2023-12-31"

dfs = []
for code in stock_codes:
    # 获取日线数据（前复权）
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    if rs.error_code != '0':
        print(f"数据获取失败（{code}）: {rs.error_msg}")
        continue
    df = rs.get_data()
    df["stock_code"] = code  # 添加股票代码列
    dfs.append(df)

# 合并数据并转换数值类型
data = pd.concat(dfs, ignore_index=True)
data[["open", "high", "low", "close", "volume"]] = data[["open", "high", "low", "close", "volume"]].astype(float)
bs.logout()  # 退出登录

# ==================== 数据预处理 ====================
# 选择特征和目标变量
features = ["open", "high", "low", "volume"]  # 输入特征（开盘价、最高价、最低价、成交量）
target = "close"  # 预测目标：收盘价

X = data[features].values
y = data[target].values

# 标准化数据（特征和目标）
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # 转换为一维数组

# 划分训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ==================== 构建MLP模型（TensorFlow） ====================
model = models.Sequential([
    # 输入层（特征数量决定输入维度）
    layers.Input(shape=(len(features),)),
    # 隐藏层1（64个神经元，ReLU激活）
    layers.Dense(64, activation='relu'),
    # 隐藏层2（32个神经元，ReLU激活）
    layers.Dense(32, activation='relu'),
    # 输出层（1个神经元，线性激活，用于回归）
    layers.Dense(1)
])

# 编译模型：指定优化器、损失函数和评估指标
model.compile(
    optimizer='adam',  # Adam优化器（自适应学习率）
    loss='mse',       # 均方误差（回归问题常用）
    metrics=['mae']   # 平均绝对误差（辅助评估）
)

# 打印模型结构
model.summary()

# ==================== 训练模型 ====================
history = model.fit(
    X_train, y_train,
    epochs=100,         # 训练轮数
    batch_size=32,      # 批次大小
    validation_split=0.1,  # 10%训练数据作为验证集（监控过拟合）
    verbose=1           # 输出训练进度（0=静默，1=进度条，2=每轮一行）
)
# ==================== 评估与预测 ====================
# 在测试集上评估模型
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n测试集性能：均方误差(MSE)={test_loss:.4f}，平均绝对误差(MAE)={test_mae:.4f}")

# 预测测试集并反标准化（还原为真实价格）
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # 反标准化预测值
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))  # 反标准化真实值

# 打印部分预测结果
print("\n部分预测值 vs 真实值（元）：")
for i in range(5):
    print(f"样本{i+1}: 预测={y_pred[i][0]:.2f}, 真实={y_test_original[i][0]:.2f}")