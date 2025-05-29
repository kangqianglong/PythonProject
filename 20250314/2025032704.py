import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tushare as ts

# 设置 tushare token
ts.set_token('f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23')
pro = ts.pro_api()

# 获取格力电器历史数据
df = pro.daily(ts_code='000651.SZ', start_date='20200101', end_date='20231231')
df.sort_values('trade_date', inplace=True)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)

# 计算双均线
short_window = 50
long_window = 200
df['short_ma'] = df['close'].rolling(window=short_window).mean()
df['long_ma'] = df['close'].rolling(window=long_window).mean()
df['ma_diff'] = df['short_ma'] - df['long_ma']

# 生成双均线策略信号
df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
df = df.dropna()

# 归一化处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['close', 'ma_diff']])

# 构建时间序列数据集
sequence_length = 10
X = []
y = []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(df['signal'].iloc[i + sequence_length])
X = np.array(X)
y = np.array(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, batch_size=1, epochs=1)

# 预测信号
predictions = model.predict(X_test)
df['predicted_signal'] = np.nan
df.loc[df.index[-len(predictions):], 'predicted_signal'] = predictions.flatten()
df['optimized_signal'] = np.where(df['predicted_signal'] > 0, 1, -1)
df['optimized_position'] = df['optimized_signal'].diff()

# 策略回测
df['returns'] = df['close'].pct_change()
df['optimized_strategy_returns'] = df['optimized_position'].shift(1) * df['returns']
df['cumulative_optimized_strategy_returns'] = (1 + df['optimized_strategy_returns']).cumprod()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['cumulative_optimized_strategy_returns'], label='Optimized Strategy')
plt.title('Dual Moving Average Strategy with LSTM (Gree Electric)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
