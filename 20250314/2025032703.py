import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

# 准备特征和目标变量
X = df[['close', 'vol', 'ma_diff']]
y = df['signal'].shift(-1).dropna()

# 划分训练集和测试集
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 训练支持向量机模型
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 预测信号
df['predicted_signal'] = model.predict(X)
df['optimized_signal'] = df['predicted_signal']
df['optimized_position'] = df['optimized_signal'].diff()

# 策略回测
df['returns'] = df['close'].pct_change()
df['optimized_strategy_returns'] = df['optimized_position'].shift(1) * df['returns']
df['cumulative_optimized_strategy_returns'] = (1 + df['optimized_strategy_returns']).cumprod()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['cumulative_optimized_strategy_returns'], label='Optimized Strategy')
plt.title('Dual Moving Average Strategy with SVM (Gree Electric)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
