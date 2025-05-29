import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

# 1. 初始化tushare（需替换为自己的token）
ts.set_token("f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23")
pro = ts.pro_api()

# 2. 获取苹果公司（AAPL）美股数据
#df = pro.us_daily(ts_code="000651.SZ", start_date="20220101", end_date="20231231")
df = pro.daily(ts_code='600104.SH', start_date='20200101', end_date='20240101')
if df.empty:
    print("数据获取失败，检查token或日期范围")
    exit()

# 处理数据格式
df = df.sort_values('trade_date')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date')[['close']].rename(columns={'close': 'Close'})

# 3. 计算均线
df['short_ma'] = df['Close'].rolling(window=50).mean()
df['long_ma'] = df['Close'].rolling(window=200).mean()

# 4. 生成交易信号
df['signal'] = 0  # 0：无操作，1：买入，-1：卖出
df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
df['position'] = df['signal'].diff()  # 持仓变化（1：开多，-1：平仓）

# 5. 回测结果
df['returns'] = np.log(df['Close']).diff()
df['strategy_returns'] = df['position'].shift(1) * df['returns']  # 策略收益
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

# 6. 计算年化收益率
if len(df) > 0:
    annual_return = (df['cumulative_returns'].iloc[-1] ** (252/len(df))) - 1
    print("策略年化收益率：", annual_return)
else:
    print("数据不足，无法计算年化收益率")

# 7. 可视化
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Price')
plt.plot(df['short_ma'], label='50-day MA')
plt.plot(df['long_ma'], label='200-day MA')
plt.title('Dual Moving Average Strategy')
plt.legend()
plt.show()