import pandas as pd
import numpy as np
import yfinance as yf
import time  # 用于添加延迟

# 1. 获取数据（以苹果公司股票为例）
tickers = ['AAPL']
data = {}
for ticker in tickers:
    time.sleep(5)  # 添加延迟，降低请求频率
    try:
        df_single = yf.download(ticker, start='2020-01-01', end='2023-12-31')
        if not df_single.empty:
            data[ticker] = df_single
        else:
            print(f"{ticker} 数据获取失败")
    except Exception as e:
        print(f"{ticker} 下载出错: {e}")

if not data:
    print("所有数据获取失败，程序终止")
else:
    df = data['AAPL'][['Close']].copy()

    # 2. 计算均线
    df['short_ma'] = df['Close'].rolling(window=50).mean()
    df['long_ma'] = df['Close'].rolling(window=200).mean()

    # 3. 生成交易信号
    df['signal'] = 0  # 0：无操作，1：买入，-1：卖出
    df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
    df['position'] = df['signal'].diff()  # 持仓变化（1：开多，-1：平仓）

    # 4. 回测结果
    df['returns'] = np.log(df['Close']).diff()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']  # 策略收益
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

    # 5. 计算年化收益率（增加数据行数校验）
    if len(df) > 0:
        annual_return = (df['cumulative_returns'].iloc[-1] ** (252 / len(df))) - 1
        print("策略年化收益率：", annual_return)
    else:
        print("数据不足，无法计算年化收益率")

    # 可视化
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Price')
    plt.plot(df['short_ma'], label='50-day MA')
    plt.plot(df['long_ma'], label='200-day MA')
    plt.title('Dual Moving Average Strategy')
    plt.legend()
    plt.show()