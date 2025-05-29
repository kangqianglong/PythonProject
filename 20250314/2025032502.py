import tushare as ts
import pandas as pd
from sklearn.linear_model import LinearRegression

# 设置tushare token
ts.set_token('f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23')
pro = ts.pro_api()

#设置你的token
df = pro.user(token='f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23')

print(df)

# 获取最新股票数据
latest_df = pro.daily(ts_code='000651.SZ', start_date='20230101')
latest_df = latest_df.sort_values('trade_date')
latest_df['ma_200'] = latest_df['close'].rolling(window=200).mean()
latest_df = latest_df.dropna()

# 准备特征（确保格式与训练时一致）
last_ma_200 = pd.DataFrame(latest_df[['ma_200']].iloc[-1]).T  # 转为DataFrame

# 训练模型
model = LinearRegression()
X = latest_df[['ma_200']]
y = latest_df['close']
model.fit(X, y)

# 预测
predicted_price = model.predict(last_ma_200)
print(f"预测的明日股票价格：{predicted_price[0]}")