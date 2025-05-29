import tushare as ts
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 获取股票数据（需提前注册 tushare 获取 token）
ts.set_token("f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23")
pro = ts.pro_api()
df = pro.daily(ts_code="600835.SH", start_date="20230101", end_date="20240101")
df = df.sort_values("trade_date").reset_index(drop=True)


df2 = pro.daily(ts_code='600835.SH', start_date='20250324', end_date='')

#查看数据
print(df2)



# 2. 构造特征与标签（简单用前一日收盘价预测当日收盘价）
df["prev_close"] = df["close"].shift(1)
df = df.dropna()

X = df[["prev_close"]]  # 特征：前一日收盘价
y = df["close"]         # 标签：当日收盘价

# 3. 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 预测并可视化
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual Price", color="blue")
plt.plot(y_test.index, y_pred, label="Predicted Price", color="red", linestyle="--")
plt.title("上海机电股票")
plt.xlabel("日期")
plt.ylabel("股票价格")
plt.legend()
plt.show()