import tushare as ts
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 关键字体配置
plt.rcParams.update({
    "font.family": "SimHei",  # 指定黑体，需与系统安装字体一致
    "axes.unicode_minus": False  # 正确显示负号
})

# 设置tushare的token，需要替换为你自己的token
ts.set_token('f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23')
pro = ts.pro_api()

# 获取股票数据，这里以600835.SH（上海机电）为例，你可以替换为其他股票代码
df = pro.daily(ts_code='000651.SZ', start_date='20200101', end_date='20240101')
df = df.sort_values('trade_date')

# 计算200日均线
df['ma_200'] = df['close'].rolling(window=200).mean()
# 去除包含NaN值的行
df = df.dropna()

# 准备特征和标签
X = df[['ma_200']]
y = df['close']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差: {mse}")

# 可视化结果
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='实际值')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')
plt.title('基于200日均线的股票价格预测')
plt.xlabel('200日均线')
plt.ylabel('收盘价')
plt.legend()
plt.show()
