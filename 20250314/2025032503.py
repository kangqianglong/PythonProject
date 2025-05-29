import tushare as ts
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 配置tushare（需替换为自己的token）
ts.set_token("f1318f5ebd2a2a76919be7ed1fea9468cd898c036dc9cf00a0eeff23")
pro = ts.pro_api()

# 2. 获取数据（注意：2025年数据未到，调整为合理时间范围）
# 深证成指数据
sz_index = pro.index_daily(ts_code="399001.SZ", start_date="20200101", end_date="20240101")
sz_index = sz_index[['trade_date', 'close']].rename(columns={'close': 'sz_close'})

# 格力电器数据
gree = pro.daily(ts_code="000651.SZ", start_date="20200101", end_date="20240101")
gree = gree[['trade_date', 'close']].rename(columns={'close': 'gree_close'})

# 3. 合并数据
data = pd.merge(sz_index, gree, on='trade_date', how='inner')
data['trade_date'] = pd.to_datetime(data['trade_date'])

# 4. 准备特征和标签
X = data[['sz_close']]  # 深证指数收盘价作为特征
y = data['gree_close']  # 格力电器收盘价作为标签

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 7. 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差: {mse}")
print(f"R²分数: {r2}")

# 8. 可视化结果
plt.figure(figsize=(12, 6))
plt.scatter(X_test['sz_close'], y_test, color='blue', label='实际格力股价')
plt.plot(X_test['sz_close'], y_pred, color='red', linewidth=2, label='预测格力股价')
plt.title('深证指数与格力电器股价线性回归预测')
plt.xlabel('深证指数收盘价')
plt.ylabel('格力电器收盘价')
plt.legend()
plt.show()