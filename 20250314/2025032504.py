import pandas_datareader.data as web
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 获取数据
# 深证成指代码为^SZSE
start_date = '2015-01-01'
end_date = '2024-12-31'
sz_index = web.DataReader('^SZSE', 'yahoo', start_date, end_date)
sz_index = sz_index[['Close']].rename(columns={'Close':'sz_close'})

# 格力电器代码为000651.SZ
gree = web.DataReader('000651.SZ', 'yahoo', start_date, end_date)
gree = gree[['Close']].rename(columns={'Close': 'gree_close'})

# 2. 合并数据
data = pd.merge(sz_index, gree, left_index=True, right_index=True, how='inner')
data.index.name = 'trade_date'

# 3. 准备特征和标签
X = data[['sz_close']]  # 深证指数收盘价作为特征
y = data['gree_close']  # 格力电器收盘价作为标签

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差: {mse}")
print(f"R²分数: {r2}")

# 7. 可视化结果
plt.figure(figsize=(12, 6))
plt.scatter(X_test['sz_close'], y_test, color='blue', label='实际格力股价')
plt.plot(X_test['sz_close'], y_pred, color='red', linewidth=2, label='预测格力股价')
plt.title('深证指数与格力电器股价线性回归预测')
plt.xlabel('深证指数收盘价')
plt.ylabel('格力电器收盘价')
plt.legend()
plt.show()