import pandas_datareader.data as web
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 定义时间范围（2025年数据未真实产生，示例用至2024年）
start_date = '2015-01-01'
end_date = '2024-12-31'

# 2. 获取深证成指数据（更换为stooq数据源）
sz_index = web.DataReader('^SZSE', 'stooq', start_date, end_date)
sz_index = sz_index[['Close']].rename(columns={'Close': 'sz_close'})

# 3. 获取格力电器数据（000651.SZ，更换为stooq数据源）
gree = web.DataReader('000651.SZ', 'stooq', start_date, end_date)
gree = gree[['Close']].rename(columns={'Close': 'gree_close'})

# 4. 合并数据
data = pd.merge(sz_index, gree, left_index=True, right_index=True, how='inner')
data.index.name = 'trade_date'

# 5. 准备特征和标签
X = data[['sz_close']]  # 深证指数收盘价作为特征
y = data['gree_close']  # 格力电器收盘价作为标签

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 8. 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差: {mse}")
print(f"R²分数: {r2}")

# 9. 可视化结果
plt.figure(figsize=(12, 6))
plt.scatter(X_test['sz_close'], y_test, color='blue', label='实际格力股价')
plt.plot(X_test['sz_close'], y_pred, color='red', linewidth=2, label='预测格力股价')
plt.title('深证指数与格力电器股价线性回归预测')
plt.xlabel('深证指数收盘价')
plt.ylabel('格力电器收盘价')
plt.legend()
plt.show()