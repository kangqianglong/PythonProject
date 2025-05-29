import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_datareader.data as web
import datetime
import baostock as bs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import torch
#import torch.nn as nn

# 登录 Baostock（无需注册，直接登录）
lg = bs.login()
if lg.error_code != '0':
    print(f"登录失败，错误码：{lg.error_code}，错误信息：{lg.error_msg}")
else:
    print("登录成功")

# 定义股票代码列表（A股代码格式：sh.601318、sz.000651）
stock_codes = ["sz.000651", "sz.300498", "sh.603259"]  # 格力电器、温氏股份、药明康德
start_date = "2010-01-01"
end_date = "2023-12-31"

# 存储所有股票数据
dfs = []

for code in stock_codes:
    # 获取历史K线数据（参数说明见Baostock文档）
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",  # 需要的字段：日期、开盘价、最高价、最低价、收盘价、成交量
        start_date=start_date,
        end_date=end_date,
        frequency="d",  # 日线
        adjustflag="3"  # 复权类型：3为前复权（常用）
    )

    # 转换为DataFrame
    if rs.error_code != '0':
        print(f"获取数据失败，代码：{code}，错误信息：{rs.error_msg}")
        continue
    df = rs.get_data()
    df["stock_code"] = code  # 添加股票代码列
    dfs.append(df)

# 合并所有股票数据
data = pd.concat(dfs, ignore_index=True)




# 定义股票代码和日期范围
stock_codes = ['000651.SZ', '300498.SZ', '603259.SH']  # 格力电器、温氏股份、药明康德
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2023, 12, 31)

# 获取股票数据
#dfs = []
#for code in stock_codes:
 #   try:
 #       df = web.DataReader(code, 'yahoo', start_date, end_date)
  #      df['Stock_Code'] = code
  #      dfs.append(df)
  #  except Exception as e:
   #     print(f"Error fetching data for {code}: {e}")

# 合并数据
#data = pd.concat(dfs)

# 选择特征和目标变量
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features].values
y = data[target].values

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建 MLP 模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(len(features),)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 在测试集上评估模型
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"测试集上的均方误差: {test_loss}")
print(f"测试集上的平均绝对误差: {test_mae}")

# 进行预测
predictions = model.predict(X_test)
# 反标准化预测结果
predictions = scaler_y.inverse_transform(predictions)
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

print("部分预测结果:", predictions[:5])