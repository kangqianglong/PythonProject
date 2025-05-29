import baostock as bs
import pandas as pd
import torch
import numpy as np  # 添加这一行导入 numpy 库
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim


# 登录 baostock
lg = bs.login()
if lg.error_code != '0':
    print(f"登录失败，错误码：{lg.error_code}，错误信息：{lg.error_msg}")
else:
    print("登录成功")

# 获取格力电器的历史数据
stock_code = "sz.000651"
start_date = "2020-01-01"
end_date = "2025-05-07"  # 可以根据需要修改为最新日期
rs = bs.query_history_k_data_plus(
    stock_code,
    "date,open,high,low,close,volume",
    start_date=start_date,
    end_date=end_date,
    frequency="d",
    adjustflag="3"
)
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
data = pd.DataFrame(data_list, columns=rs.fields)

# 退出登录
bs.logout()

# 数据处理
data[["open", "high", "low", "close", "volume"]] = data[["open", "high", "low", "close", "volume"]].astype(float)
data['date'] = pd.to_datetime(data['date'])

# 定义特征和标签，标签为整年的涨跌幅度
features = ["open", "high", "low", "volume"]
X = []
y = []
# 按年份遍历数据，计算每年的涨跌幅和特征均值
for year in data['date'].dt.year.unique():
    year_data = data[data['date'].dt.year == year]
    if len(year_data) > 0:
        # 获取年度首尾交易日的收盘价
        start_price = year_data.iloc[0]['close']
        end_price = year_data.iloc[-1]['close']
        # 计算年度涨跌幅
        change_rate = (end_price - start_price) / start_price
        # 提取年度特征均值作为输入特征
        year_features = year_data[features].mean().values
        X.append(year_features)
        y.append(change_rate)

    # 将特征和目标变量转换为numpy数组
    X = np.array(X)
    y = np.array(y)

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 模型定义
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(len(features), 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = RegressionModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 模型评估
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    # 可以计算一些评估指标，如均方根误差（RMSE）
    mse = criterion(outputs, y_test_tensor)
    rmse = torch.sqrt(mse)
    print(f'RMSE: {rmse.item()}')

# 预测最新一年数据（假设数据包含最新一年）
latest_year_data = data[data['date'].dt.year == data['date'].dt.year.max()][features].mean().values
latest_year_data = scaler.transform(latest_year_data.reshape(1, -1))
latest_year_data_tensor = torch.tensor(latest_year_data, dtype=torch.float32)
with torch.no_grad():
    prediction = model(latest_year_data_tensor)
    print(f"预测格力电器最新一年股价的变化幅度为：{prediction.item()}")

'''代码修改说明
标签定义：通过按年份对数据进行分组，计算每年年初收盘价与年末收盘价的变化率，以此作为该年的涨跌幅度标签。
特征处理：使用每年的特征均值作为该年的特征表示。
预测部分：选取数据中最新一年的特征均值，进行归一化处理后输入模型，得到该年的涨跌幅度预测值。
'''