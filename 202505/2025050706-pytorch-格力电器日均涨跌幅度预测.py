import baostock as bs
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

# 登录baostock
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

# 定义特征和标签，标签为收盘价的变化率
features = ["open", "high", "low", "volume"]
X = data[features].values
y = (data['close'].shift(-1) - data['close']) / data['close']
X = X[:-1]
y = y[:-1].values  # 去除最后一行，因为最后一行没有对应的标签

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
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

# 预测最新数据
latest_data = X[-1].reshape(1, -1)
latest_data_tensor = torch.tensor(latest_data, dtype=torch.float32)
with torch.no_grad():
    prediction = model(latest_data_tensor)
    print(f"预测格力电器股价的变化幅度为：{prediction.item()}")

'''
    在上述代码中，将模型的输出层改为一个神经元，用于预测股票价格的变化率。使用均方误差（MSE）作为损失函数，
    Adam 作为优化器。在训练和评估过程中，计算并输出损失值和均方根误差（RMSE）。
    最后，使用训练好的模型对最新数据进行预测，输出预测的股票价格变化幅度。
'''