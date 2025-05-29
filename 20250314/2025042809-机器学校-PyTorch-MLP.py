import torch
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成模拟回归数据（1000样本，5特征）
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = (y - y.mean()) / y.std()  # 标准化目标值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 调整为二维张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),  # 输入层→隐藏层
            torch.nn.ReLU(),                            # 激活函数
            torch.nn.Linear(hidden_size, output_size)   # 隐藏层→输出层
        )

    def forward(self, x):
        return self.layers(x)

# 初始化模型（输入5特征，隐藏层32神经元，输出1）
model = MLP(input_size=5, hidden_size=32, output_size=1)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()  # 均方误差（回归问题）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 打印训练进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # 测试模式（关闭Dropout/BatchNorm等）
        model.eval()
        with torch.no_grad():  # 禁用梯度计算（加速推理）
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            print(f"测试集损失: {test_loss.item():.4f}")