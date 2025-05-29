import matplotlib.pyplot as plt
import numpy as np

# 生成 x 轴数据，范围从 0 到 2π，共 1000 个点
x = np.linspace(0, 2 * np.pi, 1000)
# 计算对应的正弦值
y = np.sin(x)

# 创建一个图形窗口
plt.figure(figsize=(8, 6))
# 绘制正弦函数图像
plt.plot(x, y, label='sin(x)')

# 设置图形标题
plt.title('Sine Function')
# 设置 x 轴标签
plt.xlabel('x')
# 设置 y 轴标签
plt.ylabel('y')
# 显示图例
plt.legend()
# 显示网格线
plt.grid(True)

# 显示图形
plt.show()
    