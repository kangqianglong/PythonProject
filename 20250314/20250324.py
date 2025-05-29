import matplotlib.pyplot as plt
import numpy as np

# 准备数据
x = np.linspace(0, 10, 100)
y = x ** 2

fig = plt.figure()  # 创建画布
ax = fig.add_subplot(111)  # 创建坐标系绘图区域
ax.plot(x, y)  # 绘制曲线
plt.show()
