import numpy as np
import matplotlib.pyplot as plt

# 构造数据
x1 = np.arange(-5, 5, 0.25)
y1 = np.arange(-5, 5, 0.25)

X, Y = np.meshgrid(x1, y1)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_wireframe(X, Y, Z, rstride=3, cstride=2)  # 绘制三维线框图
plt.show()
