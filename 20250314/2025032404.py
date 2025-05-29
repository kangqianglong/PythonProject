import pandas as pd
import matplotlib.pyplot as plt

# 配置中文字体（以黑体为例，需确保系统有该字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 模拟数据（假设你的Excel数据类似）
data = {
    "月份": ["1月", "2月", "3月", "4月", "5月", "6月"],
    "营业额（元）": [120, 95, 150, 130, 160, 145]
}
df = pd.DataFrame(data)

# 创建图形
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')  # 隐藏坐标轴

# 绘制表格
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'  # 表格内容居中
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)  # 调整表格大小

plt.title("每月营业额统计")
plt.savefig("business_table.png", bbox_inches='tight')
plt.show()