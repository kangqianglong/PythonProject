import pandas as pd
import matplotlib.pyplot as plt

# 配置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel 数据（替换为你的文件路径）
file_path = 'C:/Users/hx_DBA/Desktop/datas.csv'
#excel_path = "C:/Users/hx_DBA/Desktop/datas.csv"  # 修改为你的Excel文件路径
#df = pd.read_csv(excel_path,encoding='gbk')
try:
    df = pd.read_csv(file_path,encoding='gbk')
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}")
    exit()

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(df['月份'], df['营业额（元）'], marker='o', linestyle='-', linewidth=2)

# 添加图表元素
plt.title('每月营业额收入变化趋势')
plt.xlabel('月份')
plt.ylabel('营业额（元）')
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

# 显示图形
plt.show()