import pandas as pd
import matplotlib.pyplot as plt

# 1. 从Excel读取数据（示例文件：sales_data.xlsx）
excel_path = "C:/Users/hx_DBA/Desktop/datas.csv"  # 修改为你的Excel文件路径
df = pd.read_csv(excel_path,encoding='gbk')
df.describe()
print (df)
# 2. 数据预处理（假设列名为"月份"和"营业额"）
# 转换为列表格式供matplotlib使用
months = df["月份"].tolist()
revenues = df["营业额"].tolist()

# 3. 创建表格
fig, ax = plt.subplots(figsize=(10, 4))  # 设置画布大小

# 隐藏坐标轴
ax.axis("off")

# 创建表格对象
table = ax.table(
    cellText=df.values,          # 表格数据
    colLabels=df.columns,        # 列标题
    loc="center",               # 表格位置
    cellLoc="center",           # 单元格对齐方式
    colColours=["#f0f0f0"]*len(df.columns)  # 列标题背景色
)

# 4. 样式设置
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # 调整表格宽高

# 设置标题
plt.title("月度营业额统计表", fontsize=14, pad=20)

# 5. 显示/保存
plt.tight_layout()
plt.savefig("sales_table.png", dpi=300)  # 保存为图片
plt.show()