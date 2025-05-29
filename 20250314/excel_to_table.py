import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 读取 Excel 文件
file_path = 'C:/Users/hx_DBA/Desktop/datas.csv'
try:
    df = pd.read_csv(file_path,encoding='gbk')
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}。")
else:
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 隐藏坐标轴
    ax.axis('off')

    # 创建表格
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # 设置表格字体大小
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # 设置标题
    plt.title('每月营业额收入')

    # 保存为 PDF 文件
    with PdfPages('monthly_revenue_table.pdf') as pdf:
        pdf.savefig(fig)

    # 显示图形
    plt.show()
    