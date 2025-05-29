from docx import Document
import pandas as pd

# 桌面路径（注意使用双反斜杠或原始字符串）
desktop_path = r"C:\Users\hx_DBA\Desktop"

# 读取桌面的Word文档
doc = Document(f"{desktop_path}\\系统.docx")

# 提取表格数据
data = ((cell.text for cell in row.cells) for row in doc.tables[0].rows)  # 假设使用第一个表格

# 转换为DataFrame并输出到桌面Excel
pd.DataFrame(data).to_excel(f"{desktop_path}\\output.xlsx", index=False)