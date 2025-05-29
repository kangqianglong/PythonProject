import pandas as pd
import numpy as np

# 创建一维数组
arr1 = np.array([1, 2, 3, 4])
print(arr1)  # 输出：[1 2 3 4]

# 创建二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# 输出：
# [[1 2 3]
#  [4 5 6]]

# 创建一个 3x3 的单位矩阵
arr = np.eye(4)
print(arr)
# 输出：
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 生成一个随机浮点数数组，长度为5
arr = np.random.rand(5)
print(arr)  # 输出类似：[0.26674455 0.85674365 0.83504875 0.0947853  0.59059363]

# 生成一个2x3的随机数组
arr_2d = np.random.rand(2, 3)
print(arr_2d)
#输出：[[0.53906339 0.94653916 0.48469193]
# [0.72947324 0.55864621 0.23548281]]

# 生成1到10之间的一个随机整数
rand_int = np.random.randint(1, 10)
print(rand_int)#输出：9

# 生成1到10之间的2x3的随机整数数组
arr = np.random.randint(1, 10, size=(2, 3))
print(arr)
#输出：[[9 8 2]
# [9 8 5]]

a1 = np.array([(1,2,3,4),(4,5,2,4)])

print(a1)
np.savetxt("a.txt", a1)
#np.loadtxt("a.txt")
#c2 = np.load("a.txt")
#print(c2)


print(np.eye(4))


arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 沿行方向拼接（axis=0）
result = np.concatenate((arr1, arr2), axis=0)
print(result);
# 输出:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 沿列方向拼接（axis=1）
result = np.concatenate((arr1, arr2), axis=1)
print(result)
# 输出:
# [[1 2 5 6]
#  [3 4 7 8]]

# 一维数组的创建
data = pd.Series([2,3,4,1,4,5])
data = data.drop_duplicates()   # 对数据进行去重
print (data)
                            # 系统会给一个默认的从0开始的索引
# 读文件 基本都是csv文件
# header = None 不把第一行读成列名   names 指定读取的列名
stu_df = pd.read_csv('C:/Users/hx_DBA/Desktop/abc.csv',header = None,names =['name','age','time'])
stu_df.describe()
print (stu_df)


print ('天空')
# 读取 CSV 文件
data = pd.read_csv('C:/Users/hx_DBA/Desktop/data.csv')
# 处理缺失值
data = data.fillna(0)

# 处理重复值
data = data.drop_duplicates()

# 处理异常值
#z_scores = np.abs((data['column_name'] - data['column_name'].mean()) / data['column_name'].std())
#data = data[z_scores <= 3]

# 数据类型转换
#data['column_name'] = data['column_name'].astype(int)

# 处理字符串数据
#data['string_column'] = data['string_column'].str.strip()
#data['string_column'] = data['string_column'].str.lower()


# 去除字符串列的前后空格
data['gender'] = data['gender'].str.strip()

# 将字符串列转换为小写
data['name'] = data['name'].str.upper()

# 按某列的值进行升序排序
data = data.sort_values(by='age')

# 保存清洗后的数据
data.to_csv('C:/Users/hx_DBA/Desktop/cleaned_data.csv', index=False)




# 使用字典创建数据框
data = {"Name":["Alice", "Bob", "Charlie", "David", "Ella"],
        "Age":[21, 22, 23, 24, 25],
        "Gender":["F", "M", "M", "M", "F"]}
df1 = pd.DataFrame(data)  #将字典转换为数据框
print(df1)  #打印数据框
# 使用列表创建数据框
data = [['Alice', 25], ['Bob', 20], ['Charlie', 30]]
df2 = pd.DataFrame(data, columns=['Name', 'Age'])
print(df2)


# 创建一个示例DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [True, False, True]})
# 打印DataFrame的基本信息
df.info()


