import torch

# 创建张量
# 从Python列表创建
tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("从列表创建的张量:")
print(tensor_from_list)

# 创建全零张量
zeros_tensor = torch.zeros((2, 3))
print("\n全零张量:")
print(zeros_tensor)

# 索引与切片
print("\n索引与切片操作:")
print("第一行:", tensor_from_list[0])
print("第一列:", tensor_from_list[:, 0])

# 数学运算
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
# 加法
add_result = tensor1 + tensor2
print("\n加法结果:")
print(add_result)
# 矩阵乘法
matmul_result = torch.matmul(tensor1, tensor2)
print("\n矩阵乘法结果:")
print(matmul_result)

# 形状变换
original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
flattened_tensor = original_tensor.flatten()
print("\n展平后的张量:")
print(flattened_tensor)