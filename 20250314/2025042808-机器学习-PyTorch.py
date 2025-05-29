import torch

# 检查版本
print(f"PyTorch版本: {torch.__version__}")

# 检查GPU是否可用（NVIDIA显卡）
print(f"GPU可用: {torch.cuda.is_available()}")

# 检查MPS是否可用（Mac M系列芯片）
print(f"MPS可用: {torch.backends.mps.is_available()}")

# 创建一个简单的张量并计算
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y
print(f"计算结果: {z}")  # 应输出 tensor([5., 7., 9.])