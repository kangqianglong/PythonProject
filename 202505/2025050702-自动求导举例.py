import torch

# 创建需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 定义一个简单的函数
z = x**2 + y**3

# 计算梯度
z.backward()

# 查看梯度
print("x的梯度:", x.grad)
print("y的梯度:", y.grad)
'''
在这个例子中，我们首先创建了两个需要计算梯度的张量x和y，然后定义了一个函数z = x**2 + y**3。
接着调用z.backward()进行反向传播，自动计算z对x和y的梯度。最后打印出x和y的梯度。根据求导公式，
z对x的导数为2x，当x=2时，导数为4；z对y的导数为3y2，当y=3时，导数为27。所以输出结果应该是x的梯度:
 tensor([4.])和y的梯度: tensor([27.])
'''