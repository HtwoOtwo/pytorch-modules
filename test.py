import torch

# 定义 x 和 y 的范围
x_range = torch.arange(0, 3)
y_range = torch.arange(0, 2)

# 使用 torch.meshgrid 创建网格
X, Y = torch.meshgrid(x_range, y_range)

# X 和 Y 是生成的网格
print(X.shape, Y.shape)
print(X)
print(Y)

print(X.flatten())
print(Y.flatten())


