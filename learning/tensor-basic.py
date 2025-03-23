import torch

# 1D Tensor (Vector)
x1 = torch.tensor([1, 2, 3])  # Shape: [3] (a 1D array)

# 2D Tensor (Matrix)
x2 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: [2, 3] (a 2D rectangular array)

# 3D Tensor (Stack of Matrices)
x3 = torch.randn(2, 3, 4)  # Shape: [2, 3, 4] (2 matrices of size 3x4)

print(x1.shape)  # torch.Size([3])
print(x2.shape)  # torch.Size([2, 3])
print(x3.shape)  # torch.Size([2, 3, 4])

