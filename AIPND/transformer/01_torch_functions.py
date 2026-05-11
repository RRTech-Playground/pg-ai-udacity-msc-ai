import torch

# torch.rand()
print(torch.rand(5))
print(torch.rand(5, 5))

# torch.ones()
print(torch.ones(5))
print(torch.ones(5, 5))

# torch.arange()
print(torch.arange(0, 10))

# torch.tril()
print(torch.tril(torch.rand(5, 5)))

# torch.cat()
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along rows
print(torch.cat((tensor_a, tensor_b), dim=0))

# Concatenate along columns
print(torch.cat((tensor_a, tensor_b), dim=1))

# tensor.view()
x = torch.arange(12)
print(x)

x_reshaped = x.view(3, 4)
print(x_reshaped)

print(x_reshaped.view(12))

