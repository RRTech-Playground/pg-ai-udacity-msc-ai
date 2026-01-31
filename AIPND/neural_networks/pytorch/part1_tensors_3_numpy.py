import numpy as np
import torch

# To create a tensor from Numpy array, use `torch.from_numpy()`.
# To convert a tensor to a numpy array, use the `.numpy()` method.

a = np.random.randn(4, 3)
print(a)

b = torch.from_numpy(a)
print(b)

print(b.numpy())

# The memory is shared between the NumPy array and Torch tensor, so if you change the
# values in-place of one object, the other will change as well.

# Multiply PyTorch Tensor by 2, in place
b.mul_(2)
# Numpy array matches new values from Tensor
print(b)
print(a)
