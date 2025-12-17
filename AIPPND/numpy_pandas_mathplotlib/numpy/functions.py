import numpy as np

x = np.zeros((3, 4))
print(x)
print(x.dtype)

x = np.zeros((3, 4), dtype=int)
print(x)
print(x.dtype)

x = np.ones((2, 3))
print(x)
print(x.dtype)

x = np.ones((2, 3), dtype=np.int64)
print(x)
print(x.dtype)

x = np.full((2, 2), 7)
print(x)
print(x.dtype)

x = np.eye(5, dtype=np.int64)
print(x)
print(x.dtype)

x = np.diag([10, 20, 30, 50])
print(x)
print(x.dtype)

x = np.arange(10)
print(x)
print(x.dtype)

x = np.arange(4,10)
print(x)
print(x.dtype)

x = np.arange(1,14,3)
print(x)
print(x.dtype)

x = np.linspace(0, 25, 10)
print(x)
print(x.dtype)

x = np.linspace(0, 25, 10, endpoint=False)
print(x)
print(x.dtype)

x = np.arange(20)
print(x)

x = np.reshape(x, (4, 5))
print(x)
print(x.dtype)

x = np.arange(20).reshape(4, 5)
print(x)
print(x.dtype)

x = np.linspace(0, 50, 10, endpoint=False).reshape(5, 2)
print(x)
print(x.dtype)

x = np.random.random((3, 3))
print(x)
print(x.dtype)

x = np.random.randint(4,15, size=(3, 2))
print(x)
print(x.dtype)

# We create a 1000 x 1000 ndarray of random floats drawn from normal (Gaussian) distribution
# with a mean of zero and a standard deviation of 0.1.
x = np.random.normal(0, 0.1, size=(1000,1000))
print(x)
print(x.dtype)

# We print information about X
print('X has dimensions:', x.shape)
print('X is an object of type:', type(x))
print('The elements in X are of type:', x.dtype)
print('The elements in X have a mean of:', x.mean())
print('The maximum value in X is:', x.max())
print('The minimum value in X is:', x.min())
print('X has', (x < 0).sum(), 'negative numbers')
print('X has', (x > 0).sum(), 'positive numbers')


# Using the NumPy functions you learned about on the previous page, create a 4 x 4 ndarray that only contains
# consecutive even numbers from 2 to 32 (inclusive).

x = np.arange(2,33,2).reshape(4,4)
print(x)
print(np.arange(2,34,2))

