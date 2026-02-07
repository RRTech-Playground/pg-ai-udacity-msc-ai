import numpy as np

x = np.array([1, 2, 3, 4, 5])

print(x)
print(type(x))

print(x.dtype)
print(x.shape)

x = np.array([1, 2, 3])
print('ndim:', x.ndim)

## 2-D array
y = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])
print('ndim:', y.ndim)

## Here the`zeros()` is an inbuilt function that you'll study on the next page. ## The tuple (2, 3, 4( passed as an argument represents the shape of the ndarray
y = np.zeros((2, 3, 4))
print('ndim:', y.ndim)

y = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9], [10, 11, 12]])
print(y.shape)

print(y)
print(y[2][1])

print(y.size)

x = np.array(["Hello", "World"])
print(x)

print('shape:', x.shape)
print('type:', type(x))
print('dtype:', x.dtype)
print('size:', x.size)

x = np.array([1, 2, 'World'])
print('dtype:', x.dtype)
print(x)

x = np.array([1, 2.5, 4])
print(x, x.dtype)

x = np.array([1.5, 2.2, 3.7], dtype=np.int64)
print(x)
print('dtype:', x.dtype)

x = np.array([1, 2, 3, 4, 5])
np.save('my_array', x) # creates my_array.npy
np.savez('my_array.npz', a=x) # creates my_array.npz
y = np.load('my_array.npy')
z = np.load('my_array.npz')
print(y)
print(z)
print(z['a'])
