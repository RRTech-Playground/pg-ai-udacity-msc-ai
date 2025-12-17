import numpy as np

X = np.arange(1, 21).reshape(4, 5)
print(X)

z = X[1:4, 2:4]
print(z)

z = X[:, 2]  # all rows start:end in colum 2
print(z)

z = X[:, 2:3]  # all rows start:end in colum 2
print(z)

z = X[:3, 2:]  # first is the row, second is the column
print(z)

z = np.copy(X[:1, 2:])
print(z)

indices = np.array([1, 3])
print(indices)

y = X[indices, :]
print(y)

y = X[:, indices]
print(y)

z = np.diag(X)
print(z)

z = np.diag(X, k=1)
print(z)

z = np.diag(X, k=-1)
print(z)

X = np.array([[1, 2, 3], [5, 2, 8], [1, 2, 3]])
print(X)

print(np.unique(X))

X = np.arange(25).reshape(5, 5)
print(X)

print(X[X > 10])

print(X[X <= 7])

print(X[(X > 10) & (X < 17)])

X[(X > 10) & (X < 17)] = -1
print(X)

x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 2, 8, 4])

print(np.intersect1d(x, y))

print(np.setdiff1d(x, y))

print(np.union1d(x, y))

x = np.random.randint(1, 11, size=(10,))
print(x)

print(np.sort(x))  # np.sort() doesn't change the underlying array
print(x)

print(np.sort(np.unique(x)))

print(x)
x.sort()
print(x)

X = np.random.randint(1, 11, size=(5, 5))
print(X)

print(np.sort(X, axis=0))  # sort by column

print(np.sort(X, axis=1))  # sort by row