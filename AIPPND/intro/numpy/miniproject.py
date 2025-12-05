import numpy as np

# Normalize the date
X = np.random.randint(0, 5001, size=(1000, 20))
print(X)

ave_cols = np.mean(X, axis=0)
std_cols = np.std(X, axis=0)

print(ave_cols.shape)
print(std_cols.shape)

X_norm = (X - ave_cols) / std_cols
print(X_norm)

print("The average of all the values of X_norm is: ")
print(np.mean(X_norm))
print(X_norm.mean())

print("The average of the minimum value in each column of X_norm is: ")
print(X_norm.min(axis = 0).mean())
print(np.mean(np.sort(X_norm, axis=0)[0]))

print("The average of the maximum value in each column of X_norm is: ")
print(np.mean(np.sort(X_norm, axis=0)[-1]))
print(X_norm.max(axis = 0).mean())

# Create the 3 data sets
print(np.random.permutation(5))
row_indices = np.random.permutation(X_norm.shape[0])

# You have to extract the number of rows in each set using row_indices.
# Note that the row_indices are random integers in a 1-D array.
# Hence, if you use row_indices for slicing, it will NOT give the correct result.

# Let's get the count of 60% and 80% of the rows. Since, len(X_norm) has a lenght 1000, therefore, 60% = 600
sixty = int(len(X_norm) * 0.6)  # 60% of the data
eighty = int(len(X_norm) * 0.8)  # 80% of the data

# Here row_indices[:sixty] will give you first 600 values, e.g., [93 255 976 505 281 292 977,.....]
# Those 600 values will will be random, because row_indices is a 1-D array of random integers.
# Next, extract all rows represented by these 600 indices, as
X_train = X_norm[row_indices[:sixty], :]
X_crossVal = X_norm[row_indices[sixty: eighty], :]
X_test = X_norm[row_indices[eighty: ], :]

print(X_train.shape)
print(X_crossVal.shape)
print(X_test.shape)

