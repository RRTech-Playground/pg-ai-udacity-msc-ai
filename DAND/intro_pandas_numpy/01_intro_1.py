counts = [5, 2, 6, 3, 4]
mean = sum(counts) / len(counts)
print(mean)


counts = [5, 2, 6, 3, 4]
def compute_mean(counts):
    return sum(counts) / len(counts)

mean = compute_mean(counts)
print(mean)


import numpy as np

counts = [5, 2, 6, 3, 4]
mean = np.mean(counts)
logs = np.log(counts)
cosines = np.cos(counts)

print(mean)
print(logs)
print(cosines)