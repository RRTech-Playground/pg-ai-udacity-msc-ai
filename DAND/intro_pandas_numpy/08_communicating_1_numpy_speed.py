import numpy as np
import time

a = np.random.randn(int(1e8))

start = time.time()
sum(a) / len(a)
print(time.time() - start, 'seconds')

start = time.time()
np.mean(a)
print(time.time() - start, 'seconds')