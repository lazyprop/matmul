import time
import numpy as np

n = 1024

a = np.random.rand(n, n)
b = np.random.rand(n, n)

start = time.time()
c = a @ b
duration = time.time() - start

print(2*(n**3) / (duration * 1e9), 'GFLOPs/s')
