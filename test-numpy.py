import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import numpy as np


N = 1920
x = np.random.randn(N, N).astype(np.float32)
y = np.random.randn(N, N).astype(np.float32)

seconds = 0
runs = 100
for _ in range(runs):
    start = time.time_ns()
    z = np.dot(x, y)
    seconds += time.time_ns() - start

seconds /= 1e9

avg = seconds / runs
gflops = 2 * (N**3) / 1e9 / avg

print(f'average time per run: {avg}')
print(f'gflops: {gflops}')


