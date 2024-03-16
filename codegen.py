for k in range(8):
  print(f'_mm256_store_ps(&to_ptr[{k}*N], from[{k}]);')

for k in range(8):
    print(f'to[{k}] = _mm256_load_ps(&from_ptr[{k}*N]);')

for k in range(8):
    print(f'cv[{k}] = _mm256_setzero_ps();')
