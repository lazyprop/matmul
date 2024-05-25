#include <iostream>
#include <memory>
#include <chrono>

#include "util.h"
#include "matmul.h"


#define BENCH_KERNEL(name, N, kernel, grid_dim, block_dim, a, b, c)  \
  { \
    const int runs = 10; \
    double seconds = 0; \
    for (int i = 0; i < runs; i++) { \
      zero_cuda<N>(c);  \
      auto start = std::chrono::high_resolution_clock::now(); \
      kernel<<<grid_dim, block_dim>>>(a, b, c);                     \
      auto end = std::chrono::high_resolution_clock::now();            \
      std::chrono::duration<double, std::milli> ms_double = end - start; \
      seconds += ms_double.count();                                     \
    } \
    std::cout << name << ": " << (2.0 * N * N * N) / (seconds / runs / 1000 * 1e9) \
              << " GFLOPS/s\n"; \
  }



template<int N, int Mc>
__global__ void baseline_cuda_kernel(float* a, float* b, float* c) {
  int i = blockIdx.x * N + threadIdx.x;
  int j = blockIdx.y * N + threadIdx.y;
  float acc = 0;
  for (int k = 0; k < N; k++) {
    acc += a[i*N+j] * b[k*N+j];
  }
  c[i*N+j] = acc;
}

template <int N>
void baseline_cuda(float* a, float* b, float* c) {
  float* d_a;
  float* d_b;
  float* d_c;

  int matsize = N * N * sizeof(float);

  cudaMalloc(&d_a, matsize);
  cudaMalloc(&d_b, matsize);
  cudaMalloc(&d_c, matsize);
  
  cudaMemcpy(d_a, a, matsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, matsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, matsize, cudaMemcpyHostToDevice);

  dim3 block_dim(32, 32, 1);
  dim3 grid_dim(N / 32, N / 32, 1);

  BENCH_KERNEL("baseline_cuda", N, (baseline_cuda_kernel<N, 32>), grid_dim, block_dim,
               d_a, d_b, d_c);

  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
}


int main() {
  const int N = 256;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  //seq_init<N>(a);
  //seq_init<N>(b);
  rand_matrix<N>(a);
  rand_matrix<N>(b);
  zero_matrix<N>(c);
  zero_matrix<N>(ans);

  baseline<N>(a, b, ans);

  baseline_cuda<N>(a, b, c);
}