#include <iostream>
#include <memory>
#include <chrono>

#include "util.h"
#include "matmul.h"

#define CHECK_ERROR() \
  { \
    cudaError_t e = cudaGetLastError(); \
    std::cout << cudaGetErrorName(e) << '\n'; \
    std::cout << cudaGetErrorString(e) << '\n'; \
  }

#define BENCH_KERNEL(name, N, kernel, grid_dim, block_dim, d_a, d_b, d_c)  \
  { \
    const int runs = 100; \
    double ms = 0; \
    for (int i = 0; i < runs; i++) { \
      zero_cuda<N>(d_c); \
      cudaDeviceSynchronize(); \
      auto start = std::chrono::high_resolution_clock::now(); \
      kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c);                     \
      cudaDeviceSynchronize(); \
      auto end = std::chrono::high_resolution_clock::now();            \
      std::chrono::duration<double, std::milli> ms_double = end - start; \
      ms += ms_double.count();                                     \
    } \
    std::cout << name << ": " << (2.0 * N * N * N) / (1e9 * ms / runs / 1000) \
              << " GFLOPS/s\n"; \
  }


#define DEBUG_KERNEL(name, N, kernel, grid_dim, block_dim, d_a, d_b, d_c)  \
  { \
    kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c);                     \
    cudaDeviceSynchronize(); \
    CHECK_ERROR(); \
  }




template<int N>
__global__ void baseline_cuda(float* a, float* b, float* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  float acc = 0;
  for (int k = 0; k < N; k++) {
    acc += a[i*N+k] * b[k*N+j];
  }
  c[i*N+j] = acc;
}


template<int N, int Mc>
__global__ void gmem_coalesced(float* a, float* b, float* c) {
  int i = blockIdx.x * Mc + (threadIdx.x / Mc);
  int j = blockIdx.y * Mc + (threadIdx.x % Mc);
  float acc = 0;
  for (int k = 0; k < N; k++) {
    acc += a[i*N+k] * b[k*N+j];
  }
  c[i*N+j] = acc;
}

template<int N, int Mc>
__global__ void smem_blocked(float* a, float* b, float* c) {
  int i = blockIdx.x * Mc;
  int j = blockIdx.y * Mc;
  int ii = threadIdx.x / Mc;
  int jj = threadIdx.x % Mc;

  __shared__ float aa[Mc][Mc], bb[Mc][Mc], cc[Mc][Mc];
  cc[ii][jj] = 0;
  for (int k = 0; k < N; k += Mc) {
    // abusing the index notation here
    // basically each thread loads one Mc x Mc element of a and b
    aa[ii][jj] = a[(i+ii)*N+k+jj];
    bb[ii][jj] = b[(k+ii)*N+j+jj];

    __syncthreads();
    for (int kk = 0; kk < Mc; kk++) {
      cc[ii][jj] += aa[ii][kk] * bb[kk][jj];
    }
    __syncthreads();
  }

  c[(i+ii)*N+j+jj] = cc[ii][jj];
}

template<int N, int Mc>
__global__ void smem_blocked2(float* a, float* b, float* c) {
  int i = blockIdx.x * Mc;
  int j = blockIdx.y * Mc;
  int ii = threadIdx.x / Mc;
  int jj = threadIdx.x % Mc;

  __shared__ float aa[Mc][Mc], bb[Mc][Mc];
  float acc = 0;
  for (int k = 0; k < N; k += Mc) {
    // abusing the index notation here
    // basically each thread loads one Mc x Mc element of a and b
    aa[ii][jj] = a[(i+ii)*N+k+jj];
    bb[ii][jj] = b[(k+ii)*N+j+jj];

    __syncthreads();
    for (int kk = 0; kk < Mc; kk++) {
      acc += aa[ii][kk] * bb[kk][jj];
    }
    __syncthreads();
  }

  c[(i+ii)*N+j+jj] = acc;
}


template <int N, int Mc>
__global__ void thread_blocked(float* a, float* b, float* c) {
  const int TM = Mc / 32;
  const int i = blockIdx.x * Mc;
  const int j = blockIdx.y * Mc;
  const int ii = (threadIdx.x / 32) * TM;
  const int jj = (threadIdx.x % 32) * TM;

  __shared__ float aa[Mc][Mc], bb[Mc][Mc];
  float cc[TM][TM] = {};

  for (int k = 0; k < N; k += Mc) {

    for (int iii = 0; iii < TM; iii++) {
      for (int jjj = 0; jjj < TM; jjj++) {
        aa[ii+iii][jj+jjj] = a[(i+ii+iii)*N+k+jj+jjj];
      }
    }

    for (int iii = 0; iii < TM; iii++) {
      for (int jjj = 0; jjj < TM; jjj++) {
        bb[ii+iii][jj+jjj] = b[(k+ii+iii)*N+j+jj+jjj];
      }
    }
    __syncthreads();

    for (int iii = 0; iii < TM; iii++) {
      for (int jjj = 0; jjj < TM; jjj++) {
        for (int kk = 0; kk < Mc; kk++) {
          cc[iii][jjj] += aa[ii+iii][kk] * bb[kk][jj+jjj];
        }
      }
    }
    __syncthreads();

  }

  for (int iii = 0; iii < TM; iii++) {
    for (int jjj = 0; jjj < TM; jjj++) {
      c[(i+ii+iii)*N+j+jj+jjj] = cc[iii][jjj];
    }
  }
}


template <int N, int Mc>
__global__ void thread_blocked2(float* a, float* b, float* c) {
  const int TM = Mc / 32;
  const int i = blockIdx.x * Mc;
  const int j = blockIdx.y * Mc;
  const int ii = threadIdx.x / 32;
  const int jj = threadIdx.x % 32;

  __shared__ float aa[Mc][Mc], bb[Mc][Mc];
  float cc[TM][TM] = {};

  for (int k = 0; k < N; k += Mc) {

    for (int rowb = 0; rowb < TM; rowb++) {
      for (int colb = 0; colb < TM; colb++) {
        aa[rowb*32+ii][colb*32+jj] = a[(i+(rowb*32)+ii) * N + k+(colb*32)+jj];
      }
    }

    for (int rowb = 0; rowb < TM; rowb++) {
      for (int colb = 0; colb < TM; colb++) {
        bb[rowb*32+ii][colb*32+jj] = b[(k+(rowb*32)+ii) * N + j+(colb*32)+jj];
      }
    }
    __syncthreads();

    for (int rowb = 0; rowb < TM; rowb++) {
      for (int colb = 0; colb < TM; colb++) {
        for (int kk = 0; kk < Mc; kk++) {
          cc[rowb][colb] += aa[rowb*32 + ii][kk] * bb[kk][colb*32+jj];
        }
      }
    }
    __syncthreads();

  }

  for (int rowb = 0; rowb < TM; rowb++) {
    for (int colb = 0; colb < TM; colb++) {
      c[(i+(rowb*32)+ii) * N + j + (colb * 32) + jj] = cc[rowb][colb];
    }
  }
  
}



int main() {
  const int N = 2048;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<N>(a);
  rand_matrix<N>(b);
  zero_matrix<N>(c);
  zero_matrix<N>(ans);

  baseline<N>(a, b, ans);
  //print_matrix<N>(ans);

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

  BENCH_KERNEL("baseline_cuda", N, baseline_cuda<N>,
               dim3(N / 32, N / 32), dim3(32, 32), d_a, d_b, d_c);
  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
  check_matrix<N>(c, ans);

  BENCH_KERNEL("gmem_coalesced", N, (gmem_coalesced<N, 32>),
               dim3(N / 32, N / 32), dim3(32 * 32), d_a, d_b, d_c);
  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
  check_matrix<N>(c, ans);

  BENCH_KERNEL("smem_blocked", N, (smem_blocked<N, 32>),
               dim3(N / 32, N / 32), dim3(32 * 32), d_a, d_b, d_c);
  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
  check_matrix<N>(c, ans);

  BENCH_KERNEL("smem_blocked2", N, (smem_blocked2<N, 32>),
               dim3(N / 32, N / 32), dim3(32 * 32), d_a, d_b, d_c);
  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
  check_matrix<N>(c, ans);


  zero_matrix<N>(c);
  BENCH_KERNEL("thread_blocked", N, (thread_blocked<N, 64>),
               dim3(N/64, N/64), dim3(1024), d_a, d_b, d_c);
  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
  check_matrix<N>(c, ans);

  zero_matrix<N>(c);
  BENCH_KERNEL("thread_blocked2", N, (thread_blocked2<N, 64>),
               dim3(N/64, N/64), dim3(1024), d_a, d_b, d_c);
  cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
  //print_matrix<N>(c);
  check_matrix<N>(c, ans);



  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
