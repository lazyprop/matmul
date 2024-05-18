#include <iostream>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <cmath>

const int N = 1024;

double time_to_gflops_s(const double seconds) {
  double total_flops = 2.0 * N * N * N;
  double gflops_second = total_flops / (seconds * 1e9);
  return gflops_second;
}

void rand_matrix(float* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N+j] = (float) rand() / (float) RAND_MAX;
    }
  }
}

void zero_matrix(float* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N+j] = 0;
    }
  }
}

void baseline(float* a, float* b, float* c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i*N+k] += a[i*N+j] * b[j*N+k];
      }
    }
  }
}


void random_init() {
  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix(a);
  rand_matrix(b);
  zero_matrix(c);

  const int runs = 100;

  long double sum1=0;
  for(int i=0; i<runs; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    baseline(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    sum1 += ms_double.count();
    for(int i=0; i<N*N; i++) c[i] = 0.f;
  }
  std::cout << "random init: " << ((2.f*N*N*N)/((sum1/runs)/1000))/1e9 << " GFLOPS" << std::endl;
}

void no_random_init() {
  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix(a);
  //rand_matrix(b);
  zero_matrix(c);

  const int runs = 100;

  long double sum1=0;
  for(int i=0; i<runs; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    baseline(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    sum1 += ms_double.count();
    for(int i=0; i<N*N; i++) c[i] = 0.f;
  }
  std::cout << "no random init: " << ((2.f*N*N*N)/((sum1/runs)/1000))/1e9 << " GFLOPS" << std::endl;
}

int main() {
  random_init();
  no_random_init();
}
