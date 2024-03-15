#include <iostream>
#include <iomanip>
#include <omp.h>

#include "matmul.h"
#include "util.h"

int main() {
  const int N = 1024;

  float* a = (float*) malloc(sizeof(float) * N * N);
  float* b = (float*) malloc(sizeof(float) * N * N);
  float* c = (float*) malloc(sizeof(float) * N * N);
  float* ans = (float*) malloc(sizeof(float) * N * N);

  rand_matrix<float, N>(a);
  rand_matrix<float, N>(b);

  #ifdef DEBUG
  std::cout << "a:\n";
  print_matrix(a);
  std::cout << "b:\n";
  print_matrix(b);
  #endif

  std::cout << std::fixed << std::setprecision(2);

  double begin = omp_get_wtime();
  baseline<float, N>(a, b, ans);
  std::cout << "baseline: " << time_to_gflops_s<N>(omp_get_wtime() - begin)
            << " GFLOPS/s\n";
  #ifdef DEBUG
  std::cout << "answer:\n";
  print_matrix<float, N>(ans);
  #endif

  zero_matrix<float, N>(c);
  transpose_matrix<float, N>(b);
  test_program<float, N>("transposed", transposed<float, N>, a, b, c, ans);
  test_program<float, N>("simd", simd<float, N>, a, b, c, ans);
  transpose_matrix<float, N>(b);

  test_program<float, N>("blocked_2x2", blocked_2x2<float, N>, a, b, c, ans);
  test_program<float, N>("blocked_8x8", blocked<float, N, 8>, a, b, c, ans);
  test_program<float, N>("blocked_16x16", blocked<float, N, 16>, a, b, c, ans);


  test_program<float, N>("parallel", parallel<float, N>, a, b, c, ans);
  transpose_matrix<float, N>(b);
  test_program<float, N>("parallel_simd", parallel_simd<float, N>, a, b, c, ans);

  return 0;
}
