#include <iostream>
#include <iomanip>
#include <omp.h>

#include "matmul.h"
#include "goto.h"
#include "layered.h"
#include "util.h"
#include "blis.h"

void bench_1920() {
  const int N = 1920;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<N>(a);
  rand_matrix<N>(b);

  //test_program<N>("baseline", baseline<N>, a, b, ans, ans);
  test_program<N>("blis_12x8", blis_12x8<N, 96, 48, 960>, a, b, c, c);
}

void bench_1024() {
  const int N = 1024;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<N>(a);
  rand_matrix<N>(b);

  test_program<N>("baseline", baseline<N>, a, b, ans, ans);

  test_program<N>("layered", gemm<N>, a, b, c, ans);
  test_program<N>("layered2", gemm2<N>, a, b, c, ans);
  test_program<N>("blis", blis<N, 128, 64, 1024>, a, b, c, ans);
}


int main() {
  std::cout << "N = 1024\n";
  //bench_1024();

  std::cout << "\nN = 1920\n";
  bench_1920();
}
