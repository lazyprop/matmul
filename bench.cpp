#include <iostream>
#include <iomanip>
#include <omp.h>

#include "matmul.h"
#include "goto.h"
#include "layered.h"
#include "util.h"
#include "blis.h"

int main() {
  const int N = 1920;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<N>(a);
  //zero_matrix<N>(b);

  test_program<N>("baseline", baseline<N>, a, b, ans, ans);

  //test_program<N>("goto2", goto2<N>, a, b, c, ans);
  //test_program<N>("goto3", goto3<N>, a, b, c, ans);
  test_program<N>("layered", gemm<N>, a, b, c, ans);
  test_program<N>("layered2", gemm2<N>, a, b, c, ans);
  test_program<N>("blis", blis<N>, a, b, c, ans);

  transpose_matrix<N>(b);
  test_program<N>("parallel_tranposed_simd",
                         parallel_tranposed_simd<N>, a, b, c, ans);
  transpose_matrix<N>(b);

  return 0;
}
