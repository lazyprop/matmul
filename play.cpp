#include <cstdlib>
#include <immintrin.h>

#include "util.h"
#include "layered.h"
#include "blis.h"

int main() {
  const int N = 1024;
  //bench();
  //return 0;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  //seq_init<N>(a);
  //seq_init<N>(b);
  rand_matrix<N>(a);
  //rand_matrix<N>(b);
  zero_matrix<N>(c);
  zero_matrix<N>(ans);

  baseline<N>(a, b, ans);
  //for (int i = 0; i < 100; i++) {
  //blis<N>(a, b, c);
  //}

  //print_matrix<N>(a);
  //print_matrix<N>(b);
  //print_matrix<N>(ans);



  //test_program<N>("layered", gemm<N>, a, b, c, ans);
  //test_program<N>("layered2", gemm2<N>, a, b, c, ans);
  test_program<N>("baseline", baseline<N>, a, b, c, ans);
  test_program<N>("blis", blis<N>, a, b, c, ans);
}
