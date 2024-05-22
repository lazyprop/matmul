#include <cstdlib>
#include <immintrin.h>

#include "util.h"
#include "layered.h"
#include "blis.h"

int main() {
  const int N = 1920;

  //bench();
  //return 0;

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
  test_program<N>("blis_12x8", blis_12x8<N, 96, 48, 960>, a, b, c, ans);
}
