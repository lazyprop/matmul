#include <stdlib.h>

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
  zero_matrix<float, N>(c);
  zero_matrix<float, N>(ans);

  blocked_2x2<float, N>(a, b, ans);

  test_program<float, N>("blocked 2", blocked<float, N, 2>, a, b, c, ans);
  test_program<float, N>("blocked 4", blocked<float, N, 4>, a, b, c, ans);
  test_program<float, N>("blocked 8", blocked<float, N, 8>, a, b, c, ans);
  test_program<float, N>("blocked 16", blocked<float, N, 16>, a, b, c, ans);
  test_program<float, N>("blocked 32", blocked<float, N, 32>, a, b, c, ans);

  std::cout << "matches!\n";

  return 0;
}
