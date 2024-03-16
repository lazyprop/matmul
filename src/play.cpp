#include <cstdlib>

#include "matmul.h"
#include "util.h"

int main() {
  const int N = 16;

  float* a = static_cast<float*>(std::aligned_alloc(64, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(64, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(64, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(64, sizeof(float) * N * N));

  rand_matrix<float, N>(a);
  rand_matrix<float, N>(b);
  zero_matrix<float, N>(c);
  zero_matrix<float, N>(ans);

  std::cout << "A:\n";
  print_matrix<float, N>(a);

  blocked_2x2<float, N>(a, b, ans);
  std::cout << "computed blocked 2x2\nans:\n";
  //print_matrix<float, N>(ans);

  test_program<float, N>("blocked3: ", blocked3<float, N, 8>, a, b, c, ans);
  //blocked3<float, N, 8>(a, b, c);

  std::cout << "c:\n";
  //print_matrix<float, N>(c);

  return 0;
}
