#include <cstdlib>
#include <immintrin.h>

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
  std::cout << "B:\n";
  print_matrix<float, N>(b);


  /*
  alignas(32) float bx[8*8];
  __m256 bv[8];
  pack_8x8<float, N>(bv, b, 0, 0);
  unpack_8x8<float, N>(bx, bv, 0, 0);
  std::cout << bx[63] << '\n';
  print_matrix<float, 8>(bx);
  std::cout << '\n';
  */
  
  blocked_2x2<float, N>(a, b, ans);
  std::cout << "computed blocked 2x2\nans:\n";
  print_matrix<float, N>(ans);

  //test_program<float, N>("blocked3: ", blocked3<float, N, 8>, a, b, c, ans);
  blocked3<float, N, 8>(a, b, c);

  std::cout << "c:\n";
  print_matrix<float, N>(c);

  std::cout << check_matrix<float, N>(c, ans);

  return 0;
}


int test() {
  const int N = 16;
  float* mat = static_cast<float*>(std::aligned_alloc(64, sizeof(float)*N*N));
  rand_matrix<float, N>(mat);
  print_matrix<float, N>(mat);

  __m256 rows[8];
  /*
  for (int k = 0; k < N; k++) {
    rows[k] = _mm256_load_ps(&mat[k*N]);
  }
  */
  pack_8x8<float, N>(rows, mat, 0, 0);

  float* mat2 = static_cast<float*>(std::aligned_alloc(64, sizeof(float)*N*N));
  /*
  for (int k = 0; k < 8; k++) {
    _mm256_store_ps(&mat2[k*8], rows[k]);
  }
  */
  unpack_8x8<float, N>(mat2, rows, 0, 0);

  print_matrix<float, 8>(mat2);

  return 0;
}
