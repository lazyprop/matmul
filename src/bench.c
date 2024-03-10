#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

#include "matmul.h"
#include "util.h"

int main() {
  DTYPE* a = malloc(sizeof(DTYPE) * N * N);
  DTYPE* b = malloc(sizeof(DTYPE) * N * N);
  DTYPE* c = malloc(sizeof(DTYPE) * N * N);
  DTYPE* ans = malloc(sizeof(DTYPE) * N * N);

  rand_matrix(a);
  rand_matrix(b);

  #ifdef DEBUG
  printf("a:\n");
  print_matrix(a);
  printf("b:\n");
  print_matrix(b);
  #endif

  double begin = omp_get_wtime();
  baseline(a, b, ans);
  printf("baseline: %.0f ms\n", 1000 * (omp_get_wtime() - begin));
  #ifdef DEBUG
  printf("answer:\n");
  print_matrix(ans);
  #endif

  zero_matrix(c);
  transpose_matrix(b);
  test_program("transposed", transposed, a, b, c, ans);
  test_program("simd", simd, a, b, c, ans);

  transpose_matrix(b);
  test_program("tiled", tiled, a, b, c, ans);
  test_program("parallel", parallel, a, b, c, ans);
  transpose_matrix(b);

  test_program("parallel_simd", parallel_simd, a, b, c, ans);

  return 0;
}
