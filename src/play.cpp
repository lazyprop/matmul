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
  const int N = 1024;

  float* a = (float*) malloc(sizeof(float) * N * N);
  float* b = (float*) malloc(sizeof(float) * N * N);
  float* ans = (float*) malloc(sizeof(float) * N * N);

  rand_matrix<float, N>(a);
  rand_matrix<float, N>(b);
  zero_matrix<float, N>(ans);

  baseline<float, N>(a, b, ans);

  /*j
  print_matrix(ans);
  printf("\n");
  print_matrix(c);
  */

  return 0;
}
