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

  rand_matrix(a);
  rand_matrix(b);

  ans = simd(a, b, ans);

  return 0;
}
