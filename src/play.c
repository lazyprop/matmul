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
  zero_matrix(c);
  zero_matrix(ans);

  baseline(a, b, ans);

  test_program("baseline", baseline, a, b, c, ans);
  test_program("blocked", blocked, a, b, c, ans);

  /*j
  print_matrix(ans);
  printf("\n");
  print_matrix(c);
  */

  return 0;
}
