#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <x86intrin.h>
#include <omp.h>

//#define DEBUG

#define DTYPE float
#define ERR 1e-4

#ifdef DEBUG
#define N 16
#define BLOCK_SIZE 4
#else
#define N 1024
#define BLOCK_SIZE 32
#endif

#include "util.h"

void baseline(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i*N+k] += a[i*N+j] * b[j*N+k];
      }
    }
  }
}

void transposed(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        c[i*N+k] += a[i*N+j] * b[k*N+j];
      }
    }
  }
}

void simd(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      __m256 ans = _mm256_setzero_ps();
      for (int j = 0; j < N; j += 8) {
        __m256 x = _mm256_loadu_ps(&a[i*N+j]);
        __m256 y = _mm256_loadu_ps(&b[k*N+j]);
        ans = _mm256_fmadd_ps(x, y, ans);
      }
      DTYPE vec[8];
      _mm256_storeu_ps(vec, ans);
      for (int x = 0; x < 8; x++) {
        c[i*N+k] += vec[x];
      }
    }
  }
}

void blocked(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int hblock = 0; hblock < N; hblock += BLOCK_SIZE) {
    for (int vblock = 0; vblock < N; vblock += BLOCK_SIZE) {
#pragma omp parallel for collapse(2)
      for (int row = vblock; row < vblock + BLOCK_SIZE; row++) {
        for (int col = hblock; col < hblock + BLOCK_SIZE; col++) {
          for (int k = 0; k < N; k++) {
            assert(row >= 0 && row < N);
            assert(col >= 0 && col < N);
            assert(k >= 0 && k < N);
            c[row*N+col] += a[row*N+k] * b[k*N+col];
          }
        }
      }
    }
  }
}


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
  printf("baseline: %.2f s\n", omp_get_wtime() - begin);
  #ifdef DEBUG
  printf("answer:\n");
  print_matrix(ans);
  #endif

  zero_matrix(c);
  transpose_matrix(b);
  test_program("transposed", transposed, a, b, c, ans);
  test_program("simd", simd, a, b, c, ans);

  transpose_matrix(b);
  test_program("blocked", blocked, a, b, c, ans);
  transpose_matrix(b);

  return 0;
}
