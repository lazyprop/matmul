#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

//#define DEBUG

#define DTYPE float

#ifdef DEBUG
#define N 3
#else
#define N 1024
#endif

void transpose_matrix(DTYPE* mat);
void print_matrix(DTYPE* mat);
void rand_matrix(DTYPE* mat);
void zero_matrix(DTYPE* mat);

void print_matrix(DTYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%.1f ", mat[i*N+j]);
    }
    printf("\n");
  }
  printf("\n");
}

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
  transpose_matrix(b);
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        c[i*N+k] += a[i*N+j] * b[k*N+j];
      }
    }
  }
}

void transpose_matrix(DTYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      DTYPE tmp = mat[j*N+i];
      mat[j*N+i] = mat[i*N+j];
      mat[i*N+j] = tmp;
    }
  }
}

void rand_matrix(DTYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N+j] = (DTYPE) rand() / (DTYPE) RAND_MAX;
    }
  }
}

void zero_matrix(DTYPE* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N+j] = 0;
    }
  }
}

bool check_matrix(DTYPE* mat, DTYPE* ans) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (mat[i*N+j] != ans[i*N+j]) return false;
    }
  }
  return true;
}

int main() {
  DTYPE* a = malloc(sizeof(DTYPE) * N * N);
  DTYPE* b = malloc(sizeof(DTYPE) * N * N);
  DTYPE* c = malloc(sizeof(DTYPE) * N * N);
  DTYPE* ans = malloc(sizeof(DTYPE) * N * N);

  rand_matrix(a);
  rand_matrix(b);

  #ifdef DEBUG
  print_matrix(a);
  print_matrix(b);
  #endif

  clock_t begin;

  begin = clock();
  baseline(a, b, ans);
  printf("baseline: %f s\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

  begin = clock();
  transposed(a, b, c);
  printf("transposed: %f s\n", (double) (clock() - begin) / CLOCKS_PER_SEC);
  #ifdef DEBUG
  print_matrix(c);
  #endif
  assert(check_matrix(c, ans));
  zero_matrix(c);


  #ifdef DEBUG
  printf("answer:\n");
  print_matrix(ans);
  #endif

  return 0;
}
