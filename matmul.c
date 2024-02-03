#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

#define DEBUG

#define DTYPE float

#ifdef DEBUG
const int N = 3, M = 3, O = 3;
#else
const int N = 1024, M = 1024, O = 1024;
#endif

void print_matrix(DTYPE* mat, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%.1f ", mat[i*n+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void baseline(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < O; k++) {
        c[i*N+k] += a[i*N+j] * b[j*M+k];
      }
    }
  }
}

void transposed(DTYPE* a, DTYPE* _b, DTYPE* c) {
  print_matrix(_b, N, N);
  int* b = malloc(sizeof(DTYPE) * N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      b[i*N+j] = _b[j*N+i];
    }
  }

  // TODO: why is B zero?
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < O; k++) {
      for (int j = 0; j < M; j++) {
        c[i*N+k] += a[i*N+j] * b[k*N+j];
        printf("%1f\n", b[k*N+j]);
      }
    }
  }
  free(b);
}

void simd(DTYPE* a, DTYPE* b, DTYPE* c) {
}



void rand_matrix(DTYPE* mat, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      mat[i*n+j] = (DTYPE) rand() / (DTYPE) RAND_MAX;
    }
  }
}

void zero_matrix(DTYPE* mat, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      mat[i*n+j] = 0;
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

  rand_matrix(a, N, M);
  rand_matrix(b, M, O);

  #ifdef DEBUG
  print_matrix(a, N, N);
  print_matrix(b, N, N);
  #endif

  clock_t begin, end;

  begin = clock();
  baseline(a, b, ans);
  end = clock();
  printf("baseline: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);

  begin = clock();
  transposed(a, b, c);
  end = clock();
  printf("transposed: %f s\n", (double) (end - begin) / CLOCKS_PER_SEC);
  #ifdef DEBUG
  printf("transposed output:\n");
  print_matrix(c, N, N);
  #endif
  assert(check_matrix(c, ans));
  zero_matrix(c, N, O);


  #ifdef DEBUG
  print_matrix(a, N, M);
  print_matrix(b, M, O);
  print_matrix(ans, N, O);
  #endif

  return 0;
}

