#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <time.h>

#define DTYPE float
#define DEBUG

#ifdef DEBUG
#define N 16
#else
#define N 268435456
#endif


void vecadd(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void vecadd_simd(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i += 8) {
    __m256 x = _mm256_loadu_ps(&a[i]);
    __m256 y = _mm256_loadu_ps(&b[i]);
    __m256 z = _mm256_add_ps(x, y);
    _mm256_storeu_ps(&c[i], z);
  }
}

DTYPE vecsum(DTYPE* a) {
  DTYPE s = 0;
  for (int i = 0; i < N; i++) s += a[i];
  return s;
}

DTYPE vecsum_simd(DTYPE* a) {
  __m256 vec;
  for (int i = 0; i < N; i += 8) {
    __m256 x = _mm256_loadu_ps(&a[i]);
    vec = _mm256_add_ps(x, vec);
  }
  DTYPE v[8], ans = 0;
  _mm256_storeu_ps(v, vec);
  for (int i = 0; i < 8; i++) ans += v[i];
  return ans;
}

void print_arr(DTYPE* a) {
  for (int i = 0; i < N; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");
}

int main() {
  DTYPE* a = malloc(sizeof(DTYPE) * N);
  DTYPE* b = malloc(sizeof(DTYPE) * N);
  DTYPE* c = malloc(sizeof(DTYPE) * N);
  DTYPE* d = malloc(sizeof(DTYPE) * N);

  for (int i = 0; i < N; i++) {
    a[i] = (DTYPE) i+1;
    b[N-i-1] = (DTYPE) i;
    c[i] = (DTYPE) 0;
    d[i] = (DTYPE) 0;
  }

  #ifdef DEBUG
  print_arr(a);
  print_arr(b);
  #endif

  clock_t begin;
  
  begin = clock();
  vecadd(a, b, c);
  printf("vecadd baseline: %f s\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

  begin = clock();
  vecadd_simd(a, b, d);
  printf("vecadd simd: %f s\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

  #ifdef DEBUG
  print_arr(c);
  print_arr(d);
  #endif

  free(b);
  free(c);
  free(d);

  #ifdef DEBUG
  for (int i = 0; i < N; i++) {
    if (c[i] != d[i]) {
      printf("wrong\n");
      return 0;
    }
  }
  printf("correct\n");
  #endif

  printf("\n");
  
  begin = clock();
  DTYPE sum = vecsum(a);
  printf("vecsum baseline: %f s\n", (double) (clock() - begin) / CLOCKS_PER_SEC);

  begin = clock();
  DTYPE _sum = vecsum_simd(a);
  printf("vecsum simd: %f s\n", (double) (clock() - begin) / CLOCKS_PER_SEC);
  
  printf("difference in sum: %.1f\n", abs(sum - _sum));
}
