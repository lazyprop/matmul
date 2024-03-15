#include <x86intrin.h>
#include <omp.h>
#include <stdio.h>

#include "matmul.h"
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

void tiled(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int iblock = 0; iblock < N; iblock += BLOCK_SIZE) {
    for (int kblock = 0; kblock < N; kblock += BLOCK_SIZE) {
      for (int jblock = 0; jblock < N; jblock += BLOCK_SIZE) {
        for (int i = iblock; i < iblock + BLOCK_SIZE; i++) {
          for (int k = kblock; k < kblock + BLOCK_SIZE; k++) {
            for (int j = jblock; j < jblock + BLOCK_SIZE; j++) {
              c[i*N+k] += a[i*N+j] * b[j*N+k];
            }
          }
        }
      }
    }
  }
}

/*
 * compute a 2x2 block of C = AB whose top-left corner is at (x, y)
 * maximize register use by using many accumulators
 * then write to memory in the end
 */

void kernel_2x2(DTYPE* a, DTYPE* b, DTYPE* c, int x, int y) {
  // zero accumulators
  DTYPE c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (int k = 0; k < N; k++) {
    // read the rows and columns
    DTYPE a0 = a[x*N+k], a1 = a[(x+1)*N+k];
    DTYPE b0 = b[k*N+y], b1 = b[k*N+y+1];
    c00 += a0 * b0;
    c01 += a0 * b1;
    c10 += a1 * b0;
    c11 += a1 * b1;
  }
  c[x*N+y] = c00;
  c[x*N+y+1] = c01;
  c[(x+1)*N+y] = c10;
  c[(x+1)*N+y+1] = c11;
}

void blocked_2x2(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int i = 0; i < N; i += 2) {
    for (int j = 0; j < N; j += 2) {
      kernel_2x2(a, b, c, i, j);
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



void parallel(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int hblock = 0; hblock < N; hblock += BLOCK_SIZE) {
    for (int vblock = 0; vblock < N; vblock += BLOCK_SIZE) {
#pragma omp parallel for collapse(2)
      for (int row = vblock; row < vblock + BLOCK_SIZE; row++) {
        for (int col = hblock; col < hblock + BLOCK_SIZE; col++) {
          for (int k = 0; k < N; k++) {
            c[row*N+col] += a[row*N+k] * b[k*N+col];
          }
        }
      }
    }
  }
}

void parallel_simd(DTYPE* a, DTYPE* b, DTYPE* c) {
  for (int hblock = 0; hblock < N; hblock += BLOCK_SIZE) {
    for (int vblock = 0; vblock < N; vblock += BLOCK_SIZE) {
#pragma omp parallel for collapse(2)
      for (int row = vblock; row < vblock + BLOCK_SIZE; row++) {
        for (int col = hblock; col < hblock + BLOCK_SIZE; col++) {
          __m256 ans = _mm256_setzero_ps();
          for (int k = 0; k < N; k += 8) {
            __m256 x = _mm256_loadu_ps(&a[row*N+k]);
            __m256 y = _mm256_loadu_ps(&b[col*N+k]);
            ans = _mm256_fmadd_ps(x, y, ans);
          }
          DTYPE vec[8];
          _mm256_storeu_ps(vec, ans);
          for (int x = 0; x < 8; x++) {
            c[row*N+col] += vec[x];
          }
        }
      }
    }
  }
}
