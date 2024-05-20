#ifndef MAfloatMUL_H
#define MAfloatMUL_H

#include <x86intrin.h>
#include <omp.h>

#include "util.h"

const int BLOCK_SIZE = 32;

template<size_t N>
void baseline(float* a, float* b, float* c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i*N+k] += a[i*N+j] * b[j*N+k];
      }
    }
  }
}

template<size_t N>
void transposed(float* a, float* b, float* c) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        c[i*N+k] += a[i*N+j] * b[k*N+j];
      }
    }
  }
}

template<size_t N>
void tiled(float* a, float* b, float* c) {
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

template<size_t N>
void kernel_2x2(float* a, float* b, float* c, int x, int y) {
  // zero accumulators
  float c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (int k = 0; k < N; k++) {
    // read the rows and columns
    float a0 = a[x*N+k], a1 = a[(x+1)*N+k];
    float b0 = b[k*N+y], b1 = b[k*N+y+1];
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


template<size_t N, size_t B>
void kernel(float* a, float* b, float* c, int x, int y) {
  float cx[B][B] = {};
  for (int k = 0; k < N; k++) {
    float ax[B], bx[B];
    for (int i = 0; i < B; i++) ax[i] = a[(x+i)*N+k];
    for (int j = 0; j < B; j++) bx[j] = b[k*N+y+j];
    for (int i = 0; i < B; i++) {
      for (int j = 0; j < B; j++) {
        cx[i][j] += ax[i] * bx[j];
      }
    }
  }
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      c[(x+i)*N+y+j] = cx[i][j];
    }
  }
}

template<size_t N, size_t B>
void blocked(float* a, float* b, float* c) {
  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      kernel<N, B>(a, b, c, i, j);
    }
  }
}


/*
 * Pack a BxB submatrix `from` at (x, y) to a contiguous array `to`
 */
template<size_t N, size_t B>
inline void pack(float* to, float* from, int x, int y) {
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      to[i*B+j] = from[(x+i)*N+(y+j)];
    }
  }
}

/*
 * floatranspose and pack BxB submatrix `from` at (x, y) to a contiguous array `to`
 */
template<size_t N, size_t B>
inline void pack_transpose(float* to, float* from, int x, int y) {
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      to[j*B+i] = from[(x+i)*N+(y+j)];
    }
  }
}

template<size_t N, size_t B>
void kernel2(float* a, float* b, float* c, int x, int y) {
  float ax[B*B], bx[B*B], cx[B*B] = {};
  for (int zz = 0; zz < N; zz += B) {
    pack<N, B>(ax, a, x, zz);
    pack_transpose<N, B>(bx, b, zz, y);
    for (int i = 0; i < B; i++) {
      for (int k = 0; k < B; k++) {
        for (int j = 0; j < B; j++) {
          cx[i*B+j] += ax[i*B+k] * bx[j*B+k];
        }
      }
    }
  }
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      c[(x+i)*N+(y+j)] = cx[i*B+j];
    }
  }
}

template<size_t N, size_t B>
void blocked2(float* a, float* b, float* c) {
  #pragma o
  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      kernel2<N, B>(a, b, c, i, j);
    }
  }
}

/*
 * Unpack an array of __m256 `from` to 8x8 submatrix at `to[x][y]`
 * N is the side length of `to`
 */
template<size_t N>
inline void unpack_from_vecs(float* to, __m256* from, int x, int y) {
  float* to_ptr = &to[x*N+y];
  _mm256_store_ps(&to_ptr[0*N], from[0]);
  _mm256_store_ps(&to_ptr[1*N], from[1]);
  _mm256_store_ps(&to_ptr[2*N], from[2]);
  _mm256_store_ps(&to_ptr[3*N], from[3]);
  _mm256_store_ps(&to_ptr[4*N], from[4]);
  _mm256_store_ps(&to_ptr[5*N], from[5]);
  _mm256_store_ps(&to_ptr[6*N], from[6]);
  _mm256_store_ps(&to_ptr[7*N], from[7]);
}

/*
 * Pack a 8x8 submatrix `from` at (x, y) to an array of _m256
 * N is the side length of `from`
 */
template<size_t N>
inline void pack_into_vecs(__m256* to, float* _from, int x, int y) {
  float from[8*8];
  pack<N, 8>(from, _from, x, y);
  float* from_ptr = &from[0];
  to[0] = _mm256_load_ps(&from_ptr[0*8]);
  to[1] = _mm256_load_ps(&from_ptr[1*8]);
  to[2] = _mm256_load_ps(&from_ptr[2*8]);
  to[3] = _mm256_load_ps(&from_ptr[3*8]);
  to[4] = _mm256_load_ps(&from_ptr[4*8]);
  to[5] = _mm256_load_ps(&from_ptr[5*8]);
  to[6] = _mm256_load_ps(&from_ptr[6*8]);
  to[7] = _mm256_load_ps(&from_ptr[7*8]);
}


template<size_t N, size_t B>
void kernel_8x8(float* a, float* b, float* c, int x, int y) {
  alignas(32) float ax[8*8];
  __m256 bv[8], cv[8];
  cv[0] = _mm256_setzero_ps();
  cv[1] = _mm256_setzero_ps();
  cv[2] = _mm256_setzero_ps();
  cv[3] = _mm256_setzero_ps();
  cv[4] = _mm256_setzero_ps();
  cv[5] = _mm256_setzero_ps();
  cv[6] = _mm256_setzero_ps();
  cv[7] = _mm256_setzero_ps();
  for (int zz = 0; zz < N; zz += B) {
    pack<N, 8>(ax, a, x, zz);
    pack_into_vecs<N>(bv, b, zz, y);
    // calculate product of submatrces Ax and Bx here
    for (int i = 0; i < 8; i++) {
      __m256 alpha;
      alpha = _mm256_broadcast_ss(&ax[i*8+0]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[0], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+1]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[1], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+2]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[2], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+3]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[3], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+4]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[4], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+5]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[5], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+6]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[6], cv[i]);
      alpha = _mm256_broadcast_ss(&ax[i*8+7]);
      cv[i] = _mm256_fmadd_ps(alpha, bv[7], cv[i]);
    }
  }
  // store cv[] into c
  unpack_from_vecs<N>(c, cv, x, y);
}

template<size_t N, size_t B>
void blocked3(float* a, float* b, float* c) {
  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      kernel_8x8<N, B>(a, b, c, i, j);
      //print_matrix<N>(c);
    }
  }
}

template<size_t N>
void blocked_2x2(float* a, float* b, float* c) {
  for (int i = 0; i < N; i += 2) {
    for (int j = 0; j < N; j += 2) {
      kernel_2x2<N>(a, b, c, i, j);
    }
  }
}

template<size_t N>
void transpose_simd(float* a, float* b, float* c) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      __m256 ans = _mm256_setzero_ps();
      for (int j = 0; j < N; j += 8) {
        __m256 x = _mm256_load_ps(&a[i*N+j]);
        __m256 y = _mm256_load_ps(&b[k*N+j]);
        ans = _mm256_fmadd_ps(x, y, ans);
      }
      float vec[8];
      _mm256_store_ps(vec, ans);
      for (int x = 0; x < 8; x++) {
        c[i*N+k] += vec[x];
      }
    }
  }
}

template<size_t N>
void parallel(float* a, float* b, float* c) {
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

template<size_t N>
void parallel_tranposed_simd(float* a, float* b, float* c) {
  for (int hblock = 0; hblock < N; hblock += BLOCK_SIZE) {
    for (int vblock = 0; vblock < N; vblock += BLOCK_SIZE) {
#pragma omp parallel for collapse(2)
      for (int row = vblock; row < vblock + BLOCK_SIZE; row++) {
        for (int col = hblock; col < hblock + BLOCK_SIZE; col++) {
          __m256 ans = _mm256_setzero_ps();
          for (int k = 0; k < N; k += 8) {
            __m256 x = _mm256_load_ps(&a[row*N+k]);
            __m256 y = _mm256_load_ps(&b[col*N+k]);
            ans = _mm256_fmadd_ps(x, y, ans);
          }
          float vec[8];
          _mm256_store_ps(vec, ans);
          for (int x = 0; x < 8; x++) {
            c[row*N+col] += vec[x];
          }
        }
      }
    }
  }
}

#endif
