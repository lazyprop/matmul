#ifndef MATMUL_H
#define MATMUL_H

#include <x86intrin.h>
#include <omp.h>

#include "util.h"

const int BLOCK_SIZE = 32;

template<typename T, size_t N>
void baseline(T* a, T* b, T* c) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i*N+k] += a[i*N+j] * b[j*N+k];
      }
    }
  }
}

template<typename T, size_t N>
void transposed(T* a, T* b, T* c) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        c[i*N+k] += a[i*N+j] * b[k*N+j];
      }
    }
  }
}

template<typename T, size_t N>
void tiled(T* a, T* b, T* c) {
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

template<typename T, size_t N>
void kernel_2x2(T* a, T* b, T* c, int x, int y) {
  // zero accumulators
  T c00 = 0, c01 = 0, c10 = 0, c11 = 0;
  for (int k = 0; k < N; k++) {
    // read the rows and columns
    T a0 = a[x*N+k], a1 = a[(x+1)*N+k];
    T b0 = b[k*N+y], b1 = b[k*N+y+1];
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


template<typename T, size_t N, size_t B>
void kernel(T* a, T* b, T* c, int x, int y) {
  T cx[B][B] = {};
  for (int k = 0; k < N; k++) {
    T ax[B], bx[B];
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

template<typename T, size_t N, size_t B>
void blocked(T* a, T* b, T* c) {
  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      kernel<T, N, B>(a, b, c, i, j);
    }
  }
}


/*
 * Pack a BxB submatrix `from` at (x, y) to a contiguous array `to`
 */
template<typename T, size_t N, size_t B>
inline void pack(T* to, T* from, int x, int y) {
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      to[i*B+j] = from[(x+i)*N+(y+j)];
    }
  }
}

/*
 * Transpose and pack BxB submatrix `from` at (x, y) to a contiguous array `to`
 */
template<typename T, size_t N, size_t B>
inline void pack_transpose(T* to, T* from, int x, int y) {
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < B; j++) {
      to[j*B+i] = from[(x+i)*N+(y+j)];
    }
  }
}

template<typename T, size_t N, size_t B>
void kernel2(T* a, T* b, T* c, int x, int y) {
  T ax[B*B], bx[B*B], cx[B*B] = {};
  for (int zz = 0; zz < N; zz += B) {
    pack<T, N, B>(ax, a, x, zz);
    pack_transpose<T, N, B>(bx, b, zz, y);
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


/*
 * Unpack an array of __m256 `from` to 8x8 submatrix at `to[x][y]`
 */
template<typename T, size_t N>
inline void unpack_8x8(T* to, __m256* from, int x, int y) {
  for (int k = 0; k < 8; k++) {
    _mm256_store_ps(&to[(x+k)*N+y], from[k]);
  }
}

/*
 * Pack a 8x8 submatrix `from` at (x, y) to an array of _m256
 */
template<typename T, size_t N>
inline void pack_8x8(__m256* to, T* from, int x, int y) {
  for (int i = 0; i < 8; i++) {
    to[i] = _mm256_load_ps(&from[i*N]);
  }
}

/*
 * Pack a 8x8 submatrix `from` at (x, y) to an array of _m256
 */
template<typename T, size_t N>
inline void pack_8x8_transpose(__m256* to, T* _from, int x, int y) {
  alignas(64) T from[8*8];
  pack_transpose<float, 8, 8>(from, _from, x, y);
  for (int i = 0; i < 8; i++) {
    to[i] = _mm256_load_ps(&from[i*N]);
  }
}

template<typename T, size_t N, size_t B>
void kernel_8x8(T* a, T* b, T* c, int x, int y) {
  assert(B == 8);
  alignas(64) T ax[8*8];
  __m256 bv[8], cv[8];
  for (int k = 0; k < 8; k++) cv[k] = _mm256_setzero_ps();
  for (int zz = 0; zz < N; zz += B) {
    pack<T, N, 8>(ax, a, x, zz);
    //std::cout << "Ax:\n";
    //print_matrix<float, 8>(ax);
    pack_8x8<T, N>(bv, b, zz, y);
    // calculate product of submatrces Ax and Bx here
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        // broadcast ax[i][j]
        alignas(64) const T _alpha = ax[i*8+j];
        __m256 alpha = _mm256_broadcast_ss(&_alpha);
        // cv[i] += alpha (broadcast) * b[j]
        cv[i] = _mm256_fmadd_ps(alpha, bv[j], cv[i]);
      }
    }
  }
  // store cv[] into c
  unpack_8x8<float, N>(c, cv, x, y);
  alignas(64) T cx[8*8];
  unpack_8x8<float, N>(cx, cv, 0, 0);
  std::cout << "Cx: " << x << " " << y << "\n";
  print_matrix<float, 8>(cx);
}

template<typename T, size_t N, size_t B>
void kernel2_8x8(T* a, T* b, T* c, int x, int y) {
  assert(B == 8);
  __m256 cv[8];
  for (int k = 0; k < 8; k++) cv[k] = _mm256_setzero_ps();
  for (int i = 0; i < 8; i++) {
    for (int k = 0; k < N; k++) {
      const T _alpha = a[(x+i)*N+k];
      __m256 alpha = _mm256_broadcast_ss(&_alpha);
      for (int j = 0; j < 8; j++) {
      }
    }
  }
  // store cv[] into c
  unpack_8x8<float, N>(c, cv, x, y);
  alignas(64) T cx[8*8];
  unpack_8x8<float, N>(cx, cv, 0, 0);
  std::cout << "Cx: " << x << " " << y << "\n";
  print_matrix<float, 8>(cx);
}

template<typename T, size_t N, size_t B>
void blocked3(T* a, T* b, T* c) {
  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      kernel2_8x8<T, N, B>(a, b, c, i, j);
    }
  }
}

template<typename T, size_t N, size_t B>
void blocked2(T* a, T* b, T* c) {
  for (int i = 0; i < N; i += B) {
    for (int j = 0; j < N; j += B) {
      kernel2<T, N, B>(a, b, c, i, j);
    }
  }
}



template<typename T, size_t N>
void blocked_2x2(T* a, T* b, T* c) {
  for (int i = 0; i < N; i += 2) {
    for (int j = 0; j < N; j += 2) {
      kernel_2x2<T, N>(a, b, c, i, j);
    }
  }
}

template<typename T, size_t N>
void transpose_simd(T* a, T* b, T* c) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      __m256 ans = _mm256_setzero_ps();
      for (int j = 0; j < N; j += 8) {
        __m256 x = _mm256_load_ps(&a[i*N+j]);
        __m256 y = _mm256_load_ps(&b[k*N+j]);
        ans = _mm256_fmadd_ps(x, y, ans);
      }
      T vec[8];
      _mm256_store_ps(vec, ans);
      for (int x = 0; x < 8; x++) {
        c[i*N+k] += vec[x];
      }
    }
  }
}

template<typename T, size_t N>
void parallel(T* a, T* b, T* c) {
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

template<typename T, size_t N>
void parallel_tranposed_simd(T* a, T* b, T* c) {
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
          T vec[8];
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
