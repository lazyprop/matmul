#include <cstdlib>
#include <immintrin.h>

#include "util.h"
#include "matmul.h"

//#define PREFETCH

// TODO fast transpose packing
template <int N, int Mc, int Kc, int Mr>
void pack_a(float aa[Mc][Kc], float* wa, int ii) {
  for (int iii = 0; iii < Mr; iii++) {
    for (int kkk = 0; kkk < Kc; kkk++) {
      aa[ii+iii][kkk] = wa[iii*N+kkk];
      //aa[kkk][ii+iii] = wa[iii*N+kkk];
    }
  }
}

template <int N, int Kc>
inline void pack_b(float bb[Kc][N], float* wb, int jj) {
  for (int kk = 0; kk < Kc; kk++) {
    for (int jjj = 0; jjj < 8; jjj++) {
      bb[kk][jj+jjj] = wb[kk*N+jjj];
    }
  }
}

template <int N, int Mc, int Kc>
inline void kernel_8x8(__m256 cc[8], float aa[Mc][Kc],
                       float bb[Kc][N], int ii, int jj) {
  __m256 av, bv;
  for (int kkk = 0; kkk < Kc; kkk++) {
    bv = _mm256_load_ps(&bb[kkk][jj]);

    for (int iii = 0; iii < 8; iii++) {
      av = _mm256_broadcast_ss(&aa[ii+iii][kkk]);
      cc[iii] = _mm256_fmadd_ps(av, bv, cc[iii]);
    }
  }
}

template <int N, int Mc, int Kc>
inline void kernel_12x8(__m256 cc[12], float aa[Mc][Kc],
                        float bb[Kc][N], int ii, int jj) {
  __m256 av, bv;
  for (int kkk = 0; kkk < Kc; kkk++) {
    bv = _mm256_load_ps(&bb[kkk][jj]);

    for (int iii = 0; iii < 12; iii++) {
      av = _mm256_broadcast_ss(&aa[ii+iii][kkk]);
      cc[iii] = _mm256_fmadd_ps(av, bv, cc[iii]);
    }
  }
}


template <int N, int Mc, int Kc, int Nc>
void blis(float* a, float* b, float* c) {
  alignas(32) float aa[Mc][Kc];
  alignas(32) float bb[Kc][N];

  for (int j = 0; j < N; j += Nc) {
    for (int i = 0; i < N; i += Mc) {
      for (int k = 0; k < N; k += Kc) {
        for (int jj = 0; jj < Nc; jj += 8) {
          alignas(32) float* wb = &b[k*N+j+jj];
          pack_b<N, Kc>(bb, wb, jj);

          for (int ii = 0; ii < Mc; ii += 8) {
            alignas(32) float* wa = &a[(i+ii)*N+k];
            if (jj == 0) pack_a<N, Mc, Kc, 8>(aa, wa, ii);
            else {
              for (int iii = 0; iii < 8; iii++) {
                for (int kkk = 0; kkk < Kc; kkk += 16) {
                  __builtin_prefetch(&aa[ii+iii][kkk]);
                }
              }
            }

            __m256 cc[8];
            for (int iii = 0; iii < 8; iii++) {
              cc[iii] = _mm256_load_ps(&c[(i+ii+iii)*N+(j+jj)]);
            }

            kernel_8x8<N, Mc, Kc>(cc, aa, bb, ii, jj);

            for (int iii = 0; iii < 8; iii++) {
              _mm256_store_ps(&c[(i+ii+iii)*N+(j+jj)], cc[iii]);
            }
          }
        }
      }
    }
  }
}
template <int N, int Mc, int Kc, int Nc>
void blis_12x8(float* a, float* b, float* c) {
  alignas(32) float aa[Mc][Kc];
  alignas(32) float bb[Kc][N];

#pragma omp parallel for collapse(2)
  for (int j = 0; j < N; j += Nc) {
    for (int i = 0; i < N; i += Mc) {
      for (int k = 0; k < N; k += Kc) {
        for (int jj = 0; jj < Nc; jj += 8) {
          alignas(32) float* wb = &b[k*N+j+jj];
          pack_b<N, Kc>(bb, wb, jj);

          for (int ii = 0; ii < Mc; ii += 12) {
            alignas(32) float* wa = &a[(i+ii)*N+k];
            if (jj == 0) pack_a<N, Mc, Kc, 12>(aa, wa, ii);
            else {
              for (int iii = 0; iii < 12; iii++) {
                for (int kkk = 0; kkk < Kc; kkk += 16) {
                  __builtin_prefetch(&aa[ii+iii][kkk]);
                }
              }
            }

            __m256 cc[12];
            for (int iii = 0; iii < 12; iii++) {
              cc[iii] = _mm256_load_ps(&c[(i+ii+iii)*N+(j+jj)]);
            }

            kernel_12x8<N, Mc, Kc>(cc, aa, bb, ii, jj);

            for (int iii = 0; iii < 12; iii++) {
              _mm256_store_ps(&c[(i+ii+iii)*N+(j+jj)], cc[iii]);
            }
          }
        }
      }
    }
  }
}
