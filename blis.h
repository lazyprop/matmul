#include <cstdlib>
#include <immintrin.h>

#include "util.h"
#include "matmul.h"


template <size_t N>
void blis(float* a, float* b, float* c) {
  const int Mc = 128;
  for (int k = 0; k < N; k += Mc) {
    for (int i = 0; i < N; i += Mc) {
      alignas(32) float aa[Mc][Mc], bb[Mc][N];

      for (int jj = 0; jj < N; jj += 8) {

        // pack b 
        for (int kk = 0; kk < Mc; kk++) {
          for (int jjj = 0; jjj < 8; jjj++) {
            bb[kk][jj+jjj] = b[(k+kk)*N+jj+jjj];
          }
        }

        for (int ii = 0; ii < Mc; ii += 8) {
          // pack a
          for (int iii = 0; iii < 8; iii++) {
            for (int kkk = 0; kkk < Mc; kkk++) {
              aa[ii+iii][kkk] = a[(i+ii+iii)*N+k+kkk];
            }
          }

          // now we have packs of a and b
          // compute 8x8 block at c[i+ii][[jj]

          __m256 cc[8];
          for (int iii = 0; iii < 8; iii++) {
            cc[iii] = _mm256_load_ps(&c[(i+ii+iii)*N+(jj)]);
          }

          __m256 av, bv;
          for (int kkk = 0; kkk < Mc; kkk++) {
            bv = _mm256_load_ps(&bb[kkk][jj]);
            for (int iii = 0; iii < 8; iii++) {
              av = _mm256_broadcast_ss(&aa[ii+iii][kkk]);
              cc[iii] = _mm256_fmadd_ps(av, bv, cc[iii]);
            }
          }

          for (int iii = 0; iii < 8; iii++) {
            _mm256_store_ps(&c[(i+ii+iii)*N+(jj)], cc[iii]);
          }
        }
      }
    }
  }
}



