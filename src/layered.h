#include <cstdlib>
#include <immintrin.h>

#include "util.h"
#include "matmul.h"


template <size_t N, size_t Mb>
inline void gebp(float ab[Mb][Mb], float bp[Mb][N], float cp[Mb][N]) {
  for (int ii = 0; ii < Mb; ii++) {
    for (int kk = 0; kk < Mb; kk++) {
      for (int jj = 0; jj < N; jj++) {
        cp[ii][jj] += ab[ii][kk] * bp[kk][jj];
      }
    }
  }
}

template <size_t N, size_t Mb>
inline void gebp2(float ab[Mb][Mb], float bp[Mb][N], float cp[Mb][N]) {
  for (int ii = 0; ii < Mb; ii++) {
    for (int kk = 0; kk < Mb; kk++) {
      for (int jj = 0; jj < N; jj++) {
        cp[ii][jj] += ab[ii][kk] * bp[kk][jj];
      }
    }
  }
}



template <size_t N>
void gemm(float* a, float* b, float* c) {
  const int Mb = 128;
  for (int k = 0; k < N; k += Mb) {
    // GEPP implicit to avoid polluting the cache
    for (int i = 0; i < N; i += Mb) {
      alignas(32) float ab[Mb][Mb];
      for (int ii = 0; ii < Mb; ii++) {
        for (int kk = 0; kk < Mb; kk++) {
          ab[ii][kk] = a[(i+ii)*N+(k+kk)];
        }
      }

      alignas(32) float bp[Mb][N];
      for (int kk = 0; kk < Mb; kk++) {
        for (int jj = 0; jj < N; jj++) {
          bp[kk][jj] = b[(k+kk)*N+jj];
        }
      }

      alignas(32) float cp[Mb][N];
      for (int ii = 0; ii < Mb; ii++) {
        for (int jj = 0; jj < N; jj++) {
          cp[ii][jj] = c[(i+ii)*N+jj];
        }
      }

      gebp<N, Mb>(ab, bp, cp);

      for (int ii = 0; ii < Mb; ii++) {
        for (int jj = 0; jj < N; jj++) {
          c[(i+ii)*N+jj] = cp[ii][jj];
        }
      }

    }
  }
}
