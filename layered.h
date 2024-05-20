#include <cstdlib>
#include <immintrin.h>

#include "util.h"
#include "matmul.h"

template <size_t N, size_t Mb>
void gebp(float ab[Mb][Mb], float bp[Mb][N], float cp[Mb][N]) {
  for (int ii = 0; ii < Mb; ii++) {
    for (int kk = 0; kk < Mb; kk++) {
      for (int jj = 0; jj < N; jj++) {
        cp[ii][jj] += ab[ii][kk] * bp[kk][jj];
      }
    }
  }
}

// compute Mb x Mb locks at a time
template <size_t N, size_t Mb>
inline void gebp2(float ab[Mb][Mb], float bp[Mb][N], float cp[Mb][N]) {
  for (int j = 0; j < N; j += Mb) {
    alignas (32) float bb[Mb][Mb];
    for (int kk = 0; kk < Mb; kk++) {
      for (int jj = 0; jj < Mb; jj++) {
        bb[kk][jj] = bp[kk][j+jj];
      }
    }

    alignas(32) float cc[Mb][Mb];
    for (int ii = 0; ii < Mb; ii++) {
      for (int jj = 0; jj < Mb; jj++) {
        cc[ii][jj] = cp[ii][j+jj];
      }
    }

    for (int ii = 0; ii < Mb; ii++) {
      for (int kk = 0; kk < Mb; kk++) {
        for (int jj = 0; jj < Mb; jj++) {
          cc[ii][jj] += ab[ii][kk] * bb[kk][jj];
        }
      }
    }

    for (int ii = 0; ii < Mb; ii++) {
      for (int jj = 0; jj < Mb; jj++) {
        cp[ii][j+jj] = cc[ii][jj];
      }
    }
  }
}

// compute Mb x Mb locks at a time
template <size_t N, size_t Mb>
inline void gebp3(float aa[Mb][Mb], float* b, float* c, int i, int k) {
  for (int j = 0; j < N; j += Mb) {
    alignas (32) float bb[Mb][Mb];
    for (int kk = 0; kk < Mb; kk++) {
      for (int jj = 0; jj < Mb; jj++) {
        bb[kk][jj] = b[(k+kk)*N+(j+jj)];
      }
    }

    alignas(32) float cc[Mb][Mb];
    for (int ii = 0; ii < Mb; ii++) {
      for (int jj = 0; jj < Mb; jj++) {
        cc[ii][jj] = c[(i+ii)*N+(j+jj)];
      }
    }

    for (int ii = 0; ii < Mb; ii++) {
      for (int kk = 0; kk < Mb; kk++) {
        for (int jj = 0; jj < Mb; jj++) {
          cc[ii][jj] += aa[ii][kk] * bb[kk][jj];
        }
      }
    }

    for (int ii = 0; ii < Mb; ii++) {
      for (int jj = 0; jj < Mb; jj++) {
        c[(i+ii)*N+(j+jj)] = cc[ii][jj];
      }
    }
  }
}



template <size_t N>
void gemm(float* a, float* b, float* c) {
  const int Mb = 64;
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

template <size_t N>
void gemm2(float* a, float* b, float* c) {
  const int Mb = 64;
  for (int k = 0; k < N; k += Mb) {
    // GEPP implicit to avoid polluting the cache
    for (int i = 0; i < N; i += Mb) {
      alignas(32) float aa[Mb][Mb];
      for (int ii = 0; ii < Mb; ii++) {
        for (int kk = 0; kk < Mb; kk++) {
          aa[ii][kk] = a[(i+ii)*N+(k+kk)];
        }
      }

      gebp3<N, Mb>(aa, b, c, i, k);

    }
  }
}
