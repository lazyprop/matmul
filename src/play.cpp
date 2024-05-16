#include <cstdlib>
#include <immintrin.h>

#include "util.h"

template<typename T, size_t N>
void goto1(T* a, T* b, T* c) {
  const int P = 8;
  for (int k = 0; k < N; k += P) {
    // GEPP
    for (int i = 0; i < N; i += P) {
      // GEBP
      for (int kk = k; kk < k+P; kk++) {
        // GEBB
        for (int j = 0; j < N; j += P) {
          for (int ii = i; ii < i+P; ii++) {
            for (int jj = j; jj < j+P; jj++) {
              c[ii*N+jj] += a[ii*N+kk] * b[kk*N+jj];
            }
          }
        }
      }
    }
  }
}

template<typename T, size_t N>
void goto2(T* a, T* b, T* c) {
  const int P = 8;
  for (int k = 0; k < N; k += P) {
    for (int i = 0; i < N; i += P) {
      // pack a[i:i+P][k:k+K]
      alignas(32) T aa[P][P];
      for (int ii = 0; ii < P; ii++) {
        for (int kk = 0; kk < P; kk++) {
          aa[ii][kk] = a[(i+ii)*N+(k+kk)];
        }
      }

      alignas(32) T bb[P][P];
      for (int j = 0; j < N; j += P) {
        for (int kk = 0; kk < P; kk++) {
          for (int jj = 0; jj < P; jj++) {
            bb[kk][jj] = b[(k+kk)*N+(j+jj)];
          }
        }

        for (int ii = 0; ii < P; ii++) {
          for (int jj = 0; jj < P; jj++) {
            T acc = 0;
            for (int kk = 0; kk < P; kk++) {
              acc += aa[ii][kk] * bb[kk][jj];
            }
            c[(i+ii)*N+(j+jj)] += acc;
          }
        }
      }
    }
  }
}

struct packed_4x8 {
  __m128 b00, b04, b10, b14, b20, b24, b30, b34;
};

template<typename T, size_t N, size_t P>
inline void pack_transpose1(T* b, T bb[8][8], int k, int j) {
  for (int kk = 0; kk < P; kk++) {
    for (int jj = 0; jj < P; jj++) {
      bb[jj][kk] = b[(k+kk)*N+(j+jj)];
    }
  }
}

inline struct packed_4x8 read_packed_4x8(float* mat, int i, int j, int N) {
  packed_4x8 p;
  p.b00 = _mm_load_ps(&mat[i*N+j]);
  p.b04 = _mm_load_ps(&mat[i*N+j+4]);
  p.b10 = _mm_load_ps(&mat[(i+1)*N+j]);
  p.b14 = _mm_load_ps(&mat[(i+1)*N+j+4]);
  p.b20 = _mm_load_ps(&mat[(i+2)*N+j]);
  p.b24 = _mm_load_ps(&mat[(i+2)*N+j+4]);
  p.b30 = _mm_load_ps(&mat[(i+3)*N+j]);
  p.b34 = _mm_load_ps(&mat[(i+3)*N+j+4]);
  return p;
}

template<typename T, size_t N, size_t P>
inline void pack_transpose2(T* b, T bb[8][8], int k, int j) {
  packed_4x8 p = read_packed_4x8(b, k, j, N);
  _MM_TRANSPOSE4_PS(p.b00, p.b10, p.b20, p.b30);
  _MM_TRANSPOSE4_PS(p.b04, p.b14, p.b24, p.b34);

  float temp[16];
  _mm_store_ps(&temp[0], p.b00);
  _mm_store_ps(&temp[4], p.b10);
  _mm_store_ps(&temp[8], p.b20);
  _mm_store_ps(&temp[12], p.b30);

  _mm_store_ps(&bb[0][0], p.b00);
  _mm_store_ps(&bb[1][0], p.b10);
  _mm_store_ps(&bb[2][0], p.b20);
  _mm_store_ps(&bb[3][0], p.b30);

  _mm_store_ps(&bb[4][0], p.b04);
  _mm_store_ps(&bb[5][0], p.b14);
  _mm_store_ps(&bb[6][0], p.b24);
  _mm_store_ps(&bb[7][0], p.b34);

  p = read_packed_4x8(b, k+4, j, N);
  _MM_TRANSPOSE4_PS(p.b00, p.b10, p.b20, p.b30);
  _MM_TRANSPOSE4_PS(p.b04, p.b14, p.b24, p.b34);

  _mm_store_ps(&bb[0][4], p.b00);
  _mm_store_ps(&bb[1][4], p.b10);
  _mm_store_ps(&bb[2][4], p.b20);
  _mm_store_ps(&bb[3][4], p.b30);

  _mm_store_ps(&bb[4][4], p.b04);
  _mm_store_ps(&bb[5][4], p.b14);
  _mm_store_ps(&bb[6][4], p.b24);
  _mm_store_ps(&bb[7][4], p.b34);
}


template<typename T, size_t N>
void goto3(T* a, T* b, T* c) {
  const int P = 8;
  for (int k = 0; k < N; k += P) {
    for (int i = 0; i < N; i += P) {
      // pack a[i:i+P][k:k+K]
      alignas(32) T aa[P][P];
      for (int ii = 0; ii < P; ii++) {
        for (int kk = 0; kk < P; kk++) {
          aa[ii][kk] = a[(i+ii)*N+(k+kk)];
        }
      }

      alignas(32) T bb[P][P];
      for (int j = 0; j < N; j += P) {
        pack_transpose1<T, N, P>(b, bb, k, j);

        for (int ii = 0; ii < P; ii++) {
          for (int jj = 0; jj < P; jj++) {
            T acc = 0;
            for (int kk = 0; kk < P; kk++) {
              acc += aa[ii][kk] * bb[jj][kk];
            }
            c[(i+ii)*N+(j+jj)] += acc;
          }
        }
      }
    }
  }
}

int bench() {
  const int N = 1024;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<float, N>(a);
  rand_matrix<float, N>(b);
  zero_matrix<float, N>(c);
  zero_matrix<float, N>(ans);

  //print_matrix<float, N>(b);

  test_program<float, N>("baseline", baseline<float, N>, a, b, ans, ans);
  zero_matrix<float, N>(c);
  //test_program<float, N>("goto1", goto1<float, N>, a, b, c, ans);
  //zero_matrix<float, N>(c);
  test_program<float, N>("goto2", goto2<float, N>, a, b, c, ans);
  //goto2<float, N>(a, b, c);
  //check_matrix<float, N>(c, ans);
  zero_matrix<float, N>(c);
  test_program<float, N>("goto3", goto3<float, N>, a, b, c, ans);

  return 0;
}


int main() {
  const int N = 1024;
  bench();
  return 0;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<float, N>(a);
  rand_matrix<float, N>(b);
  zero_matrix<float, N>(c);


  const int runs = 100;
  for (int i = 0; i < runs; i++) {
    goto3<float, N>(a, b, c);
  }
}
