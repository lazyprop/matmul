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



template<typename T, size_t N, size_t P>
inline void pack_transpose3(T* b, T bb[8][8], int k, int j) {
  __m256 row0 = _mm256_load_ps(&b[(k+0)*N+j]);
  __m256 row1 = _mm256_load_ps(&b[(k+1)*N+j]);
  __m256 row2 = _mm256_load_ps(&b[(k+2)*N+j]);
  __m256 row3 = _mm256_load_ps(&b[(k+3)*N+j]);
  __m256 row4 = _mm256_load_ps(&b[(k+4)*N+j]);
  __m256 row5 = _mm256_load_ps(&b[(k+5)*N+j]);
  __m256 row6 = _mm256_load_ps(&b[(k+6)*N+j]);
  __m256 row7 = _mm256_load_ps(&b[(k+7)*N+j]);

  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(row0, row1);
  __t1 = _mm256_unpackhi_ps(row0, row1);
  __t2 = _mm256_unpacklo_ps(row2, row3);
  __t3 = _mm256_unpackhi_ps(row2, row3);
  __t4 = _mm256_unpacklo_ps(row4, row5);
  __t5 = _mm256_unpackhi_ps(row4, row5);
  __t6 = _mm256_unpacklo_ps(row6, row7);
  __t7 = _mm256_unpackhi_ps(row6, row7);
  __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
  __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
  __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
  __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
  __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
  __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
  __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
  __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
  row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);


  _mm256_store_ps(&bb[0][0], row0);
  _mm256_store_ps(&bb[1][0], row1);
  _mm256_store_ps(&bb[2][0], row2);
  _mm256_store_ps(&bb[3][0], row3);
  _mm256_store_ps(&bb[4][0], row4);
  _mm256_store_ps(&bb[5][0], row5);
  _mm256_store_ps(&bb[6][0], row6);
  _mm256_store_ps(&bb[7][0], row7);
}

template<typename T, size_t N, size_t P>
inline void pack_transpose4(T* b, T bb[8][8], int k, int j) {
  __m256  r0, r1, r2, r3, r4, r5, r6, r7;
  __m256  t0, t1, t2, t3, t4, t5, t6, t7;

  r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+0)*N+(j+0)])), _mm_load_ps(&b[(k+4)*N+(j+0)]), 1);
  r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+1)*N+(j+0)])), _mm_load_ps(&b[(k+5)*N+(j+0)]), 1);
  r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+2)*N+(j+0)])), _mm_load_ps(&b[(k+6)*N+(j+0)]), 1);
  r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+3)*N+(j+0)])), _mm_load_ps(&b[(k+7)*N+(j+0)]), 1);
  r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+0)*N+(j+4)])), _mm_load_ps(&b[(k+4)*N+(j+4)]), 1);
  r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+1)*N+(j+4)])), _mm_load_ps(&b[(k+5)*N+(j+4)]), 1);
  r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+2)*N+(j+4)])), _mm_load_ps(&b[(k+6)*N+(j+4)]), 1);
  r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&b[(k+3)*N+(j+4)])), _mm_load_ps(&b[(k+7)*N+(j+4)]), 1);
  
  t0 = _mm256_unpacklo_ps(r0,r1);
  t1 = _mm256_unpackhi_ps(r0,r1);
  t2 = _mm256_unpacklo_ps(r2,r3);
  t3 = _mm256_unpackhi_ps(r2,r3);
  t4 = _mm256_unpacklo_ps(r4,r5);
  t5 = _mm256_unpackhi_ps(r4,r5);
  t6 = _mm256_unpacklo_ps(r6,r7);
  t7 = _mm256_unpackhi_ps(r6,r7);

  r0 = _mm256_shuffle_ps(t0,t2, 0x44);
  r1 = _mm256_shuffle_ps(t0,t2, 0xEE);
  r2 = _mm256_shuffle_ps(t1,t3, 0x44);
  r3 = _mm256_shuffle_ps(t1,t3, 0xEE);
  r4 = _mm256_shuffle_ps(t4,t6, 0x44);
  r5 = _mm256_shuffle_ps(t4,t6, 0xEE);
  r6 = _mm256_shuffle_ps(t5,t7, 0x44);
  r7 = _mm256_shuffle_ps(t5,t7, 0xEE);

  _mm256_store_ps(&bb[0][0], r0);
  _mm256_store_ps(&bb[1][0], r1);
  _mm256_store_ps(&bb[2][0], r2);
  _mm256_store_ps(&bb[3][0], r3);
  _mm256_store_ps(&bb[4][0], r4);
  _mm256_store_ps(&bb[5][0], r5);
  _mm256_store_ps(&bb[6][0], r6);
  _mm256_store_ps(&bb[7][0], r7);

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

template<typename T, size_t N>
void goto4(T* a, T* b, T* c) {
  const int P = 8;
  for (int k = 0; k < N; k += P) {
    for (int i = 0; i < N; i += P) {
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

        __m256 rows[8];
        for (int ii = 0; ii < P; ii++) {
          rows[ii] = _mm256_load_ps(&c[(i+ii)*N+j]);
        }

        for (int ii = 0; ii < P; ii++) {
          for (int kk = 0; kk < P; kk++) {
            __m256 alpha = _mm256_broadcast_ss(&aa[ii][kk]);
            __m256 brow = _mm256_load_ps(&bb[kk][0]);
            rows[ii] = _mm256_fmadd_ps(alpha, brow, rows[ii]);
          }
        }

        for (int ii = 0; ii < P; ii++) {
          _mm256_stream_ps(&c[(i+ii)*N+j], rows[ii]);
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
  test_program<float, N>("goto2", goto2<float, N>, a, b, c, ans);
  //goto2<float, N>(a, b, c);
  //check_matrix<float, N>(c, ans);
  zero_matrix<float, N>(c);
  test_program<float, N>("goto3", goto3<float, N>, a, b, c, ans);
  zero_matrix<float, N>(c);
  test_program<float, N>("goto4", goto4<float, N>, a, b, c, ans);

  return 0;
}


int main() {
  const int N = 1024;
  //bench();
  //return 0;

  float* a = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* b = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* c = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));
  float* ans = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * N * N));

  rand_matrix<float, N>(a);
  rand_matrix<float, N>(b);
  zero_matrix<float, N>(c);


  //const int runs = 100;
  //for (int i = 0; i < runs; i++) {
  //goto2<float, N>(a, b, c);
  //}
  baseline<float, N>(a, b, c);
}
