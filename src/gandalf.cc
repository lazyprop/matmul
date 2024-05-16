#include <algorithm>
#include <immintrin.h>

typedef union {
  __m256 v;
  float f[8];
} m256_t;

inline void pack_a(int k, const float* a, int lda, float* to) {
  for(int j=0; j<k; j++) {
    const float *a_ij_ptr = &a[(j*lda)+0]; 
    *to = *a_ij_ptr;
    *(to+1) = *(a_ij_ptr+1);
    *(to+2) = *(a_ij_ptr+2);
    *(to+3) = *(a_ij_ptr+3);
    *(to+4) = *(a_ij_ptr+4);
    *(to+5) = *(a_ij_ptr+5);
    *(to+6) = *(a_ij_ptr+6);
    *(to+7) = *(a_ij_ptr+7);
    to += 8;
  }
}

inline void pack_b(int k, const float* b, int ldb, float* to) {
  int i;
  const float *b_i0_ptr = &b[0], *b_i1_ptr = &b[(1*ldb)],
              *b_i2_ptr = &b[(2*ldb)], *b_i3_ptr = &b[(3*ldb)],
              *b_i4_ptr = &b[(4*ldb)], *b_i5_ptr = &b[(5*ldb)],
              *b_i6_ptr = &b[(6*ldb)], *b_i7_ptr = &b[(7*ldb)];
  for(i=0; i<k; i++) {
    *to     = *b_i0_ptr;
    *(to+1) = *(b_i1_ptr);
    *(to+2) = *(b_i2_ptr);
    *(to+3) = *(b_i3_ptr);
    *(to+4) = *(b_i4_ptr);
    *(to+5) = *(b_i5_ptr);
    *(to+6) = *(b_i6_ptr);
    *(to+7) = *(b_i7_ptr);
    to += 8;
    b_i0_ptr++; b_i1_ptr++; b_i2_ptr++;
    b_i3_ptr++; b_i4_ptr++; b_i5_ptr++;
    b_i6_ptr++; b_i7_ptr++;
  }
}


template <int mb, int kb, int th, int m, int n, int k>
inline void sgemm(float* a, float* b, float* c, int lda, int ldb, int ldc) {

  #pragma omp parallel for shared(a, b, c, lda, ldb, ldc) default(none) collapse(1) num_threads(th)
  for(int i=0; i<k; i+=kb) {
    int ib = std::min(k-i, kb);
    float* pb = new alignas(32) float[ib*n];
    for(int ii=0; ii<m; ii+=mb) {
      int iib = std::min(m-ii, mb);

      float* pa = new alignas(32) float[ib*iib];

      float* wa = &a[i*k+ii];
      float* wb = &b[i];

      for(int iii=0; iii<n; iii+=8) { // loop over all columns of C unrolled by 8
        if(ii==0) pack_b(ib, &wb[(iii*ldb)], n, &pb[iii*ib]);

        for(int iiii=0; iiii<iib; iiii+=8) {   // loop over rows of c until block, unrolled by 8
          if(iii==0) pack_a(ib, &wa[iiii], k, &pa[iiii*ib]);

          float* wpa = &pa[iiii*ib];
          float* wpb = &pb[iii*ib];
          float* wc = &c[ii+iii*n+iiii];

          m256_t c0007, c1017, c2027, c3037,
                 c4047, c5057, c6067, c7077,
                 a_vreg, b_p0_vreg;

          //m256_t b0, b1, b2, b3, b4, b5, b6, b7;

          c0007.v = _mm256_setzero_ps();
          c1017.v = _mm256_setzero_ps();
          c2027.v = _mm256_setzero_ps();
          c3037.v = _mm256_setzero_ps();
          c4047.v = _mm256_setzero_ps();
          c5057.v = _mm256_setzero_ps();
          c6067.v = _mm256_setzero_ps();
          c7077.v = _mm256_setzero_ps();

          for(int iiiii=0; iiiii<iib; iiiii++) {
            __builtin_prefetch(wpa+8);
            __builtin_prefetch(wpb+8);

            a_vreg.v = _mm256_load_ps( (float*) wpa );
            wpa += 8;

            /*
            b0.v = _mm256_broadcast_ss( (float*) &wpa[0] );
            b1.v = _mm256_broadcast_ss( (float*) &wpa[1] );
            b2.v = _mm256_broadcast_ss( (float*) &wpa[2] );
            b3.v = _mm256_broadcast_ss( (float*) &wpa[3] );
            b4.v = _mm256_broadcast_ss( (float*) &wpa[4] );
            b5.v = _mm256_broadcast_ss( (float*) &wpa[5] );
            b6.v = _mm256_broadcast_ss( (float*) &wpa[6] );
            b7.v = _mm256_broadcast_ss( (float*) &wpa[7] );
            */
            b_p0_vreg.v = _mm256_load_ps( (float*) wpb);
            wpb += 8;

            c0007.v += a_vreg.v * b_p0_vreg.f[0];
            c1017.v += a_vreg.v * b_p0_vreg.f[1];
            c2027.v += a_vreg.v * b_p0_vreg.f[2];
            c3037.v += a_vreg.v * b_p0_vreg.f[3];
            c4047.v += a_vreg.v * b_p0_vreg.f[4];
            c5057.v += a_vreg.v * b_p0_vreg.f[5];
            c6067.v += a_vreg.v * b_p0_vreg.f[6];
            c7077.v += a_vreg.v * b_p0_vreg.f[7];
/*
            c0007.v = _mm256_fmadd_ps(a_vreg.v, b0.v, c0007.v);
            c1017.v = _mm256_fmadd_ps(a_vreg.v, b1.v, c1017.v);
            c2027.v = _mm256_fmadd_ps(a_vreg.v, b2.v, c2027.v);
            c3037.v = _mm256_fmadd_ps(a_vreg.v, b3.v, c3037.v);
            c4047.v = _mm256_fmadd_ps(a_vreg.v, b4.v, c4047.v);
            c5057.v = _mm256_fmadd_ps(a_vreg.v, b5.v, c5057.v);
            c6067.v = _mm256_fmadd_ps(a_vreg.v, b6.v, c6067.v);
            c7077.v = _mm256_fmadd_ps(a_vreg.v, b7.v, c7077.v);
            */
          }

          m256_t w0, w1, w2, w3, w4, w5, w6, w7;

          w0.v = _mm256_load_ps((float*)&wc[0*ldc]);
          w1.v = _mm256_load_ps((float*)&wc[1*ldc]);
          w2.v = _mm256_load_ps((float*)&wc[2*ldc]);
          w3.v = _mm256_load_ps((float*)&wc[3*ldc]);
          w4.v = _mm256_load_ps((float*)&wc[4*ldc]);
          w5.v = _mm256_load_ps((float*)&wc[5*ldc]);
          w6.v = _mm256_load_ps((float*)&wc[6*ldc]);
          w7.v = _mm256_load_ps((float*)&wc[7*ldc]);

          c0007.v = _mm256_add_ps(c0007.v, w0.v);
          c1017.v = _mm256_add_ps(c1017.v, w1.v);
          c2027.v = _mm256_add_ps(c2027.v, w2.v);
          c3037.v = _mm256_add_ps(c3037.v, w3.v);
          c4047.v = _mm256_add_ps(c4047.v, w4.v);
          c5057.v = _mm256_add_ps(c5057.v, w5.v);
          c6067.v = _mm256_add_ps(c6067.v, w6.v);
          c7077.v = _mm256_add_ps(c7077.v, w7.v);

          _mm256_store_ps( &wc[0*ldc], c0007.v);
          _mm256_store_ps( &wc[1*ldc], c1017.v);
          _mm256_store_ps( &wc[2*ldc], c2027.v);
          _mm256_store_ps( &wc[3*ldc], c3037.v);
          _mm256_store_ps( &wc[4*ldc], c4047.v);
          _mm256_store_ps( &wc[5*ldc], c5057.v);
          _mm256_store_ps( &wc[6*ldc], c6067.v);
          _mm256_store_ps( &wc[7*ldc], c7077.v);

/*
          wc[(0*ldc)+0] += c0007.f[0]; wc[(1*ldc)+0] += c1017.f[0]; 
          wc[(2*ldc)+0] += c2027.f[0]; wc[(3*ldc)+0] += c3037.f[0]; 
          wc[(4*ldc)+0] += c4047.f[0]; wc[(5*ldc)+0] += c5057.f[0]; 
          wc[(6*ldc)+0] += c6067.f[0]; wc[(7*ldc)+0] += c7077.f[0]; 

          wc[(0*ldc)+1] += c0007.f[1]; wc[(1*ldc)+1] += c1017.f[1]; 
          wc[(2*ldc)+1] += c2027.f[1]; wc[(3*ldc)+1] += c3037.f[1]; 
          wc[(4*ldc)+1] += c4047.f[1]; wc[(5*ldc)+1] += c5057.f[1]; 
          wc[(6*ldc)+1] += c6067.f[1]; wc[(7*ldc)+1] += c7077.f[1]; 

          wc[(0*ldc)+2] += c0007.f[2]; wc[(1*ldc)+2] += c1017.f[2]; 
          wc[(2*ldc)+2] += c2027.f[2]; wc[(3*ldc)+2] += c3037.f[2]; 
          wc[(4*ldc)+2] += c4047.f[2]; wc[(5*ldc)+2] += c5057.f[2]; 
          wc[(6*ldc)+2] += c6067.f[2]; wc[(7*ldc)+2] += c7077.f[2]; 

          wc[(0*ldc)+3] += c0007.f[3]; wc[(1*ldc)+3] += c1017.f[3]; 
          wc[(2*ldc)+3] += c2027.f[3]; wc[(3*ldc)+3] += c3037.f[3]; 
          wc[(4*ldc)+3] += c4047.f[3]; wc[(5*ldc)+3] += c5057.f[3]; 
          wc[(6*ldc)+3] += c6067.f[3]; wc[(7*ldc)+3] += c7077.f[3]; 

          wc[(0*ldc)+4] += c0007.f[4]; wc[(1*ldc)+4] += c1017.f[4]; 
          wc[(2*ldc)+4] += c2027.f[4]; wc[(3*ldc)+4] += c3037.f[4]; 
          wc[(4*ldc)+4] += c4047.f[4]; wc[(5*ldc)+4] += c5057.f[4]; 
          wc[(6*ldc)+4] += c6067.f[4]; wc[(7*ldc)+4] += c7077.f[4]; 

          wc[(0*ldc)+5] += c0007.f[5]; wc[(1*ldc)+5] += c1017.f[5]; 
          wc[(2*ldc)+5] += c2027.f[5]; wc[(3*ldc)+5] += c3037.f[5]; 
          wc[(4*ldc)+5] += c4047.f[5]; wc[(5*ldc)+5] += c5057.f[5]; 
          wc[(6*ldc)+5] += c6067.f[5]; wc[(7*ldc)+5] += c7077.f[5]; 

          wc[(0*ldc)+6] += c0007.f[6]; wc[(1*ldc)+6] += c1017.f[6]; 
          wc[(2*ldc)+6] += c2027.f[6]; wc[(3*ldc)+6] += c3037.f[6]; 
          wc[(4*ldc)+6] += c4047.f[6]; wc[(5*ldc)+6] += c5057.f[6]; 
          wc[(6*ldc)+6] += c6067.f[6]; wc[(7*ldc)+6] += c7077.f[6]; 

          wc[(0*ldc)+7] += c0007.f[7]; wc[(1*ldc)+7] += c1017.f[7]; 
          wc[(2*ldc)+7] += c2027.f[7]; wc[(3*ldc)+7] += c3037.f[7]; 
          wc[(4*ldc)+7] += c4047.f[7]; wc[(5*ldc)+7] += c5057.f[7]; 
          wc[(6*ldc)+7] += c6067.f[7]; wc[(7*ldc)+7] += c7077.f[7]; 
          */
        }
      }
    }
  }
}


#define N 1024 
#define I 100

#include <iostream>
#include <iomanip>
#include <chrono>

int main() { 
  float* a = new alignas(32) float[N*N];
  float* b = new alignas(32) float[N*N];
  float* c = new alignas(32) float[N*N];

  for(int i=0; i<N*N; i++) {
    a[i] = i;
    b[i] = i;
  }

  long double sum1=0;
  for(int i=0; i<I; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    sgemm<128, 128, 12, N, N, N>(a, b, c, N, N, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end - start;
    sum1 += ms_double.count();
    for(int i=0; i<N*N; i++) c[i] = 0.f;
  }
  std::cout << "sgemm "<< I << " avg: " << sum1/I << " ms " << ((2.f*N*N*N)/((sum1/I)/1000))/1e9 << " GFLOPS" << std::endl;
  std::cout << "\n\n";

/*
  std::cout << "\n\n";
  for(int i=0; i<N*N; i++) {
    if(i%N==0) std::cout << "\n";
    std::cout << std::setw(6) << c[i] << ", ";
  }
  std::cout << "\n\n";
  */
}
