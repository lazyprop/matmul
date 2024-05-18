#ifndef UTIL_H
#define UTIL_H


#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <omp.h>

#include "matmul.h"

template <typename T, size_t N>
void transpose_matrix(T* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      T tmp = mat[j*N+i];
      mat[j*N+i] = mat[i*N+j];
      mat[i*N+j] = tmp;
    }
  }
}

template <typename T, size_t N>
void rand_matrix(T* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N+j] = (T) rand() / (T) RAND_MAX;
    }
  }
}

template <typename T, size_t N>
void zero_matrix(T* mat) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[i*N+j] = 0;
    }
  }
}

template <typename T, size_t N>
int check_matrix(T* mat, T* ans) {
  const T ERR = 1e-2;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      T diff = fabsf(mat[i*N+j] - ans[i*N+j]);
      if (diff > ERR) {
        std::cout << "failed: answer does not match. difference: "
          //<< "%2f at (%d, %d)\n",
                  << diff << " at (" << i << " " << j << ")\n";
      }
      assert(diff < ERR);
    }
  }
  return true;
}

template <typename T, size_t N>
void print_matrix(T* mat) {
  std::cout << std::fixed << std::setprecision(2);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
     std::cout << mat[i*N+j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

template <size_t N>
double time_to_gflops_s(const double seconds) {
  double total_flops = 2.0 * N * N * N;
  double gflops_second = total_flops / (seconds * 1e9);
  return gflops_second;
}

template <typename T, size_t N>
void test_program(const char* name, void (*func)(T*, T*, T*),
                  T* a, T* b, T* c, T* ans) {
  double seconds = 0;
  int runs = 5;
  for (int i = 0; i < runs; i++) {
    zero_matrix<T, N>(c);
    double begin = omp_get_wtime();
    func(a, b, c);
    seconds += omp_get_wtime() - begin;
  }
  std::cout << name << ": " << time_to_gflops_s<N>(seconds / runs)
            << " GFLOPS/s\n";
  #ifdef DEBUG
  print_matrix(c);
  std::cout << "answer:\n";
  print_matrix(ans);
  #endif
  check_matrix<float, N>(c, ans);
}

#endif
