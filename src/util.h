#ifndef _UTIL_H
#define _UTIL_H

#include <stdbool.h>

void transpose_matrix(DTYPE*);
void print_matrix(DTYPE*);
void rand_matrix(DTYPE*);
void zero_matrix(DTYPE*);
bool check_matrix(DTYPE*, DTYPE*);
void test_program(const char* name, void (*func)(DTYPE*, DTYPE*, DTYPE*),
                  DTYPE* a, DTYPE* b, DTYPE* c, DTYPE* ans);
double time_to_gflops_s(double);

#endif
