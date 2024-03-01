#ifndef _MATMUL_H
#define _MATMUL_H

#define DTYPE float
#define ERR 1e-4

#ifdef DEBUG
#define N 16
#define BLOCK_SIZE 4
#else
#define N 1024
#define BLOCK_SIZE 32
#endif

void baseline(DTYPE* a, DTYPE* b, DTYPE* c);
void transposed(DTYPE* a, DTYPE* b, DTYPE* c);
void simd(DTYPE* a, DTYPE* b, DTYPE* c);
void blocked(DTYPE* a, DTYPE* b, DTYPE* c);
void blocked_simd(DTYPE* a, DTYPE* b, DTYPE* c);

#endif
