#ifndef _MATMUL_H
#define _MATMUL_H

#define DTYPE float
#define ERR 1e-3


#ifdef DEBUG
#define N 4
#define BLOCK_SIZE 4
#else
#define N 1024
#define BLOCK_SIZE 32
#endif

void baseline(DTYPE* a, DTYPE* b, DTYPE* c);
void transposed(DTYPE* a, DTYPE* b, DTYPE* c);
void tiled(DTYPE* a, DTYPE* b, DTYPE* c);
void simd(DTYPE* a, DTYPE* b, DTYPE* c);
void parallel(DTYPE* a, DTYPE* b, DTYPE* c);
void parallel_simd(DTYPE* a, DTYPE* b, DTYPE* c);

#endif
