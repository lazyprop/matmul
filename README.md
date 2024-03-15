# Fast Linear Algebra in C

Making linear algebra go brrr.

| Program       | Without O3 (GFLOPS/s) | With O3 (GFLOPS/s) |
|---------------|----------------------|-------------------|
| baseline      | 0.6                  | 23.5              |
| transposed    | 0.7                  | 30.7              |
| simd          | 2.6                  | 12.5              |
| blocked_2x2   | 1.4                  | 1.4               |
| blocked_8x8   | 0.80                 | 6.18              |
| blocked_16x16 | 0.80                 | 18.19             |
| parallel      | 2.7                  | 1.3               |
| parallel_simd | 18.8                 | 96.9              |


### Resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)


