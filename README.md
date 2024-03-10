# Fast Linear Algebra in C

Making linear algebra go brrr.

| Program       | Without O3 | With O3 |
|---------------|------------|---------|
| baseline      | 3792       | 168     |
| transposed    | 3008       | 151     |
| simd          | 837        | 196     |
| tiled         | 3218       | 6987    |
| parallel      | 1472       | 1749    |
| parallel_simd | 117        | 22      |


### Resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)

