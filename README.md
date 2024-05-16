
Important kernels:

```
baseline: 13.64 GFLOPS/s
transpose_simd: 10.52 GFLOPS/s
blocked3_8x8: 11.01 GFLOPS/s
goto2: 21.92 GFLOPS/s
goto3: 22.37 GFLOPS/s
layered: 33.53 GFLOPS/s
parallel_tranposed_simd: 93.55 GFLOPS/s
```

**goal: 200 GFLOPS/s**

### Resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [Fast 8x8 Transpose](https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2)
