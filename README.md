
benchmarks (without initializing b)
```

baseline: 41.77 GFLOPS/s
transpose_simd: 10.63 GFLOPS/s
blocked3_8x8: 8.18 GFLOPS/s
goto2: 18.84 GFLOPS/s
goto3: 10.84 GFLOPS/s
layered: 44.26 GFLOPS/s
parallel_tranposed_simd: 106.49 GFLOPS/s
```

**goal: 200 gflops**

initializing b (in c = ab) makes the baseline matmul go from 40 gflops to 15 gflops
on my computer. this does not happen on other people's computers.

### Resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [Fast 8x8 Transpose](https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2)
