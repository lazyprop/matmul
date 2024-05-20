
benchmarks (without initializing b)

for N = 1024, Mc = 64
```
baseline: 41.392 GFLOPS/s
layered: 45.9056 GFLOPS/s
layered2: 42.1783 GFLOPS/s
blis: 42.0626 GFLOPS/s
parallel_tranposed_simd: 93.0767 GFLOPS/s

blis (parallel): 157.146 GFLOPS/s
```


for N = 1920, Mc = 96
```
baseline: 41.8663 GFLOPS/s
layered: 36.5576 GFLOPS/s
layered2: 38.754 GFLOPS/s
blis: 57.6408 GFLOPS/s
parallel_tranposed_simd: 78.8357 GFLOPS/s
```



**goal: 200 gflops**

initializing b (in c = ab) makes the baseline matmul go from 40 gflops to 15 gflops
on my computer. this does not happen on other people's computers. see the output of
`baseline.cpp`

### Resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [Fast 8x8 Transpose](https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2)
