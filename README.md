benchmarks

initializing b (in c = ab) makes the baseline matmul go from 40 gflops to 15 gflops
on my computer. this does not happen on other people's computers. see the output of
`baseline.cpp`

i have randomly initialized b in the benchmarks otherwise the blis kernels are too fast
to accurately judge their performance.


```
N = 1024
baseline: 17.8267 GFLOPS/s
layered: 40.7594 GFLOPS/s
layered2: 40.2947 GFLOPS/s
blis: 148.928 GFLOPS/s

N = 1920
baseline: 7.56156 GFLOPS/s
blis_12x8: 219.674 GFLOPS/s
```

best blis configs:
```
blis<N, 128, 64, 1024>    // for N = 1024
blis_12x8<N, 96, 48, 960> // for N = 1920
```


~**goal: 200 gflops**~ destroyed

150 gflops on N = 1024. numpy gets 210.
220 gflops on N = 1920. numpy gets 280.


### resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [Fast 8x8 Transpose](https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2)
