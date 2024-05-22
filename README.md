benchmarks (without initializing b) 

initializing b (in c = ab) makes the baseline matmul go from 40 gflops to 15 gflops
on my computer. this does not happen on other people's computers. see the output of
`baseline.cpp`


```
N = 1024
baseline: 42.0589 GFLOPS/s
layered: 44.3357 GFLOPS/s
layered2: 42.5642 GFLOPS/s
blis: 175.679 GFLOPS/s

N = 1920
baseline: 57.194 GFLOPS/s
blis_12x8: 333.19 GFLOPS/s
```

best blis configs:
```
blis<N, 128, 64, 1024>    // for N = 1024
blis_12x8<N, 96, 48, 960> // for N = 1920
```


**goal: 200 gflops (multi threaded)**

reach 82 gflops on N = 1920 single threaded. numpy gets 110.
212 gflops oon N = 1920 multi threaded. numpy gets 279.


### resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [Fast 8x8 Transpose](https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2)
