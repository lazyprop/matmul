
benchmarks (without initializing b)

for N = 1024
```
baseline: 41.8347 GFLOPS/s
layered: 46.6375 GFLOPS/s
layered2: 42.5188 GFLOPS/s
blis: 48.6394 GFLOPS/s
parallel_tranposed_simd: 105.552 GFLOPS/s
```


for N = 1920
```
baseline: 41.9104 GFLOPS/s
layered: 40.0003 GFLOPS/s
layered2: 38.6014 GFLOPS/s
blis_12x8: 82.4083 GFLOPS/s
parallel_tranposed_simd: 84.0283 GFLOPS/s

blis_12x8: 212.3 GFLOPS/s (multithreaded)
```


best blis configs:
```
blis<N, 128, 64, 1024>    // for N = 1024
blis_12x8<N, 96, 48, 960> // for N = 1920
```


**goal: 200 gflops (multi threaded)**

reach 82 gflops on N = 1920 single threaded. numpy gets 110.
212 gflops oon N = 1920 multi threaded. numpy gets 279.


initializing b (in c = ab) makes the baseline matmul go from 40 gflops to 15 gflops
on my computer. this does not happen on other people's computers. see the output of
`baseline.cpp`

### Resources

- [Siboehm's article](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [Case Study in Algorithms for Modern Hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [Anatomy of High Perfomance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [Fast 8x8 Transpose](https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2)
