
learning about high performance matmul on cpu. the fastest kernels i've implemented
are `blis` and `blis_12x8` in `blis.h`

best blis configs:
```
blis<N, 128, 64, 1024>    // for N = 1024
blis_12x8<N, 96, 48, 960> // for N = 1920
```

to benchmark all (half-decent) kernels, `make bench && ./bench`. only the blis
kernels are multi threaded. remove `-fopenmp` from clags to limit to single thread.
compare against numpy with `python test-numpy.py`. change value of N as required.


initializing b (in c = ab) makes the baseline matmul go from 40 gflops to 15 gflops
on my computer. this does not happen on other people's computers. see the output of
`baseline.cpp`

### benchmarks

cpu details:
```
Model name:             AMD Ryzen 5 PRO 4650U with Radeon Graphics
  Thread(s) per core:   2
  Core(s) per socket:   6
Caches (sum of all):      
  L1d:                    192 KiB (6 instances)
  L1i:                    192 KiB (6 instances)
  L2:                     3 MiB (6 instances)
  L3:                     8 MiB (2 instances)
```


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
blis_12x8: 254.638 GFLOPS/s
```

gpu bench: (N = 2048)
```
baseline_cuda: 170.983 GFLOPS/s
gmem_coalesced: 1315.48 GFLOPS/s
smem_blocked: 1607.82 GFLOPS/s
smem_blocked2: 1649.84 GFLOPS/s
thread_blocked: 5399.37 GFLOPS/s
thread_blocked2: 3765.12 GFLOPS/s
```


**~~goal: 200 gflops~~ destroyed**

150 gflops on N = 1024. numpy gets 210. \\
250 gflops on N = 1920. numpy gets 280.


currently the blis 12x8 kernel requires N to be divisible by 12, so i can't use it with 
N = 1024. if i figure out how to handle N not divisible by 12, i should be able to get
a big boost on N = 1024. **todo** for now.


### resources

- [siboehm's article](https://siboehm.com/articles/22/fast-mmm-on-cpu)
- [marek's article](https://marek.ai/matrix-multiplication-on-cpu.html)
- [case study in algorithms for modern hardware](https://en.algorithmica.org/hpc/algorithms/matmul/)
- [blis paper](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
- [gotoblas paper](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_final.pdf)
- [avx blis implementation walkthrough and visualization by @riemannianmani](https://riemani.ca/blisgemm)
