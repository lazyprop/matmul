CXX=g++
CFLAGS=-march=native -O3 -fopenmp
PERFFLAGS=cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,faults,l1_dtlb_misses,sse_avx_stalls,L1-icache-loads,L1-icache-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,l3_accesses,l3_misses

bench: bench.cpp
	$(CXX) $(CFLAGS) bench.cpp -o bench

play: play.cpp
	$(CXX) $(CFLAGS) play.cpp -o play

baseline: baseline.cpp
	$(CXX) $(CFLAGS) baseline.cpp -o baseline

gandalf: gandalf.cc
	g++ gandalf.cc -O3 -march=native -o gandalf

perf: play
	perf stat -e $(PERFFLAGS) ./play

