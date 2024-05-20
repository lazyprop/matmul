CXX=g++
CFLAGS=-march=native -fopenmp -O3
PERFFLAGS=cycles,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,faults,l1_dtlb_misses,sse_avx_stalls,L1-icache-loads,L1-icache-load-misses,dTLB-loads,dTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses

vecadd: vecadd.cpp
	$(CXX) $(CFLAGS) vecadd.cpp -o vecadd

bench: bench.cpp
	$(CXX) $(CFLAGS) bench.cpp -o bench
	./bench

play: play.cpp
	$(CXX) $(CFLAGS) play.cpp -o play

baseline: baseline.cpp
	$(CXX) $(CFLAGS) baseline.cpp -o baseline


layered: layered.cpp
	$(CXX) $(CFLAGS) layered.cpp -o layered

run: play
	./play

gandalf: gandalf.cc
	g++ gandalf.cc -O3 -march=native -o gandalf

perf: play gandalf
	perf stat -e $(PERFFLAGS) ./play
	//perf stat -e $(PERFFLAGS) ./gandalf

