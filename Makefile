CXX=g++
CFLAGS=-march=native -fopenmp -O3
PERFFLAGS=cycles,cache-references,cache-misses,stalled-cycles-frontend,faults,l1_dtlb_misses,sse_avx_stalls

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

