CXX=clang++
CFLAGS=-march=native -fopenmp -O3
PERFFLAGS=cycles,cache-references,cache-misses,stalled-cycles-frontend,faults,l1_dtlb_misses,sse_avx_stalls

default:
	mkdir -p bin

vecadd: vecadd.cpp
	$(CXX) $(CFLAGS) vecadd.cpp -o vecadd

bench:
	$(CXX) $(CFLAGS) src/bench.cpp -o bin/bench
	bin/bench

play:
	$(CXX) $(CFLAGS) src/play.cpp -o bin/play

run: play
	bin/play

gandalf: src/gandalf.cc
	g++ src/gandalf.cc -O3 -march=native -o bin/gandalf

perf: play gandalf
	perf stat -e $(PERFFLAGS) bin/play
	perf stat -e $(PERFFLAGS) bin/gandalf

