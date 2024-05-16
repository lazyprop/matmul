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

layered:
	$(CXX) $(CFLAGS) src/layered.cpp -o bin/layered

run: layered
	bin/layered

gandalf: src/gandalf.cc
	g++ src/gandalf.cc -O3 -march=native -o bin/gandalf

perf: play gandalf layered
	perf stat -e $(PERFFLAGS) bin/layered
	perf stat -e $(PERFFLAGS) bin/gandalf

