CXX=clang++
CFLAGS=-march=native -mavx -mfma -funroll-loops -ffast-math -fopenmp -g -O3

default:
	mkdir -p bin

vecadd: vecadd.cpp
	$(CXX) $(CFLAGS) vecadd.cpp -o vecadd

bench:
	$(CXX) $(CFLAGS) src/bench.cpp -o bin/bench
	bin/bench

play:
	$(CXX) $(CFLAGS) src/play.cpp -o bin/play
	bin/play


