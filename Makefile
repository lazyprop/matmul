CC=clang
CFLAGS=-mavx -mfma -funroll-loops -ffast-math -fopenmp -g -O3

default:
	mkdir -p bin

vecadd: vecadd.c
	$(CC) $(CFLAGS) vecadd.c -o vecadd

matmul.o: src/matmul.c
	$(CC) $(CFLAGS) -c src/matmul.c -o bin/matmul.o

util.o: src/util.c
	$(CC) $(CFLAGS) -c src/util.c -o bin/util.o

bench: util.o matmul.o
	$(CC) $(CFLAGS) bin/util.o bin/matmul.o src/bench.c -o bin/bench
	bin/bench

play: util.o matmul.o
	$(CC) $(CFLAGS) bin/util.o bin/matmul.o src/play.c -o bin/play
	bin/play


