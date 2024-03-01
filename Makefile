CC=gcc
CFLAGS=-mavx -mfma -funroll-loops -ffast-math -fopenmp

vecadd: vecadd.c
	$(CC) $(CFLAGS) vecadd.c -o vecadd

run: util.h matmul.c
	$(CC) $(CFLAGS) matmul.c -o matmul
	./matmul


