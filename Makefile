CC=gcc -mavx -mfma -funroll-loops -ffast-math -fopenmp

vecadd: vecadd.c
	$(CC) vecadd.c -o vecadd

matmul: matmul.c
	$(CC) matmul.c -o matmul


