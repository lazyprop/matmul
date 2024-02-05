CC=gcc -mavx -mfma

vecadd: vecadd.c
	$(CC) vecadd.c -o vecadd

matmul: matmul.c
	$(CC) matmul.c -o matmul


