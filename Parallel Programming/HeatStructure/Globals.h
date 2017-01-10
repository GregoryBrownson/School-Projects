#include <omp.h>

#define BLOCK_SIZE 32

#define MAX_THREADS omp_get_max_threads() * 2 / 3;