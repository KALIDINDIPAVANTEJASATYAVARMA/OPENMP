#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>


#define N 1000

int A[N][N];
int B[N][N];
int C[N][N];
double start_time, end_time;
int main() 
{
    int i,j,k;
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic)
    for (i= 0; i< N; i++)
        for (j= 0; j< N; j++)
	{
            A[i][j] = 2;
            B[i][j] = 2;
	}
    start_time = omp_get_wtime();
    #pragma omp parallel for private(i,j,k) shared(A,B,C) schedule(dynamic)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end_time = omp_get_wtime();
    printf("Time taken: %f seconds\n", end_time - start_time); 

}