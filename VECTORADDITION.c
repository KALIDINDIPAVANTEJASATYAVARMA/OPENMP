#include <stdlib.h>   
#include <stdio.h>    
#include <omp.h>      

#define ARRAY_SIZE 1000000     
#define NUM_THREADS 12    

int main (int argc, char *argv[]) {
    // elements of arrays a and b will be added
    // and placed in array c
    int *a;
    int *b; 
    int *c;
        
    int n = ARRAY_SIZE;                 
    int total_threads = NUM_THREADS;    
    int i;       // loop index

    double start_time, end_time; 

    a = (int *) malloc(sizeof(int) * n);
    b = (int *) malloc(sizeof(int) * n);
    c = (int *) malloc(sizeof(int) * n);

    for(i = 0; i < n; i++) {
        a[i] = i;
    }
    for(i = 0; i < n; i++) {
        b[i] = i;
    }   

    omp_set_num_threads(total_threads);

    start_time = omp_get_wtime();

    #pragma omp parallel for shared(a, b, c) private(i) schedule(dynamic)
    for(i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    end_time = omp_get_wtime(); 

    printf("Time taken: %f seconds\n", end_time - start_time); 
	

    free(a);  
    free(b); 
    free(c);
	
    return 0;
}
