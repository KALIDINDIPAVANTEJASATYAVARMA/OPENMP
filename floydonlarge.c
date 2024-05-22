#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <time.h>

#define N 40000

typedef struct {
    int dst;
    int weight;
} Edge;

// Node structure to store edges
typedef struct {
    int num_edges;
    Edge* edges;
} Node;

Node* graph[N];

void initialize_graph() {
    int i;
    for (i = 0; i < N; i++) {
        graph[i] = (Node*)malloc(sizeof(Node));
        graph[i]->num_edges = 0;
        graph[i]->edges = NULL;
    }
}

void add_edge(int src, int dst, int weight) {
    graph[src]->num_edges++;
    graph[src]->edges = (Edge*)realloc(graph[src]->edges, graph[src]->num_edges * sizeof(Edge));
    graph[src]->edges[graph[src]->num_edges - 1].dst = dst;
    graph[src]->edges[graph[src]->num_edges - 1].weight = weight;
}

void floyd_warshall_sparse() {
    int* dist = (int*)malloc(N * sizeof(int));
    int* next = (int*)malloc(N * sizeof(int));
    int k, i, j;
    #pragma omp parallel for 
    for (k = 0; k < N; k++) {
        // Parallel for loop over all nodes
        #pragma omp parallel for schedule(dynamic) private(i, j)
        for (i = 0; i < N; i++) {
            for (j = 0; j < graph[i]->num_edges; j++) {
                int dst = graph[i]->edges[j].dst;
                int weight = graph[i]->edges[j].weight;
                if (dist[dst] > dist[i] + weight) {
                    dist[dst] = dist[i] + weight;
                    next[dst] = i;
                }
            }
        }
    }

    free(dist);
    free(next);
}

int main(int argc, char* argv[]) {
    int num_threads = 12;  
    int i, j;

    initialize_graph();

    // Add edges with random weights between 0 and 19
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        for (j = 0; j < 10; j++) {  // Each node connects to 10 other nodes on average
            int dst = rand() % N;
            if (dst != i) {
                int weight = rand() % 20;
                add_edge(i, dst, weight);
            }
        }
    }

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();

    floyd_warshall_sparse();

    double elapsed_time = omp_get_wtime() - start_time;
    printf("Total time for thread %d (in sec): %.2f\n", num_threads, elapsed_time);

    // Free the graph memory
    for (i = 0; i < N; i++) {
        free(graph[i]->edges);
        free(graph[i]);
    }

    return 0;
}
