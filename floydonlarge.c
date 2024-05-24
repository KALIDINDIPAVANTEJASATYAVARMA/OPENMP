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

void floyd_warshall_sparse(int **dist) {
    int k, i, j;
    #pragma omp parallel for private(k, i, j) schedule(dynamic)
    for (k = 0; k < N; k++) {
        for (i = 0; i < N; i++) {
            if (dist[i][k] == INT_MAX) continue;
            for (j = 0; j < graph[i]->num_edges; j++) {
                int dst = graph[i]->edges[j].dst;
                if(dist[k][dst] == INT_MAX) continue;
                int weight = graph[i]->edges[j].weight;
                if (dist[i][dst] > dist[i][k] + dist[k][dst]) {
                    dist[i][dst] = dist[i][k] + dist[k][dst];
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int num_threads = 12;  

    initialize_graph();

    // Add edges with random weights between 0 and 19
    srand(time(NULL));
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < 1; j++) {  // Each node connects to 1 other node on average
            int dst = rand() % N;
            if (dst != i) {
                int weight = rand() % 20;
                add_edge(i, dst, weight);
            }
        }
    }

    // Allocate and initialize distance matrix
    int **dist = (int **)malloc(N * sizeof(int *));
    int k;
    for (i = 0; i < N; i++) {
        dist[i] = (int *)malloc(N * sizeof(int));
        for (j = 0; j < N; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else {
                dist[i][j] = INT_MAX;
            }
        }
    }

    // Initialize distances based on edges
    for (i = 0; i < N; i++) {
        for (j = 0; j < graph[i]->num_edges; j++) {
            int dst = graph[i]->edges[j].dst;
            int weight = graph[i]->edges[j].weight;
            dist[i][dst] = weight;
        }
    }

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();

    floyd_warshall_sparse(dist);

    double elapsed_time = omp_get_wtime() - start_time;
    printf("Total time for %d threads (in sec): %.2f\n", num_threads, elapsed_time);

    // Free the graph memory
    for (i = 0; i < N; i++) {
        free(graph[i]->edges);
        free(graph[i]);
        free(dist[i]);
    }
    free(dist);

    return 0;
}
