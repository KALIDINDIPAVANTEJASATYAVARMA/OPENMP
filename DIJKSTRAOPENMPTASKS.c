#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_NODES 36692
#define INF INT_MAX

typedef struct {
    int dst;
    int weight;
} Edge;

typedef struct {
    int num_edges;
    int capacity;
    Edge* edges;
} Node;

Node* graph[MAX_NODES];

void initialize_graph() {
    for (int i = 0; i < MAX_NODES; i++) {
        graph[i] = (Node*)malloc(sizeof(Node));
        if (graph[i] == NULL) {
            fprintf(stderr, "Error: Could not allocate memory for graph node %d\n", i);
            exit(EXIT_FAILURE);
        }
        graph[i]->num_edges = 0;
        graph[i]->capacity = 4; // Initial capacity for edges
        graph[i]->edges = (Edge*)malloc(graph[i]->capacity * sizeof(Edge));
        if (graph[i]->edges == NULL) {
            fprintf(stderr, "Error: Could not allocate memory for edges of node %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void add_edge(int src, int dst, int weight) {
    if (graph[src]->num_edges >= graph[src]->capacity) {
        graph[src]->capacity *= 2;
        graph[src]->edges = (Edge*)realloc(graph[src]->edges, graph[src]->capacity * sizeof(Edge));
        if (graph[src]->edges == NULL) {
            fprintf(stderr, "Error: Could not allocate memory for edges of node %d\n", src);
            exit(EXIT_FAILURE);
        }
    }
    graph[src]->edges[graph[src]->num_edges].dst = dst;
    graph[src]->edges[graph[src]->num_edges].weight = weight;
    graph[src]->num_edges++;
}

void dijkstra(int src, int dist[]) {
    bool visited[MAX_NODES];
    int i, u, v;

    #pragma omp parallel for
    for (i = 0; i < MAX_NODES; i++) {
        dist[i] = INF;
        visited[i] = false;
    }
    dist[src] = 0;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int count = 0; count < MAX_NODES - 1; count++) {
                int min_dist = INF;

                #pragma omp task shared(u, min_dist)
                {
                    for (v = 0; v < MAX_NODES; v++) {
                        if (!visited[v] && dist[v] < min_dist) {
                            #pragma omp critical
                            {
                                if (!visited[v] && dist[v] < min_dist) {
                                    min_dist = dist[v];
                                    u = v;
                                }
                            }
                        }
                    }
                }
                #pragma omp taskwait

                if (u == -1) break; // All reachable nodes have been visited

                visited[u] = true;

                #pragma omp parallel for shared(dist, visited) private(v)
                for (i = 0; i < graph[u]->num_edges; i++) {
                    v = graph[u]->edges[i].dst;
                    int weight = graph[u]->edges[i].weight;
                    if (!visited[v] && dist[u] != INF && dist[u] + weight < dist[v]) {
                        #pragma omp critical
                        {
                            if (dist[u] + weight < dist[v]) {
                                dist[v] = dist[u] + weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    initialize_graph();

    // Read edges from email-Enron.txt
    FILE *file = fopen("/home/graphfiles/Email-Enron.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file email-Enron.txt\n");
        return 1;
    }

    int src, dst;
    while (fscanf(file, "%d\t%d", &src, &dst) != EOF) {
        if (src >= MAX_NODES || dst >= MAX_NODES) {
            fprintf(stderr, "Error: Node index out of bounds (src: %d, dst: %d)\n", src, dst);
            continue; // Skip invalid edges
        }
        add_edge(src, dst, 1);
        add_edge(dst, src, 1); // For undirected graph
    }
    fclose(file);

    for (int num_threads = 16; num_threads >= 1; num_threads--) {
        omp_set_num_threads(num_threads);

        double start_time = omp_get_wtime();

        int dist[MAX_NODES];
        dijkstra(0, dist);

        double elapsed_time = omp_get_wtime() - start_time;
        printf("Total time for %d threads (in sec): %.2f\n", num_threads, elapsed_time);
    }

    // Free the graph memory
    for (int i = 0; i < MAX_NODES; i++) {
        free(graph[i]->edges);
        free(graph[i]);
    }

    return 0;
}