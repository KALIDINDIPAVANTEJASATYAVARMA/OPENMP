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
    Edge* edges;
} Node;

typedef struct {
    int node;
    int dist;
} HeapNode;

typedef struct {
    HeapNode *nodes;
    int size;
    int capacity;
} MinHeap;

Node* graph[MAX_NODES];

void initialize_graph() {
    for (int i = 0; i < MAX_NODES; i++) {
        graph[i] = (Node*)malloc(sizeof(Node));
        if (graph[i] == NULL) {
            fprintf(stderr, "Error: Could not allocate memory for graph node %d\n", i);
            exit(EXIT_FAILURE);
        }
        graph[i]->num_edges = 0;
        graph[i]->edges = NULL;
    }
}

void add_edge(int src, int dst, int weight) {
    graph[src]->num_edges++;
    graph[src]->edges = (Edge*)realloc(graph[src]->edges, graph[src]->num_edges * sizeof(Edge));
    if (graph[src]->edges == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for edges of node %d\n", src);
        exit(EXIT_FAILURE);
    }
    graph[src]->edges[graph[src]->num_edges - 1].dst = dst;
    graph[src]->edges[graph[src]->num_edges - 1].weight = weight;
}

MinHeap* create_min_heap(int capacity) {
    MinHeap* minHeap = (MinHeap*)malloc(sizeof(MinHeap));
    minHeap->nodes = (HeapNode*)malloc(capacity * sizeof(HeapNode));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    return minHeap;
}

void swap_heap_node(HeapNode* a, HeapNode* b) {
    HeapNode t = *a;
    *a = *b;
    *b = t;
}

void min_heapify(MinHeap* minHeap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < minHeap->size && minHeap->nodes[left].dist < minHeap->nodes[smallest].dist)
        smallest = left;

    if (right < minHeap->size && minHeap->nodes[right].dist < minHeap->nodes[smallest].dist)
        smallest = right;

    if (smallest != idx) {
        swap_heap_node(&minHeap->nodes[smallest], &minHeap->nodes[idx]);
        min_heapify(minHeap, smallest);
    }
}

HeapNode extract_min(MinHeap* minHeap) {
    if (minHeap->size == 0)
        return (HeapNode){.node = -1, .dist = INF};

    HeapNode root = minHeap->nodes[0];
    minHeap->nodes[0] = minHeap->nodes[minHeap->size - 1];
    minHeap->size--;
    min_heapify(minHeap, 0);

    return root;
}

void decrease_key(MinHeap* minHeap, int node, int dist) {
    int i;
    for (i = 0; i < minHeap->size; ++i)
        if (minHeap->nodes[i].node == node)
            break;

    minHeap->nodes[i].dist = dist;
    while (i != 0 && minHeap->nodes[i].dist < minHeap->nodes[(i - 1) / 2].dist) {
        swap_heap_node(&minHeap->nodes[i], &minHeap->nodes[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

void dijkstra(int src, int dist[]) {
    bool visited[MAX_NODES];
    MinHeap* minHeap = create_min_heap(MAX_NODES);

    for (int i = 0; i < MAX_NODES; i++) {
        dist[i] = INF;
        visited[i] = false;
        minHeap->nodes[i].node = i;
        minHeap->nodes[i].dist = INF;
    }
    dist[src] = 0;
    minHeap->nodes[src].dist = 0;
    minHeap->size = MAX_NODES;

    while (minHeap->size) {
        HeapNode minHeapNode = extract_min(minHeap);
        int u = minHeapNode.node;

        if (visited[u])
            continue;

        visited[u] = true;

        #pragma omp parallel for
        for (int i = 0; i < graph[u]->num_edges; i++) {
            int v = graph[u]->edges[i].dst;
            int weight = graph[u]->edges[i].weight;

            if (!visited[v] && dist[u] != INF && dist[u] + weight < dist[v]) {
                #pragma omp critical
                {
                    if (dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                        decrease_key(minHeap, v, dist[v]);
                    }
                }
            }
        }
    }
    free(minHeap->nodes);
    free(minHeap);
}

int main(int argc, char* argv[]) {
    initialize_graph();

    // Read edges from Email-Enron.txt
    FILE *file = fopen("/home/graphfiles/Email-Enron.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file Email-Enron.txt\n");
        return 1;
    }

    int src, dst;
    while (fscanf(file, "%d\t%d", &src, &dst) != EOF) {
        if (src >= MAX_NODES || dst >= MAX_NODES) {
            fprintf(stderr, "Error: Node index out of bounds (src: %d, dst: %d)\n", src, dst);
            continue; // Skip invalid edges
        }
        add_edge(src, dst, 1);
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