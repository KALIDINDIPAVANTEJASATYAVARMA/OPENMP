#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <omp.h>
#include "../../graph.hpp"

const int INF = std::numeric_limits<int>::max();

struct Edge {
    int src, dst, weight;
    Edge(int s, int d, int w) : src(s), dst(d), weight(w) {}
};

struct Bucket {
    std::vector<std::vector<int>> buckets;
    int delta;

    Bucket(int num_nodes, int delta) : delta(delta) {
        buckets.resize((num_nodes * delta) / delta + 1);
    }

    void insert(int node, int dist) {
        int index = dist / delta;
        buckets[index].push_back(node);
    }

    std::vector<int> getMinBucket() {
        for (auto& bucket : buckets) {
            if (!bucket.empty()) {
                return bucket;
            }
        }
        return std::vector<int>();
    }

    void clearMinBucket() {
        for (auto& bucket : buckets) {
            if (!bucket.empty()) {
                bucket.clear();
                break;
            }
        }
    }

    bool empty() {
        for (auto& bucket : buckets) {
            if (!bucket.empty()) {
                return false;
            }
        }
        return true;
    }
};

void Compute_SSSP(graph& g, int* weight, int* dist, int src, int delta) {
    int num_nodes = g.num_nodes();
    std::vector<int> distances(num_nodes, INF);
    Bucket bucket(num_nodes, delta);

    distances[src] = 0;
    bucket.insert(src, 0);

    while (!bucket.empty()) {
        std::vector<int> minBucket = bucket.getMinBucket();
        bucket.clearMinBucket();

        std::vector<std::pair<int, int>> buffer;

        #pragma omp parallel for
        for (int i = 0; i < minBucket.size(); ++i) {
            int u = minBucket[i];
            for (int edge = g.indexofNodes[u]; edge < g.indexofNodes[u + 1]; ++edge) {
                int v = g.edgeList[edge];
                int new_dist = distances[u] + weight[edge];
                #pragma omp critical
                {
                    if (new_dist < distances[v]) {
                        distances[v] = new_dist;
                        buffer.push_back(std::make_pair(v, new_dist / delta));
                    }
                }
            }
        }

        for (auto& entry : buffer) {
            bucket.insert(entry.first, entry.second * delta);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; ++i) {
        dist[i] = distances[i];
    }
}

int main() {
    graph G("../../dataRecords/as-skitter.txt");
    G.parseGraph();

    int* edgeLen = G.getEdgeLen();
    int* dist = new int[G.num_nodes() + 1];

    int src = 1;
    int delta = 100;  // Example delta value
    double startTime = omp_get_wtime();
    Compute_SSSP(G, edgeLen, dist, src, delta);
    double endTime = omp_get_wtime();
    printf("RunTime : %f\n", endTime - startTime);

    for (int i = 0; i < 10; ++i) {
        printf("%d  %d\n", i, dist[i]);
    }

    delete[] dist;
    return 0;
}
