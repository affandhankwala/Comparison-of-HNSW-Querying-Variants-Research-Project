import heapq
import numpy as np
import time

class AdaptiveBeamSearch:
    def __init__(self, alpha=1.2, max_beam_size=50):
        self.alpha = alpha
        self.max_beam_size = max_beam_size
        self.comparisons = 0

    def extract_graph_adjacency(self, elements, graph, data, M):
        # Extract Adjacency list from hnswlib (approximate version using knn_query)
        adj_list = {}
        for i in range(elements):
            # Use HNSW to query neighbors as approximation
            neighbors, _ = graph.knn_query(data[i].reshape(1, -1), k=M)
            adj_list[i] = neighbors[0].tolist()
        return adj_list


    def get_closest(self, query, start_nodes, graph, data, k=10):
        visited = set()
        candidate_queue = []
        top_k = []
        in_top_k = set()

        # Initialize
        for node in start_nodes:
            query.flatten()
            d = np.linalg.norm(query - data[node], axis = 0)
            self.comparisons += 1
            heapq.heappush(candidate_queue, (d, node))
            heapq.heappush(top_k, (-d, node))  # max heap for top-k
            in_top_k.add(node)

        while candidate_queue:
            dist_curr, curr = heapq.heappop(candidate_queue)
            if curr in visited:
                continue
            visited.add(curr)

            # Get neighbors from graph
            neighbors = graph.get(curr, [])
            for n in neighbors:
                if n in visited:
                    continue
                d = np.linalg.norm(query - data[n], axis = 0)
                self.comparisons += 1
                best_so_far = -top_k[0][0]
                if d <= self.alpha * best_so_far:
                    heapq.heappush(candidate_queue, (d, n))
                    # If node already in the top_k, then dont add it
                    if n not in in_top_k:
                        # Store the node
                        in_top_k.add(n)
                        heapq.heappush(top_k, (-d, n))
                    if len(top_k) > k:
                        _, n = heapq.heappop(top_k)
                        # Remove from set
                        in_top_k.remove(n)
            if len(visited) > self.max_beam_size:
                break

        result = sorted([(-d, node) for d, node in top_k])
        return [node for _, node in result]

    def search(self, q_point, elements, k, data, adj_list):
        # Pick one entry node. Random for now
        start_time = time.time()
        self.comparisons = 0
        start_nodes = [np.random.randint(elements)]
        indices = self.get_closest(
            q_point[0], start_nodes, adj_list, data, k
        )
        end_time = time.time()
        return indices, [np.linalg.norm(q_point[0] - data[i], axis = 0) for i in indices], self.comparisons, end_time - start_time

# Replace labels, dists = index.knn_query(query, k = 10)
# With:
# dabs = AdaptiveBeamSearch(distance_fn=np.linalg.norm, alpha=1.2)
# results = dabs.search(query, start_nodes=[entry_node], graph=hnsw_graph, data=data_vectors, k=10)
