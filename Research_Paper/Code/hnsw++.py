from finger import FingerSurrogate
from adaptive_beam_search import AdaptiveBeamSearch
import numpy as np
import hnswlib

# --- LID Estimator ---
def estimate_lid(distances, k=10):
    r = distances[1:k+1]  # Skip the self-distance
    lid = -k / np.sum(np.log(r / (r[-1] + 1e-10)))
    return lid

# --- HNSW++ Class ---
class HNSWPlusPlus:
    def __init__(self, **kwargs):
        self.dim = kwargs.get("dim")
        self.max_elements = kwargs.get("max_elements", 10000)
        self.base_M = kwargs.get("base_M", 16)
        self.distance_metric = kwargs.get("distance_metric")
        self.ef_construction = kwargs.get("ef_construction", 200)
        self.graph = self.construct()
        self.querying_method = kwargs.get("querying_method")
        self.data = []
        self.skip_bridges = {}
        # Add FINGER precomputings if necessary
        if self.querying_method == "finger":
            self.finger_surrogate = FingerSurrogate(self.dim)
            self.finger_surrogate.build(self.data, self.graph, k = self.base_M)

    def construct (self):
        graph = hnswlib.Index(space = self.distance_metric, dim = self.dim)
        graph.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.base_M)
        return graph

    def add_items(self, vectors, k_lid=10, max_skips=5):
        self.data = vectors
        for i, vec in enumerate(vectors):
            if i == 0:
                self.graph.add_items(vec.reshape(1, -1))
                self.skip_bridges[i] = []
                continue

            # Estimate LID from neighbors
            labels, dists = self.graph.knn_query(vec, k=k_lid+1)
            lid = estimate_lid(dists[0], k=k_lid)

            # Add to graph
            self.graph.add_items(vec.reshape(1, -1))

            # Assign skip connections based on LID
            num_skips = int(np.clip(lid, 1, max_skips))
            skips = np.random.choice(i, size=num_skips, replace=False).tolist()
            self.skip_bridges[i] = skips

    def query(self, query_vec, k=10, traversal_steps=3):
        visited = set()
        candidates = [np.random.randint(0, len(self.data))]

        for _ in range(traversal_steps):
            next_candidates = []
            for node in candidates:
                if node not in visited:
                    visited.add(node)

                    # Main HNSW neighborhood
                    neighbors, _ = self.graph.knn_query(self.data[node], k=5)
                    for n in neighbors[0]:
                        if n not in visited:
                            next_candidates.append(n)

                    # Skip bridges
                    if node in self.skip_bridges:
                        for skip in self.skip_bridges[node]:
                            if skip not in visited:
                                next_candidates.append(skip)

            candidates = next_candidates

        # Final brute-force distance check among visited
        all_nodes = list(visited)
        dists = [np.linalg.norm(query_vec - self.data[n]) for n in all_nodes]
        top_k = np.argsort(dists)[:k]
        return [all_nodes[i] for i in top_k]

# Example usage
# Generate random data
dim = 128
num_points = 1000
data = np.random.randn(num_points, dim).astype(np.float32)

# Initialize and build HNSW++
hnswpp = HNSWPlusPlus(dim=dim, max_elements=num_points)
hnswpp.add_items(data)

# Query the structure
query = data[0] + np.random.randn(dim).astype(np.float32) * 0.05
results = hnswpp.query(query, k=10)

print("Top approximate neighbors (HNSW++):", results)
