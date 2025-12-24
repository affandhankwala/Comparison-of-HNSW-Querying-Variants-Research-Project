import numpy as np
import hnswlib

# Estiamte local density 
def local_density(distances, k=10):
    # Avoid self-distance at index 0
    return np.mean(distances[1:k+1])


class DHNSW:
    def __init__(self, dim, max_elements=10000, base_M=4096):
        self.dim = dim
        self.max_elements = max_elements
        self.base_M = base_M
        self.index = hnswlib.Index(space='l2', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=base_M)
        self.data = []
        self.local_M = []
        self.local_density_scores = []

    def build_index(self, data, k=10):
        self.data = data
        for i in range(len(data)):
            if i == 0:
                # First point — no neighbors yet
                self.index.add_items(data[i].reshape(1, -1))
                self.local_M.append(self.base_M)
                self.local_density_scores.append(0.0)
                continue

            # Find nearest neighbors so far
            labels, dists = self.index.knn_query(data[i], k=k+1)
            density = local_density(dists[0], k=k)

            # Convert density to M_i using an inverse scale
            M_i = int(np.clip(self.base_M * (1 / (density + 1e-5)), 8, 48))

            self.index.add_items(data[i].reshape(1, -1))
            self.index.set_ef(ef=int(1.5 * M_i))

            self.local_M.append(M_i)
            self.local_density_scores.append(density)

    def query(self, q, k=10):
        # Estimate query difficulty using neighbors in full data
        labels, dists = self.index.knn_query(q, k=10)
        density = local_density(dists[0])
        
        ef_q = int(np.clip(100 * (1 / (density + 1e-5)), 10, 300))
        self.index.set_ef(ef_q)

        return self.index.knn_query(q, k=k)
    


# Example usage

# Dummy data
# dim = 128
# num_elements = 10000
# data = np.random.randn(num_elements, dim).astype(np.float32)

# dhnsw = DHNSW(dim=dim)
# dhnsw.build_index(data)

# # Query
# query_vector = data[0] + np.random.randn(dim).astype(np.float32) * 0.1
# labels, distances = dhnsw.query(query_vector, k=10)

# print("Top results:", labels)

