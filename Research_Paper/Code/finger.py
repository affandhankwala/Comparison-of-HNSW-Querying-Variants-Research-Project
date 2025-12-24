import numpy as np

class FingerSurrogate:
    def __init__(self, dim, n_projections):
        self.W = np.random.randn(n_projections, dim).astype(np.float32)
        # Track comparisons
        self.comparisons = 0

    def train(self, queries, neighbors, true_dists, lr=1e-3, epochs=100):
        # Vectorized training
        for i in range(epochs):
            gradients, predictions = self.gradient(queries, neighbors, true_dists)
            self.W -= lr * gradients / len(queries) # Average gradient
            print("FINGER EPOCH: " + str(i))
            # predictions = self.predict(queries, neighbors)
            # errors = predictions - true_dists
            # gradients = (errors[:, np.newaxis] * np.sign(np.dot(differences, self.W))[:, np.newaxis] * differences)
            # grad = gradients.mean(axis = 0)
            # self.W -= lr * grad
        return self

    def build(self, data, graph, k=10, lr=1e-3, epochs=100):
        """
        Build/train the surrogate using knn from HNSW graph.

        Args:
            data: ndarray of shape (N, dim), the database.
            graph: the hnswlib.Index object already built.
            k: number of neighbors to use for training.
        """
        labels, dists = graph.knn_query(data, k=k)
        i_idx = np.repeat(np.arange(len(data)), k)
        j_idx = labels.flatten()
        queries = data[i_idx]
        neighbors = data[j_idx]
        true_dists = dists.flatten()
        self.train(queries, neighbors, true_dists, lr=lr, epochs=epochs)

    def gradient(self, queries, neighbors, true_dists):
        diffs = queries - neighbors
        diffs /= np.linalg.norm(diffs, axis=1, keepdims=True)
        projections = np.dot(diffs, self.W.T)
        predictions = np.sum(np.abs(projections), axis = 1)
        signs = np.sign(projections)
        errors = predictions - true_dists
        # Weight gradient
        error_signs = errors[:, np.newaxis] * signs
        gradients = np.einsum('bp,bd->pd', error_signs, diffs)
        return gradients, predictions


    def predict(self, query, neighbor):
        diff = query - neighbor
        projections = np.dot(self.W, diff)
        predictions = np.sum(np.abs(projections))
        self.comparisons += 1
        return predictions # simple approximation with all projections
    
    def search(self, hnsw_graph, q_point, k, data):
        self.comparisons = 0
        # Get candidate neighbors using HNSW graph
        candidate_labels, _ = hnsw_graph.knn_query(q_point, k=3*k)
        # Rerank using FINGER surrogate
        ranked = []
        for idx in candidate_labels[0]:
            neighbor = data[idx]
            score = self.predict(q_point[0], neighbor)
            ranked.append((idx, score))
        # Sort ranked by surrogate distance
        ranked.sort(key=lambda x: x[1])
        # Return top k
        top_k = ranked[:k]
        labels = [i for i, _ in top_k]
        scores = [i for _, i in top_k]
        return labels, scores, self.comparisons

# Utilization: 
# Replace dist=np.linalg.norm(query - candidtate)
# With: dist = finger_model.predict(query , candidate)