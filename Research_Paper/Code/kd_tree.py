import numpy as np
import heapq
import itertools

class KDNode:
    def __init__ (self, word, vector, axis, left_child = None, right_child = None):
        self.word = word
        self.vector = vector
        self.axis = axis
        self.left = left_child
        self.right = right_child

class KDTree:
    def __init__ (self, dim):
        self.dim = dim
        self.root = None
        self.comparisons = 0
    
    def insert(self, word, vector):
        def _insert(node, word, vector, depth):
            if node is None:
                return KDNode(word, vector, depth % self.dim)

            axis = node.axis
            if vector[axis] < node.vector[axis]:
                node.left = _insert(node.left, word, vector, depth + 1)
            else:
                node.right = _insert(node.right, word, vector, depth + 1)
            return node

        self.root = _insert(self.root, word, vector, 0)

    def k_nearest_neighbors(self, target_vec, k=5):
        heap = []  # max-heap of (-distance, node)
        self.comparisons = 0
        counter = itertools.count()
        def euclidean_distance(a, b):
            # Increment comparisons
            self.comparisons += 1
            return np.linalg.norm(a - b)
        

        def _knn(node, depth):
            if node is None:
                return

            axis = node.axis
            dist = euclidean_distance(target_vec, node.vector)

            # Push to heap (as max-heap)
            heapq.heappush(heap, (-dist, next(counter), node))
            if len(heap) > k:
                heapq.heappop(heap)

            diff = target_vec[axis] - node.vector[axis]
            close_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            _knn(close_branch, depth + 1)

            # Check if we need to explore the other side
            if len(heap) < k or abs(diff) < -heap[0][0]:
                _knn(far_branch, depth + 1)

        _knn(self.root, 0)

        nearest =  sorted([(node.word, -d) for d, _, node in heap], key=lambda x: x[1])
        labels = [nearest[i][0] for i in range(len(nearest))]
        distances = [nearest[i][1] for i in range(len(nearest))]
        return labels, distances, self.comparisons

    def build_tree(self, words, vectors):
        if len(words) != len(vectors):
            print("Length mismatch")
            return
        for i in range(len(words)):
            self.insert(words[i], vectors[i])
        