class Node:
    def __init__(self, idx, vector, M=5):
        self.idx = idx
        self.vector = vector
        self.neighbors = []  # list of neighbor node indices
        self.M = M