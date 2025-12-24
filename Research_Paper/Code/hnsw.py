from finger import FingerSurrogate
from adaptive_beam_search import AdaptiveBeamSearch
from node import Node
import hnswlib
import time

class HNSW:
    def __init__ (self, **kwargs):
        # Retrieve dataset parameters
        self.dim = kwargs.get("dim")
        self.elements = kwargs.get("elements")
        self.data = kwargs.get("data")
        self.distance_metric = kwargs.get("distance_metric")
        self.ef_construction = kwargs.get("ef_construction")
        self.ef_search = kwargs.get("ef_search")
        self.M = kwargs.get("M")
        self.construction_type = kwargs.get("construction_type", "default")
        self.querying_method = kwargs.get("querying_method", "default")
        self.k = kwargs.get("k", 10)
        self.alpha = kwargs.get("alpha", 1.2)
        self.max_beam_size = kwargs.get("max_beam_size", 30)
        self.n_projections = kwargs.get("n_projections", 16)
        self.graph = self.construct()
        # Add FINGER precomputings if necessary
        if self.querying_method == "finger":
            self.finger_surrogate = FingerSurrogate(self.dim, self.n_projections)
            self.finger_surrogate.build(self.data, self.graph, k=self.M)
        # Add Adaptive beam search (abs) parameters
        elif self.querying_method == "abs":
            self.abs = AdaptiveBeamSearch(self.alpha, self.max_beam_size)
            self.adj_list = self.abs.extract_graph_adjacency(self.elements, self.graph, self.data, self.M)
    
    def search(self, query, k):
        # Greedy beam search
        # Start at entry point
        # Returns labels, distances
        if self.querying_method == "finger": 
            return self.finger_surrogate.search(self.graph, query, k, self.data)
        elif self.querying_method == "abs":
            return self.abs.search(query, self.elements, k, self.data, self.adj_list)
        start_time = time.time()
        labels, distances = self.graph.knn_query(query, k)
        end_time = time.time()
        return labels[0], distances[0], end_time - start_time

    def construct(self):
        graph = hnswlib.Index(space=self.distance_metric, dim=self.dim)
        graph.init_index(max_elements=self.elements, ef_construction=self.ef_construction, M=self.M)
        graph.add_items(self.data)
        graph.set_ef(self.ef_search)
        return graph

    