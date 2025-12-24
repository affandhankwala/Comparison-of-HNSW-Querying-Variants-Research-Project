from load_embeddings import load_glove_embeddings
from hnsw import HNSW
from brute_force import bf_search
from kd_tree import KDTree
import numpy as np

K = 10 

def embed_data():
    glove_path = "C:/Users/teamm/Documents/JHU-Masters/Summer_2025/605.746 Advanced Machine Learning/Research_Paper/Code/datasets/glove.twitter.27B/glove.twitter.27B.25d.txt"
    vectors, words = load_glove_embeddings(glove_path, 25)
    print("data embedded")
    return vectors, words

def hnsw(vectors, words):
    hnsw = HNSW(
        dim = vectors.shape[1],
        elements = vectors.shape[0],
        data = vectors,
        distance_metric = "cosine",
        ef_construction = 200, 
        ef_search = 50, 
        M = 16,
        construction_type = "default",
        querying_method = "default"
    )
    print("hnsw constructed")
  

    labels, distances, time = hnsw.search(query_vector, K)
    labels = [words[i] for i in labels]
    return labels, distances
        
# FINGER HNSW
def fingerhnsw(vectors, words):
    finger_hnsw = HNSW(
        dim = vectors.shape[1],
        elements = vectors.shape[0],
        data = vectors,
        distance_metric = "cosine",
        ef_construction = 200, 
        ef_search = 50, 
        M = 16,
        construction_type = "default",
        querying_method = "finger"
    )
    print("finger hnsw constructed")

    query_word = "happy"
    query_idx = words.index(query_word)
    query_vector = vectors[query_idx].reshape(1, -1)

    labels, distances, comparisons = finger_hnsw.search(query_vector, K)
    print(comparisons)
    labels = [words[i] for i in labels]
    return labels, distances

# Adaptive Beam search
def abshnsw(vectors, words):
    abs_hnsw = HNSW(
        dim = vectors.shape[1],
        elements = vectors.shape[0],
        data = vectors,
        distance_metric = "cosine",
        ef_construction = 200, 
        ef_search = 50, 
        M = 16,
        construction_type = "default",
        querying_method = "abs", 
        k = K
    )
    query_word = "happy"
    query_idx = words.index(query_word)
    query_vector = vectors[query_idx].reshape(1, -1)
    labels, distances, comparisons = abs_hnsw.search(query_vector, K)
    print(comparisons)
    labels = [words[i] for i in labels]
    return labels, distances

# Brute force
def bf(vectors, words):
    query_word = "happy"
    query_idx = words.index(query_word)
    query_vector = vectors[query_idx].reshape(1, -1)
    labels, distances =  bf_search(vectors, K, query_vector)
    labels = [words[i] for i in labels]
    return labels, distances

# KD Tree
def kd(words, vectors, query_vector): 
    kd_tree = KDTree(vectors.shape[1])
    kd_tree.build_tree(words, vectors)
    labels, distances, comparisons = kd_tree.k_nearest_neighbors(query_vector, K)
    print(comparisons)
    return labels, distances

vectors, words = embed_data()
query_word = "happy"
query_idx = words.index(query_word)
query_vector = vectors[query_idx].reshape(1, -1)
labels, distances = hnsw(vectors, words)
#labels, distances = fingerhnsw(vectors, words)
#labels, distances = abshnsw(vectors, words)
#labels, distances = bf(vectors, words)

# for i, labels in enumerate(labels) : 
#     print(words[labels], " ", distances[i])

#labels, distances = kd(words, vectors, query_vector[0])
for i in range(len(labels)):
    print(labels[i], " ", distances[i])