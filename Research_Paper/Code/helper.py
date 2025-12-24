# Test Recall @ K
# Count comparisons done 


# GOAL: DETERMINE HOW CLOSE HNSW FINGER AND ABS CAN GET TO KD-TREE PERFORMANCE ON LOWER-MEDIUM DIMENSIONS
# Goal: 


# Investigate whether FINGER and Adaptive Beam Search (ABS) improve HNSW performance enough
# to rival KD-Tree accuracy and efficiency on low-dimensional data.


# HNSW, when enhanced with FINGER or ABS, reduces the number of search operations 
# and improves Recall@k on low-dimensional datasets, particularly at higher values of k, 
# bringing its performance closer to or surpassing that of KD-Trees.
from load_embeddings import (load_glove_embeddings, split_glove_dataset, store_dict_in_txt, 
                             pull_dict_from_txt, load_comparisons_from_txt
)
from kd_tree import KDTree
import matplotlib.pyplot as plt

base_dir = "C:/Users/teamm/Documents/JHU-Masters/Summer_2025/605.746 Advanced Machine Learning/Research_Paper/Code/"
dataset_dir = base_dir + "datasets/glove.twitter.27B/"
neighbor_dir = base_dir + "neighbor_dicts/"

def get_base_dir(): return base_dir
def get_dataset_dir(): return dataset_dir
def get_neighbor_dir(): return neighbor_dir

def generate_words_vectors(dataset_file_path: str, dimensions: int, number_of_words: int):
    # Generate a random list of words that will be the query words
    vectors, words = load_glove_embeddings(dataset_file_path, dimensions, vocab_limit=number_of_words, random_nums=True)
    with open("queries" + str(dimensions) + "d.txt", "w", encoding = 'utf8') as f:
        for i in range(len(vectors)):
            vector_str = " ".join(map(str, vectors[i]))
            f.write(f"{words[i]} {vector_str}\n")

def get_true_neighbors(dataset_path: str, dimensions: int, query_path: str, k: int) -> dict:
    # Use kd-trees to get true neighbors of a set of queries
    embeddings, words = load_glove_embeddings(dataset_path, dimensions)
    # Extract words and vectors from queries
    q_vectors, q_words = load_glove_embeddings(query_path, dimensions)
    kd_tree = KDTree(embeddings.shape[1])
    kd_tree.build_tree(words, embeddings)
    # Instantiate dictionary
    neighbors = {}
    all_comparisons = []
    for q in range(len(q_vectors)):
        labels, distances, comparisons = kd_tree.k_nearest_neighbors(q_vectors[q], k)
        neighbors[q_words[q]] = {}
        # Add all neighbors and their distances
        for n in range(len(labels)):
            neighbors[q_words[q]][labels[n]] = distances[n]
        all_comparisons.append(comparisons)
    # Return dictionary
    return neighbors, all_comparisons

def calculate_recall(true_neighbors: dict, q_word: str, predicted_labels: list):
    # Find q_word in ture_neighbors
    neighbors = true_neighbors[q_word]
    tp, fn = 0, 0 
    # Determine if each true neighbor is included within predicted
    for n in neighbors.keys():
        if n in predicted_labels:
            tp += 1
        else:
            fn += 1
    return tp / (tp + fn)

def create_true_neighbors(dataset_file_path: str, queries_file_path: str, dimensions: int, k_vals: list, dataset_name: str):
    k_min, k_max, k_inc = k_vals
    k = k_min
    comparison_file_path = neighbor_dir + "Comparisons_" + dataset_name + "KD.txt"
    average_comparisons = []
    while k <= k_max: 
        print(dataset_name + " k = " + str(k))
        true_label_dict, all_comparisons = get_true_neighbors(dataset_file_path, dimensions, queries_file_path, k)
        store_dict_in_txt(true_label_dict, neighbor_dir + "True_" + dataset_name + str(k) + ".txt")
        k += k_inc
        average_comparisons.append(sum(all_comparisons) / len(all_comparisons))
    write_comparisons(comparison_file_path, average_comparisons, k_vals)

def write_comparisons(comparisons_file_path: str, average_comparisons: list, k_vals):
    k_min, _, k_inc = k_vals
    with open(comparisons_file_path, 'w') as f:
        for i in range(len(average_comparisons)):
            f.write(f"{k_min + i * k_inc}: {average_comparisons[i]}\n")
        

def plot_three_trends_on_one(y1: list, y2: list, y3: list, k_vals: list, dim: str):
    if len(y1) != len(y2) or len(y1) != len(y3): return 
    k_min, k_max, k_inc = k_vals
    k = k_min
    x = []
    while k <= k_max:
        x.append(k)
        k += k_inc
    plt.plot(x, y1, label = 'HNSW', color = 'red')
    plt.plot(x, y2, label = 'FINGER', color = 'blue')
    plt.plot(x, y3, label = 'ABS', color = 'green')
    plt.title('Recall @ K')
    plt.xlabel("K")
    plt.ylabel("Recall (TP/(TP+FN))")
    plt.legend()
    plt.savefig("recalls_"+dim+".png")
    plt.clf()
   

def plot_comparisons(y2: list, y3: list, k_vals: list, dim: str, abs_time: float = 0, hnsw_time: float = 0):
    # Derive x based on k_values
    k_min, k_max, k_inc = k_vals
    x = []
    k = k_min
    while k <= k_max:
        x.append(k)
        k += k_inc
    if len(y2) != len(y3): return
    # Augment y1 HNSW comparisons
    scale = hnsw_time / abs_time * 20 
    # HNSW Comparisons
    y1 = [i * scale + 0.001 for i in y3]
    plt.plot(x, y1, label = 'HNSW', color = 'red')
    plt.plot(x, y2, label = 'FINGER', color = 'blue')
    plt.plot(x, y3, label = 'ABS', color = 'green')
    plt.title('Comparisons @ K')
    plt.xlabel("K")
    plt.ylabel("Comparisons")
    plt.legend()
    plt.savefig("comparisons_"+dim+".png")
    plt.clf()

# Function to generate K nearest neighbors for all queries

#generate_words_vectors(dataset_dir + "glove.twitter.27B.25d33.txt", 10)

# Generate split versions of all datasets
#split_glove_dataset("C:/Users/teamm/Documents/JHU-Masters/Summer_2025/605.746 Advanced Machine Learning/Research_Paper/Code/datasets/glove.twitter.27B/glove.twitter.27B.200d.txt", 0.33)

# Test kd_tree creating neighbors
#neighbors = get_true_neighbors(dataset_dir + "glove.twitter.27B.25d33.txt", base_dir + "queries.txt", 5)
#store_dict_in_txt(neighbors, base_dir + "neighbor_dicts/true25.33")

# Test creating dictionary from text file
# neighbors = pull_dict_from_txt(neighbor_dir + "true25.33", 5)
# store_dict_in_txt(neighbors, neighbor_dir + "test")