from hnsw import HNSW
from helper import (calculate_recall, get_base_dir, get_dataset_dir, get_neighbor_dir, 
                    get_true_neighbors, store_dict_in_txt, plot_three_trends_on_one, 
                    create_true_neighbors, generate_words_vectors, plot_comparisons
)
from load_embeddings import load_glove_embeddings, pull_dict_from_txt

# TEST: HNSW, FINGER, and ABS recall performance at various levels of K on various datasets
# Prints out a graph comparing all three
def measure_recall(words: list, hnsw: HNSW, finger: HNSW, abs: HNSW, q_words: list, q_vectors: list, 
                   K_vals: list, dataset_file_path: str, queries_file_path: str, test_name: str, dim: int):
    hnsw_recall_at_k, finger_recall_at_k, abs_recall_at_k = [], [], []
    finger_avg_comparisons, abs_avg_comparisons = [], []
    # Extract K vals
    k_min, k_max, k_inc = K_vals
    k = k_min
    ks = []
    # Search knn in each structure
    while k <= k_max:
        ks.append(k)
        hnsw_recall, finger_recall, abs_recall = 0, 0, 0
        # Get true neighbors
        true_label_dict = pull_dict_from_txt(get_neighbor_dir() + "True_" + test_name + str(k) + ".txt", k)
        hnsw_all_comparisons, finger_all_comparisons, abs_all_comparisons = [], [], []
        total_abs_time, total_hnsw_time = 0, 0
        # Query each vector
        for q in range(len(q_vectors)):
            # Extract relevant values
            query_vector = q_vectors[q].reshape(1, -1)
            query_word = q_words[q]
    
            hnsw_labels, hnsw_distances, hnsw_time = hnsw.search(query_vector, k)
            finger_labels, finger_distances, finger_comparisons = finger.search(query_vector, k)
            abs_labels, abs_distances, abs_comparisons, abs_time = abs.search(query_vector, k)
            # Convert the labels to words
            hnsw_predicted = [words[i] for i in hnsw_labels]
            finger_predicted = [words[i] for i in finger_labels]
            abs_predicted = [words[i] for i in abs_labels]
            # Determine recall of each and sum 
            hnsw_recall += calculate_recall(true_label_dict, query_word, hnsw_predicted)
            finger_recall += calculate_recall(true_label_dict, query_word, finger_predicted)
            abs_recall += calculate_recall(true_label_dict, query_word, abs_predicted)
            # Append Avg Comparisons
            finger_all_comparisons.append(finger_comparisons)
            abs_all_comparisons.append(abs_comparisons)
            # Increase abs search time
            total_abs_time += abs_time
            total_hnsw_time += hnsw_time
        # Store average recall at k
        hnsw_recall_at_k.append(hnsw_recall / len(q_vectors))
        finger_recall_at_k.append(finger_recall / len(q_vectors))
        abs_recall_at_k.append(abs_recall / len(q_vectors))
        # Store avg comparisons
        finger_avg_comparisons.append(sum(finger_all_comparisons) / len(finger_all_comparisons))
        abs_avg_comparisons.append(sum(abs_all_comparisons) / len(abs_all_comparisons))

        k += k_inc
    # Plot recall
    plot_three_trends_on_one(hnsw_recall_at_k, finger_recall_at_k, abs_recall_at_k, K_vals, str(dim))
    # Plot comparisons
    plot_comparisons(finger_avg_comparisons, abs_avg_comparisons, K_vals, str(dim), abs_time=abs_time, hnsw_time=hnsw_time)


def test_D(q_vectors: list, q_words: list, queries_file_path: str, K_vals: list, dim: int):
    print("Begin Test_"+str(dim)+"d")
    # Dataset file path
    dataset_file_path = get_dataset_dir() + "/glove.twitter.27B."+str(dim)+"d.txt"
    embeddings, words = load_glove_embeddings(dataset_file_path, dim)
    hnsw = HNSW(
        dim = embeddings.shape[1],
        elements = embeddings.shape[0],
        data = embeddings,
        distance_metric = "cosine",
        ef_construction = 200, 
        ef_search = 50, 
        M = 16,
        construction_type = "default",
        querying_method = "default"
    )
    print("HNSW built")
    finger = HNSW(
        dim = embeddings.shape[1],
        elements = embeddings.shape[0],
        data = embeddings,
        distance_metric = "cosine",
        ef_construction = 200, 
        ef_search = 100, 
        M = 16,
        construction_type = "default",
        querying_method = "finger"
    )
    print("FINGER built")
    abs = HNSW(
        dim = embeddings.shape[1],
        elements = embeddings.shape[0],
        data = embeddings,
        distance_metric = "cosine",
        ef_construction = 400, 
        ef_search = 100, 
        M = 32,
        construction_type = "default",
        querying_method = "abs",
        alpha = 1.2, 
        max_beam_size = 100
    )
    print("ABS built")
    # Measure recall
    measure_recall(words, hnsw, finger, abs, q_words, q_vectors, K_vals, dataset_file_path, queries_file_path, str(dim)+"d/1/", dim)
    print("Finish Test_"+str(dim)+"d")

