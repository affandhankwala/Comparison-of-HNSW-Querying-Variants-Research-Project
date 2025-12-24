from tests import test_D
from helper import get_base_dir
from load_embeddings import load_glove_embeddings
# Generate queries
# generate_words_vectors(get_dataset_dir() + "/glove.twitter.27B.25d.txt", 25, 1000)
# generate_words_vectors(get_dataset_dir() + "/glove.twitter.27B.50d.txt", 50, 1000)
# generate_words_vectors(get_dataset_dir() + "/glove.twitter.27B.100d.txt", 100, 1000)
# generate_words_vectors(get_dataset_dir() + "/glove.twitter.27B.200d.txt", 200, 1000)

# Extract words and vectors of queries
def generate_q(dim: int):
    queries_file_path = get_base_dir() + "queries" + str(dim) + "d.txt"
    q_vectors, q_words = load_glove_embeddings(queries_file_path, dim)
    print("Queries embedded")
    return q_vectors, q_words, queries_file_path
K_vals = [5, 50, 5]
# Generate true neighbors
#create_true_neighbors(get_dataset_dir() + "/glove.twitter.27B.25d.txt", queries_file_path, dimensions, K_vals, "25d/1/")
# create_true_neighbors(get_dataset_dir() + "/glove.twitter.27B.100d.txt", get_base_dir() + "queries100d.txt", 100, K_vals, "100d/1/")
# create_true_neighbors(get_dataset_dir() + "/glove.twitter.27B.200d.txt", get_base_dir() + "queries200d.txt", 200, K_vals, "200d/1/")
# create_true_neighbors(get_dataset_dir() + "/glove.twitter.27B.50d.txt", get_base_dir() + "queries50d.txt", 50, K_vals, "50d/1/")

# 25 dimensions 

def do(dims):
    for dim in dims:
        q_vectors, q_words, q_fp = generate_q(dim)
        test_D(q_vectors, q_words, q_fp, K_vals, dim=dim)

do([50, 100, 200])
