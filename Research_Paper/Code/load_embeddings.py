import random
import numpy as np
import re

def load_glove_embeddings (file_path: str, dimensions: int, vocab_limit: int=None, random_nums: bool = False):
    embeddings = []
    words = []
    with open (file_path, 'r', encoding = 'utf8') as f:
        if random_nums:
            # Length of file
            lines = f.readlines()
            # Generate a random list of indexes
            random_indexes = random.sample(range(len(lines) - 1), vocab_limit)
            # Extract words and vectors at those indices
            
            for i in random_indexes:
                parts = lines[i].strip().split()
                word = parts[0]
                vector  = np.array(parts[1:], dtype=np.float32)
                if len(vector) != dimensions:
                    # Skip
                    continue
                embeddings.append(vector)
                words.append(word)
            return np.vstack(embeddings), words

        for i, line in enumerate(f):
            if vocab_limit and i >= vocab_limit: break;
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if len(vector) != dimensions:
                # skip this. Unknown word
                continue
            embeddings.append(vector)
            
            words.append(word)
        return np.vstack(embeddings), words
    
def split_glove_dataset(i_file_path: str, split_proportion: float):
    # Split a glove dataset into a smaller proportion and create new file with those
    with open(i_file_path, 'r', encoding = 'utf8') as f_i:
        lines = f_i.readlines()
    number_of_lines = int (len(lines) * split_proportion)
    # Generate random numbers for which lines to select
    random_indices = random.sample(range(len(lines) - 1), number_of_lines)
    o_file_path = i_file_path[:-4] + (str)(round(split_proportion * 100)) + '.txt'
    with open (o_file_path, 'w', encoding = 'utf8') as f_o:
        for i in random_indices:
            f_o.write(lines[i])
        
def store_dict_in_txt(d: dict, file_name: str):
    with open(file_name, 'w', encoding = 'utf-8') as f:
        for query_name, neighbors in d.items():
            f.write(f"{query_name}\n")
            # Write all neighbors
            for neighbor, distance in neighbors.items():
                # Tab
                f.write("    ")
                f.write(f"({neighbor}, {distance})\n")

def pull_dict_from_txt(file_path: str, k: int) -> dict:
    neighbors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        f = file.readlines()
        # Loop through entire file
        i = 0 
        while i < len(f):
            # First line is query
            query = f[i][:-1]
            neighbors[query] = {}
            i+=1
            # Next k lines are neighbors
            j = 0
            while (j < k):
                # Remove leading whitespace
                parts = f[i + j].strip()
                # Split along ( and , and )
                parts = re.split(r'[(),]', parts)
                # Second part is word, Third is distance
                neighbor, distance= parts[1], parts[2]
                # Add to dictionary
                neighbors[query][neighbor] = distance
                j += 1
            # Move i pointer to next query word
            i += j
    return neighbors

def load_comparisons_from_txt(file_path: str) -> dict:
    comparisons = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.split(':')
            k = parts[0]
            c = float(parts[1].strip())
            comparisons[k] = c
    return comparisons
