import heapq
import numpy as np
def bf_search (elements, k, query):
    # Compare each element with query and retain top k
    top_k = []
    for i, e in enumerate(elements): 
        # Compare distance between two (linear)
        d = np.linalg.norm(query[0] - e, axis = 0)
        heapq.heappush(top_k, (-d, i))
        if len(top_k) > k:
            heapq.heappop(top_k)
        if i % 1000 == 0: print(i)
    result = sorted([(-d, i) for d, i in top_k])
    indices = [r[1] for r in result]
    distances = [r[0] for r in result]
    return indices, distances