import numpy as np
from sklearn.cluster import KMeans
import json

def find_kmeans_centroids(vectors, k):
    """
    Calculates k-means centroids from a list of vectors using scikit-learn.

    Args:
        vectors (list or np.array): A list of vectors (e.g., [[1, 2], [3, 4]]).
        k (int): The number of centroids (clusters) to find.

    Returns:
        np.array: An array where each row is a calculated centroid vector.
    """
    
    # 1. Convert list to a NumPy array
    # scikit-learn works best with NumPy arrays
    X = np.array(vectors)
    
    # 2. Initialize the KMeans model
    # n_init='auto' is the modern default to avoid warnings
    # random_state=42 makes the result reproducible (no random start)
    kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    
    # 3. Fit the model to the data
    kmeans_model.fit(X)
    
    # 4. Get the centroids from the fitted model
    centroids = kmeans_model.cluster_centers_
    
    return centroids

# --- Example Usage ---

# 1. Define your list of vectors
my_vectors = [
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [6, 7],
    [1.2, 1.0]
]

NUM_VECTORS = 1000
VECTOR_SIZE = 384


data = {}
try:
    with open('embeddings.json', 'r') as f:
        data = json.load(f)
    if len(data) < NUM_VECTORS + 1:
        print(f"Warning: embeddings.json has only {len(data)} items, but {NUM_VECTORS}+1 are needed.")
        # Pad with random data if insufficient
        for i in range(len(data), NUM_VECTORS + 1):
            data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})
except FileNotFoundError:
    print("embeddings.json not found. Generating random data...")
    for i in range(NUM_VECTORS + 1):
        data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})

my_vectors = [i['embedding'] for i in data]

# 2. Define how many centroids you want
k = 3

# 3. Call the function
calculated_centroids = find_kmeans_centroids(my_vectors, k)

print(type(calculated_centroids.tolist()))

print(f"Data: {len(my_vectors)} vectors")
print(f"Finding k={k} centroids...\n")
print("Calculated Centroids:")
print(calculated_centroids)