from typing import List
import numpy as np
from sklearn.cluster import KMeans


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

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
