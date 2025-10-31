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
    X = np.array(vectors)

    # Aggiungiamo un controllo: non possiamo trovare k cluster se abbiamo meno di k vettori
    if X.shape[0] < k:
        raise ValueError(f"Errore: Impossibile trovare {k} cluster con solo {X.shape[0]} vettori.")
    
    # 2. Initialize the KMeans model
    kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    
    # 3. Fit the model to the data
    kmeans_model.fit(X)
    
    # 4. Get the centroids from the fitted model
    centroids = kmeans_model.cluster_centers_
    
    return centroids

# --- Example Usage ---
# Questo blocco viene eseguito SOLO se avvii questo file direttamente
# (es. 'python k-means.py')
# NON viene eseguito quando 'qdrant_app.py' lo importa.
if __name__ == "__main__":
    
    print("--- Test della libreria K-Means ---")

    # 1. Definiamo un piccolo set di vettori di test
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
    k = 2

    print(f"Trovando {k} centroidi da {len(my_vectors)} vettori di test...")

    try:
        # 3. Call the function
        calculated_centroids = find_kmeans_centroids(my_vectors, k)

        print("\nCentroidi Calcolati:")
        print(calculated_centroids)
        
    except ValueError as e:
        print(f"Errore: {e}")