# quantizer.py
import faiss
import numpy as np
import joblib
import os

def build_quantizer(embeddings_sample: np.ndarray, n_clusters: int, save_path: str):
    """
    Costruisce un quantizzatore K-means usando Faiss e lo salva su disco.
    """
    print(f"Building quantizer with {n_clusters} clusters using Faiss K-means...")
    d = embeddings_sample.shape[1]
    kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, verbose=True)
    kmeans.train(embeddings_sample)
    
    # Salviamo solo i centroidi, che sono ciò che ci serve per la predizione.
    centroids = kmeans.centroids
    joblib.dump(centroids, save_path)
    print(f"Quantizer saved to {save_path}")
    return centroids

def load_quantizer(path: str) -> np.ndarray:
    """Carica i centroidi del quantizzatore da un file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quantizer file not found at {path}. Please build it first.")
    print(f"Loading quantizer from {path}...")
    return joblib.load(path)

def predict_cluster(centroids: np.ndarray, vector: np.ndarray) -> int:
    """
    Assegna un vettore al cluster più vicino calcolando la distanza dai centroidi.
    """
    vector_reshaped = vector.reshape(1, -1)
    distances = np.linalg.norm(centroids - vector_reshaped, axis=1)
    return int(np.argmin(distances))