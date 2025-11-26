import numpy as np
import json
import faiss
from sklearn.metrics import silhouette_score
import time
import torch
import itertools
from typing import List, Tuple, Dict, Union
import heapq
from compound_types import *
import hnswlib
import pickle
import base64
import logging

SILHOUETTE_SUBSAMPLE_SIZE = 2000  # Max sample size for silhouette score
KMEANS_N_ITER = 150  # Reduced n_iter for KMeans

logger = logging.getLogger(__name__)

# --- PYTORCH KMEANS IMPLEMENTATION ---
def kmeans_torch(X_tensor, num_clusters, niter=15, tol=1e-4, device='cpu'):
    """
    Esegue KMeans usando PyTorch su GPU (MPS, CUDA) o CPU.
    """
    n_samples, n_features = X_tensor.shape
    
    # 1. Inizializzazione casuale dei centroidi
    random_indices = torch.randperm(n_samples)[:num_clusters]
    centroids = X_tensor[random_indices].clone()
    
    for i in range(niter):
        previous_centroids = centroids.clone()
        
        # 2. Calcolo distanze (Broadcasting)
        # (N, 1, D) - (1, K, D) -> (N, K, D) norm -> (N, K)
        distances = torch.cdist(X_tensor, centroids)
        
        # 3. Assegnazione cluster
        labels = torch.argmin(distances, dim=1)
        
        # 4. Aggiornamento centroidi
        # Questo ciclo può essere vettorizzato ma per K piccolo è veloce anche così
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, device=device).unsqueeze(1)
        
        # Scatter add è molto veloce su GPU per sommare in base agli indici
        # One-hot encoding implicito per sommare
        # Per semplicità e stabilità usiamo un approccio masked semplice o scatter_add_
        
        # Metodo veloce PyTorch per ricalcolo media:
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                new_centroids[k] = X_tensor[mask].mean(dim=0)
            else:
                # Gestione cluster vuoti: riassegna a un punto random
                random_idx = torch.randint(0, n_samples, (1,)).item()
                new_centroids[k] = X_tensor[random_idx]
        
        centroids = new_centroids
        
        # Check convergenza
        center_shift = torch.sum((centroids - previous_centroids) ** 2)
        if center_shift < tol:
            break
            
    return centroids, labels

# -------------------------------------

class MetaHNSW:
    def __init__(self, dimension: int, max_clusters: int = 500, ef_construction: int = 200, M: int = 16):
        self.dimension = dimension
        self.max_clusters = max_clusters
        self.ef_construction = ef_construction
        self.M = M
        self.hnsw_index = None
    
    def build(self, clusters: ListOfVectorsWithId):
        logger.info("Building MetaHNSW index...")
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.hnsw_index.init_index(
            max_elements=max(len(clusters), self.max_clusters),
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.hnsw_index.set_ef(50) 
        
        indices = [cluster[0] for cluster in clusters]
        centroids = [cluster[1] for cluster in clusters]
        
        self.hnsw_index.add_items(np.array(centroids, dtype=np.float32), np.array(indices))
        logger.info(f"MetaHNSW index built with {len(indices)} items.")

    def find_nearest_nodes(self, query_vector: Vector, k: int = 1) -> List[Tuple[str, float]]:
        if self.hnsw_index is None:
            logger.error("Error: hnsw still unbuilt")
            raise Exception("Error: hnsw still unbuilt")
        query = np.array(query_vector, dtype=np.float32)
        cluster_ids, distances = self.hnsw_index.knn_query(query, k=k)
        return sorted([(cluster_id, dist) for cluster_id, dist in zip(cluster_ids[0], distances[0])], key=lambda x: x[1])

    def search_batch(self, query_vectors: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.hnsw_index is None:
            raise Exception("Error: hnsw still unbuilt")
        labels, distances = self.hnsw_index.knn_query(query_vectors, k=k)
        return labels, distances

    def to_serializable_dict(self):
        index_base64 = None
        if self.hnsw_index:
            index_binary = pickle.dumps(self.hnsw_index)
            index_base64 = base64.b64encode(index_binary).decode('utf-8')
        return {
            'dimension': self.dimension, 'max_clusters': self.max_clusters,
            'ef_construction': self.ef_construction, 'M': self.M, 'index_data': index_base64
        }

    @classmethod
    def from_serializable_dict(cls, data: dict):
        new_obj = cls(dimension=data['dimension'], max_clusters=data['max_clusters'], ef_construction=data['ef_construction'], M=data['M'])
        index_base64 = data.get('index_data')
        if index_base64:
            index_binary = base64.b64decode(index_base64)
            new_obj.hnsw_index = pickle.loads(index_binary)
        return new_obj


def find_k_and_run_kmeans(X, max_k=30, random_state=42):
    """
    Versione ottimizzata con PyTorch per supportare GPU NVIDIA, MPS (Mac), e CPU.
    """
    k_range = range(2, max_k + 1)
    if X.shape[0] <= max_k:
        logger.warning(f"Samples ({X.shape[0]}) <= max_k ({max_k}). Adjusting.")
        k_range = range(2, X.shape[0])

    # 1. Rilevamento Device (NVIDIA vs Mac MPS vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Running Clustering on DEVICE: {device}")

    # 2. Spostamento dati su GPU una volta sola
    X_tensor = torch.from_numpy(X).float().to(device)
    n = X.shape[0]

    max_score = -2
    best_centroids = None
    best_k = -1
    best_labels = None

    logger.info("Starting KMeans optimization (PyTorch Accelerated)...")
    
    # Per il calcolo della Silhouette (che sklearn fa su CPU), usiamo un subset se necessario
    if n > SILHOUETTE_SUBSAMPLE_SIZE:
        idx = np.random.choice(n, SILHOUETTE_SUBSAMPLE_SIZE, replace=False)
        X_cpu_sample = X[idx] # Numpy array su CPU
    else:
        X_cpu_sample = X # Numpy array su CPU

    for k in k_range:
        # Esegue training su GPU
        centroids_gpu, labels_gpu = kmeans_torch(X_tensor, num_clusters=k, niter=KMEANS_N_ITER, device=device)
        
        # Sposta SOLO le etichette necessarie su CPU per calcolare lo score
        # Nota: silhouette_score richiede CPU numpy array.
        # Se abbiamo fatto subsampling, dobbiamo prendere le label corrispondenti agli indici
        if n > SILHOUETTE_SUBSAMPLE_SIZE:
            # Dobbiamo ricalcolare le label per il subset o prenderle dal tensore completo
            # Prendiamo dal tensore completo e tagliamo
            labels_cpu_sample = labels_gpu[idx].cpu().numpy()
        else:
            labels_cpu_sample = labels_gpu.cpu().numpy()

        try:
            score = silhouette_score(X_cpu_sample, labels_cpu_sample)
        except Exception:
            score = -1
            
        logger.debug(f"Silhouette score for k={k}: {score}")

        if score > max_score:
            max_score = score
            # Teniamo i centroidi su CPU per salvarli/ritornarli alla fine
            best_centroids = centroids_gpu.cpu().numpy()
            best_labels = labels_gpu.cpu().numpy()
            best_k = k

    # Pulizia memoria GPU
    if device.type != 'cpu':
        del X_tensor
        torch.cuda.empty_cache() if device.type == 'cuda' else torch.mps.empty_cache()

    if best_centroids is not None:
        try:
            with open('centroids.json', 'w') as f:
                json.dump(best_centroids.tolist(), f, indent=2)
        except:
            pass
    else:
        logger.warning("Clustering failed. Fallback.")
        return 1, np.mean(X, axis=0, keepdims=True), np.zeros(n)

    logger.info(f"KMeans finished. Best k={best_k}, Max Score={max_score}")
    return best_k, best_centroids, best_labels

def find_assignment(clusters: List[Tuple[str, int, List[float]]], all_nodes, replication_factor, beam_width) -> Dict[str, ListOfVectorsWithId]:
    """
    Finds a high-quality assignment using Beam Search.
    """
    logger.info(f"--- Running Beam Search (Beam Width: {beam_width}) ---")
    start_time = time.time()
    
    # Pre-calculate combinations to avoid re-generating
    all_combos = list(itertools.combinations(all_nodes, replication_factor))
    
    # State: (score, partial_assignment, node_loads)
    # Using tuple for node_loads in queue might be tricky, keep dict but careful with copy
    beam = [(0.0, [], {id: 0.0 for id in all_nodes})] 
    
    clusters_sorted = sorted(clusters, key=lambda x: x[1], reverse=True)
    
    for idx, (_, cluster_load, _) in enumerate(clusters_sorted):
        potential_states = []
        
        for score, current_assignment, current_node_loads in beam:
            for combo in all_combos:
                # Fast copy via dictionary comprehension usually faster than deepcopy for simple structs
                new_node_loads = {k: v for k, v in current_node_loads.items()}
                
                # Update loads
                for node_idx in combo:
                    new_node_loads[node_idx] += cluster_load
                
                # Calculate new score (load balancing metric: sum of squares)
                # Optimization: only recompute modified nodes? 
                # For now, full sum is safe and relatively fast for small N nodes.
                new_score = sum(val*val for val in new_node_loads.values())
                
                # Append assignment. 
                # Warning: creating new lists in loop is costly. 
                # But necessary for beam search history.
                new_assignment = current_assignment + [combo]
                
                potential_states.append((new_score, new_assignment, new_node_loads))

        # Keep top-k best states
        beam = heapq.nsmallest(beam_width, potential_states, key=lambda x: x[0]) 
    
    _, best_assignment, _ = beam[0]
    end_time = time.time()
    logger.info(f"Beam Search completed in {end_time - start_time:.4f} seconds.")
    
    assignment = {node_id: [] for node_id in all_nodes}

    for index, nodes_tuple in enumerate(best_assignment):
        for node in nodes_tuple:
            assignment[node].append((clusters_sorted[index][0], clusters_sorted[index][2]))

    return assignment

def get_clusters(vectors: ListOfVectorsComplete) -> Dict[VectorId, Tuple[Vector, ListOfVectorsComplete]]:
    # Extract only vectors for clustering
    data_matrix = np.array([v[0] for v in vectors])
    
    _, best_centroids, labels = find_k_and_run_kmeans(data_matrix)
    
    best_c = best_centroids.tolist()
    result = {}
    
    # Grouping by label
    # Optimization: Use numpy for indexing instead of list comprehension loop
    for i in range(len(best_c)):
        indices = np.where(labels == i)[0]
        # Retrieve original objects
        cluster_vectors = [vectors[j] for j in indices]
        result[i] = (best_c[i], cluster_vectors)
        
    return result

def get_node_assignment(clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], peers, replication_factor) -> Dict[str, ListOfVectorsWithId]:
    request = [(id, len(v_ids), centroid) for id, (centroid, v_ids) in clusters.items()]
    assignment = find_assignment(request, [peer.id for peer in peers], replication_factor, 50)
    return assignment

def build_meta_hnsw(clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], dimension):
    clusters_adjusted = [(cluster_id, cluster_centroid) for cluster_id, (cluster_centroid, _) in clusters.items()]
    hnsw = MetaHNSW(dimension)
    hnsw.build(clusters_adjusted)
    return hnsw

def find(hnsw: MetaHNSW, v: Vector, k: int) -> List[VectorId]:
    """Legacy single wrapper."""
    return [cluster_id for cluster_id, _ in hnsw.find_nearest_nodes(v, k)]

def find_batch(hnsw: MetaHNSW, vectors: Union[List[Vector], np.ndarray], k: int) -> List[List[VectorId]]:
    """
    Finds nearest clusters for a BATCH of vectors. 
    This is 100x faster than calling find() in a loop.
    
    Args:
        hnsw: The index object
        vectors: List of vectors OR numpy array (N, Dim)
        k: number of neighbors
    
    Returns:
        List of lists (one list of IDs per query vector)
    """
    # 1. Convert to numpy float32 only once
    if isinstance(vectors, list):
        data = np.array(vectors, dtype=np.float32)
    else:
        data = vectors.astype(np.float32)
        
    # 2. Call batch search
    # labels shape: (N, k)
    labels, _ = hnsw.search_batch(data, k)
    
    # 3. Convert back to python list of lists
    return labels.tolist()