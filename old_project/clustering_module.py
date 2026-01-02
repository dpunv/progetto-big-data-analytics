import numpy as np
import json
import faiss
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
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
import os
import sys

SILHOUETTE_SUBSAMPLE_SIZE = 2000  # Max sample size for silhouette score
KMEANS_N_ITER = 150  # Reduced n_iter for KMeans

# Setup logging for clustering_module

# Logger locale per clustering_module, sempre su INFO e logs/clustering.log
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("ClusteringModule")
logger.propagate = False
if not logger.hasHandlers():
    file_handler = logging.FileHandler("logs/clustering.log", mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

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
            
        current_count = self.hnsw_index.element_count
        if k > current_count:
            # logger.warning(f"Requested k={k} is larger than index size {current_count}. Adjusting k.")
            k = current_count
            
        if k == 0:
            return []
            
        # Ensure ef is large enough
        if self.hnsw_index.ef < k:
            self.hnsw_index.set_ef(k)

        query = np.array(query_vector, dtype=np.float32)
        cluster_ids, distances = self.hnsw_index.knn_query(query, k=k)
        return sorted([(cluster_id, dist) for cluster_id, dist in zip(cluster_ids[0], distances[0])], key=lambda x: x[1])

    def search_batch(self, query_vectors: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.hnsw_index is None:
            raise Exception("Error: hnsw still unbuilt")
            
        current_count = self.hnsw_index.element_count
        if k > current_count:
            k = current_count
            
        if k == 0:
            return np.array([]), np.array([])

        if self.hnsw_index.ef < k:
            self.hnsw_index.set_ef(k)

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
    logger.info("="*60)
    logger.info("STARTING CLUSTERING PROCESS")
    logger.info(f"Input data shape: {X.shape}")
    logger.info(f"Max k to evaluate: {max_k}")
    
    k_range = range(2, max_k + 1)
    if X.shape[0] <= max_k:
        logger.warning(f"Samples ({X.shape[0]}) <= max_k ({max_k}). Adjusting k_range.")
        k_range = range(2, X.shape[0])

    # 1. Rilevamento Device (NVIDIA vs Mac MPS vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Log detailed device information
    if device.type == "cuda":
        logger.info(f"✓ GPU ACCELERATION ENABLED: CUDA")
        logger.info(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        logger.info(f"GPU ACCELERATION ENABLED: Apple MPS")
    else:
        logger.info(f"Running on CPU (no GPU acceleration)")
    
    device = torch.device("cpu") # force CPU for testing
    
    logger.info(f"Device: {device}")

    # 2. Spostamento dati su GPU una volta sola
    if device.type != 'cpu':
        try:
            X_tensor = torch.from_numpy(X).float().to(device)
            logger.info(f"Successfully moved data to {device}")
        except Exception as e:
            logger.warning(f"Failed to move data to {device}: {e}. Falling back to CPU.")
            device = torch.device("cpu")
            X_tensor = None
    else:
        X_tensor = None
    
    n = X.shape[0]

    max_score = -2
    best_centroids = None
    best_k = -1
    best_labels = None

    logger.info("Starting KMeans optimization (PyTorch Accelerated)...")
    logger.info(f"K-range to evaluate: {list(k_range)[0]} to {list(k_range)[-1]}")
    logger.info(f"KMeans iterations per k: {KMEANS_N_ITER}")
    
    # Per il calcolo della Silhouette (che sklearn fa su CPU), usiamo un subset se necessario
    if n > SILHOUETTE_SUBSAMPLE_SIZE:
        idx = np.random.choice(n, SILHOUETTE_SUBSAMPLE_SIZE, replace=False)
        X_cpu_sample = X[idx] # Numpy array su CPU
        logger.info(f"Using subsampling for silhouette score: {SILHOUETTE_SUBSAMPLE_SIZE}/{n} samples")
    else:
        X_cpu_sample = X # Numpy array su CPU
        logger.info(f"Using all {n} samples for silhouette score")

    for k in k_range:
        if device.type == 'cpu':
            # Use sklearn KMeans (CPU optimized)
            kmeans = KMeans(n_clusters=k, max_iter=KMEANS_N_ITER, random_state=42, n_init=1)
            kmeans.fit(X)
            
            if n > SILHOUETTE_SUBSAMPLE_SIZE:
                labels_cpu_sample = kmeans.labels_[idx]
            else:
                labels_cpu_sample = kmeans.labels_
                
            current_centroids = kmeans.cluster_centers_
            current_labels = kmeans.labels_
            
        else:
            # Esegue training su GPU
            centroids_gpu, labels_gpu = kmeans_torch(X_tensor, num_clusters=k, niter=KMEANS_N_ITER, device=device)
            # Sposta SOLO le etichette necessarie su CPU per calcolare lo score
            if n > SILHOUETTE_SUBSAMPLE_SIZE:
                labels_cpu_sample = labels_gpu[idx].cpu().numpy()
            else:
                labels_cpu_sample = labels_gpu.cpu().numpy()
                
            current_centroids = centroids_gpu.cpu().numpy()
            current_labels = labels_gpu.cpu().numpy()

        try:
            score = silhouette_score(X_cpu_sample, labels_cpu_sample)
        except Exception as e:
            score = -1
            logger.warning(f"Failed to compute silhouette score for k={k}: {e}")
            
        logger.info(f"  k={k:2d} | Silhouette Score: {score:+.4f}")

        if score > max_score:
            max_score = score
            best_centroids = current_centroids
            best_labels = current_labels
            best_k = k

    # Pulizia memoria GPU
    if device.type != 'cpu':
        del X_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared (CUDA)")
        else:
            torch.mps.empty_cache()
            logger.info("GPU memory cache cleared (MPS)")

    if best_centroids is not None:
        try:
            with open('centroids.json', 'w') as f:
                json.dump(best_centroids.tolist(), f, indent=2)
            logger.info("Centroids saved to centroids.json")
        except Exception as e:
            logger.warning(f"Failed to save centroids: {e}")
    else:
        logger.warning("Clustering failed. Using fallback: single cluster with mean centroid.")
        return 1, np.mean(X, axis=0, keepdims=True), np.zeros(n)

    logger.info("="*60)
    logger.info("CLUSTERING COMPLETED SUCCESSFULLY")
    logger.info(f"  Best k: {best_k}")
    logger.info(f"  Best Silhouette Score: {max_score:.4f}")
    logger.info(f"  Number of centroids: {len(best_centroids)}")
    
    # Log cluster sizes
    unique, counts = np.unique(best_labels, return_counts=True)
    logger.info(f"  Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        logger.info(f"    Cluster {cluster_id}: {count} vectors")
    logger.info("="*60)
    
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

def get_clusters(vectors: ListOfVectorsComplete, max_clusters=30) -> Dict[VectorId, Tuple[Vector, ListOfVectorsComplete]]:
    logger.info(f"get_clusters() called with {len(vectors)} vectors")
    
    # Extract only vectors for clustering
    data_matrix = np.array([v[0] for v in vectors], dtype=np.float32)
    logger.info(f"Extracted data matrix of shape: {data_matrix.shape}")
    
    _, best_centroids, labels = find_k_and_run_kmeans(data_matrix, max_clusters)
    
    best_c = best_centroids.tolist()
    result = {}
    
    # Grouping by label
    # Optimization: Use numpy for indexing instead of list comprehension loop
    logger.info("Grouping vectors by cluster labels...")
    for i in range(len(best_c)):
        indices = np.where(labels == i)[0]
        # Retrieve original objects and update cluster ID
        cluster_vectors = []
        for j in indices:
            # vectors[j] is (vector_content, vector_id, vector_payload, old_cluster_id)
            # We need to update old_cluster_id to str(i)
            v = vectors[j]
            new_vector_tuple = (v[0], v[1], v[2], str(i))
            cluster_vectors.append(new_vector_tuple)
            
        result[i] = (best_c[i], cluster_vectors)
    
    logger.info(f"Created {len(result)} clusters from vectors")
    return result

def get_node_assignment(clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], peers, replication_factor) -> Dict[str, ListOfVectorsWithId]:
    logger.info(f"Computing node assignment for {len(clusters)} clusters across {len(peers)} peers")
    logger.info(f"Replication factor: {replication_factor}")
    
    request = [(id, len(v_ids), centroid) for id, (centroid, v_ids) in clusters.items()]
    assignment = find_assignment(request, [peer.id for peer in peers], replication_factor, 50)
    
    # Log assignment summary
    for node_id, assigned_clusters in assignment.items():
        logger.info(f"  {node_id}: {len(assigned_clusters)} clusters assigned")
    
    return assignment

def build_meta_hnsw(clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], dimension):
    logger.info(f"Building MetaHNSW index for {len(clusters)} clusters (dimension={dimension})")
    
    clusters_adjusted = [(cluster_id, cluster_centroid) for cluster_id, (cluster_centroid, _) in clusters.items()]
    hnsw = MetaHNSW(dimension)
    hnsw.build(clusters_adjusted)
    
    logger.info("MetaHNSW index built successfully")
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
        if not vectors:
            return []
        data = np.array(vectors, dtype=np.float32)
    else:
        if vectors.size == 0:
            return []
        data = vectors.astype(np.float32)
        
    # 2. Call batch search
    # labels shape: (N, k)
    labels, _ = hnsw.search_batch(data, k)
    
    # 3. Convert back to python list of lists
    return labels.tolist()