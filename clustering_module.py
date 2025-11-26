import numpy as np
import json
import faiss
from sklearn.metrics import silhouette_score
import time
import itertools
from typing import List, Tuple, Dict, Union
import heapq
from compound_types import *
import hnswlib
import pickle
import base64
import logging

logger = logging.getLogger(__name__)

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
        
        # Batch insert is natively supported and fast
        self.hnsw_index.add_items(np.array(centroids, dtype=np.float32), np.array(indices))
        logger.info(f"MetaHNSW index built with {len(indices)} items.")

    def find_nearest_nodes(self, query_vector: Vector, k: int = 1) -> List[Tuple[str, float]]:
        """
        Legacy method for single vector search.
        """
        if self.hnsw_index is None:
            logger.error("Error: hnsw still unbuilt")
            raise Exception("Error: hnsw still unbuilt")
        
        # Hnswlib expects a list of vectors even for a single query if we want consistent output
        # But for single items, knn_query handles 1D arrays fine
        query = np.array(query_vector, dtype=np.float32)
        cluster_ids, distances = self.hnsw_index.knn_query(query, k=k)
        
        # Output handling for single vector
        return sorted([(cluster_id, dist) for cluster_id, dist in zip(cluster_ids[0], distances[0])], key=lambda x: x[1])

    def search_batch(self, query_vectors: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized batch search.
        Args:
            query_vectors: np.ndarray of shape (num_vectors, dimension)
            k: int
        Returns:
            labels: (num_vectors, k) matrix of IDs
            distances: (num_vectors, k) matrix of distances
        """
        if self.hnsw_index is None:
            raise Exception("Error: hnsw still unbuilt")
            
        # knn_query is heavily optimized for batches in C++
        labels, distances = self.hnsw_index.knn_query(query_vectors, k=k)
        return labels, distances

    def to_serializable_dict(self):
        """Serializes the entire object, including the binary index."""
        index_base64 = None
        if self.hnsw_index:
            index_binary = pickle.dumps(self.hnsw_index)
            index_base64 = base64.b64encode(index_binary).decode('utf-8')

        return {
            'dimension': self.dimension,
            'max_clusters': self.max_clusters,
            'ef_construction': self.ef_construction,
            'M': self.M,
            'index_data': index_base64
        }

    @classmethod
    def from_serializable_dict(cls, data: dict):
        """Reconstructs the object from a serialized dictionary."""
        new_obj = cls(
            dimension=data['dimension'],
            max_clusters=data['max_clusters'],
            ef_construction=data['ef_construction'],
            M=data['M']
        )
        
        index_base64 = data.get('index_data')
        if index_base64:
            index_binary = base64.b64decode(index_base64)
            new_obj.hnsw_index = pickle.loads(index_binary)
            
        return new_obj

def find_k_and_run_kmeans(X, max_k=10, random_state=42): # using silhouette score
    k_range = range(8, max_k + 1)
    
    if X.shape[0] <= max_k:
        logger.warning(f"Number of samples ({X.shape[0]}) is <= max_k ({max_k}). Adjusting range.")
        k_range = range(2, X.shape[0])

    X_faiss = X.astype(np.float32) # Convert to float32 for FAISS
    n, d = X_faiss.shape
    max_score = -2
    best_centroids = None
    best_k = -1
    best_labels = None

    logger.info("Starting KMeans optimization (Silhouette Score)...")
    for k in k_range:
        kmeans = faiss.Kmeans(
            d=d,
            k=k,
            niter=300,
            nredo=1,
            verbose=False,
            seed=random_state,
            gpu=False
        )
        kmeans.train(X_faiss)
        
        _, labels = kmeans.index.search(X_faiss, 1)
        labels = labels.flatten()
        
        # Silhouette score can be heavy for large N, but for 10k it's acceptable-ish.
        # Can be optimized by sampling if needed.
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1

        if score > max_score:
            logger.debug(f"New best silhouette score: {score} for k={k}")
            max_score = score
            best_centroids = kmeans.centroids
            best_labels = labels
            best_k = k

    if best_centroids is not None:
        centroids_list = best_centroids.tolist()
        try:
            with open('centroids.json', 'w') as f:
                json.dump(centroids_list, f, indent=2)
        except:
            pass
    else:
        # Fallback if clustering failed (e.g., too few points)
        logger.warning("Clustering failed to find valid K. Returning inputs as clusters.")
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