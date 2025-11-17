import numpy as np
import json
import faiss
from sklearn.metrics import silhouette_score
#from collections import Counter
import time
import itertools
from typing import List, Tuple, Dict
import heapq
from compound_types import *
import hnswlib
import pickle
import base64

class MetaHNSW:
    def __init__(self, dimension: int, max_clusters: int = 500, ef_construction: int = 200, M: int = 16):
        self.dimension = dimension
        self.max_clusters = max_clusters
        self.ef_construction = ef_construction
        self.M = M
        self.hnsw_index = None
    
    def build(self, clusters: ListOfVectorsWithId):
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.hnsw_index.init_index(
            max_elements=max(len(clusters), self.max_clusters),
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.hnsw_index.set_ef(50) # What this is?
        
        indices = [cluster[0] for cluster in clusters]
        centroids = [cluster[1] for cluster in clusters]
        self.hnsw_index.add_items(np.array(centroids), np.array(indices))

    def find_nearest_nodes(self, query_vector: Vector, k: int = None) -> List[Tuple[str, float]]:
        if self.hnsw_index == None:
            raise "Error: hnsw still unbuilt"
        cluster_ids, distances = self.hnsw_index.knn_query(query_vector, k)
        return sorted([(cluster_id, dist) for cluster_id, dist in zip(cluster_ids[0], distances[0])], key=lambda x: x[1])

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

def find_k_and_run_kmeans(X, max_k=30, random_state=42): # using silhouette score
    k_range = range(2, max_k + 1)
    
    if X.shape[0] <= max_k:
        print(f"Warning: Number of samples ({X.shape[0]}) is <= max_k ({max_k}).")
        k_range = range(2, X.shape[0])

    X_faiss = X.astype(np.float32) # Convert to float32 for FAISS
    n, d = X_faiss.shape
    max_score = -2
    best_centroids = None
    best_k = -1
    best_labels = None

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
        
        score = silhouette_score(X, labels)
        if score > max_score:
            print(score)
            max_score = score
            best_centroids = kmeans.centroids
            best_labels = labels
            best_k = k

    centroids_list = best_centroids.tolist()
    with open('centroids.json', 'w') as f:
        json.dump(centroids_list, f, indent=2)
    #label_counts = Counter(best_labels)
    #print("\n[Vector Count per Cluster]")
    #for cluster_label, count in sorted(label_counts.items()):
    #    print(f"  Cluster {cluster_label:2}: {count} vectors")

    return best_k, best_centroids, best_labels

def find_assignment(clusters: List[Tuple[str, int, List[float]]], all_nodes, replication_factor, beam_width) -> Dict[str, ListOfVectorsWithId]: # clusters è la lista di coppie 
    """
    Finds a high-quality assignment using Beam Search.
    """
    print(f"\n--- Running Beam Search (Beam Width: {beam_width}) ---")
    start_time = time.time()
    all_combos = list(itertools.combinations(all_nodes, replication_factor))
    
    beam = [(0.0, [], {id: 0.0 for id in all_nodes})] # State: (score, partial_assignment, node_loads)
    print(f"Total cluster: {len(clusters)}")
    clusters_sorted = sorted(clusters, key=lambda x: x[1], reverse=True)
    for _, cluster_load, _ in clusters_sorted:
        potential_states = []
        
        for _, current_assignment, current_node_loads in beam:
            for combo in all_combos:
                new_node_loads = dict(current_node_loads)
                
                for node_idx in combo:
                    new_node_loads[node_idx] += cluster_load
                
                new_assignment = current_assignment + [combo]
                
                partial_score = sum(load**2 for _, load in new_node_loads.items())
                
                potential_states.append(
                    (partial_score, new_assignment, new_node_loads)
                )

        beam = heapq.nsmallest(beam_width, potential_states, key=lambda x: x[0]) # Prune: Keep only the B best new states
    print(f"Total states evaluated: {len(potential_states)}")
    _, best_assignment, _ = beam[0]
    end_time = time.time()
    print(f"Beam Search completed in {end_time - start_time:.4f} seconds.")
    
    assignment = {node_id: [] for node_id in all_nodes}

    for index, nodes_tuple in enumerate(best_assignment):
        for node in nodes_tuple:
            assignment[node].append((clusters_sorted[index][0], clusters_sorted[index][2]))

    return assignment

def get_clusters(vectors: ListOfVectorsComplete) -> Dict[VectorId, Tuple[Vector, List[VectorId]]]:
    _, best_centroids, labels = find_k_and_run_kmeans(np.array([vector for vector, _, _ in vectors]))
    best_c = best_centroids.tolist()
    result = {}
    for i in range(len(best_c)):
        result[i] = (best_c[i], [vectors[j][1] for j in range(len(labels)) if labels[j] == i])
    return result

def get_node_assignment(clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], peers, replication_factor) -> Dict[str, ListOfVectorsWithId]:
    request = [(id, len(v_ids), centroid)for id, (centroid, v_ids) in clusters.items()]
    return find_assignment(request, [peer.id for peer in peers], replication_factor, 50)


def build_meta_hnsw(clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], dimension):
    clusters_adjusted = [(cluster_id, cluster_centroid) for cluster_id, (cluster_centroid, _) in clusters.items()]
    hnsw = MetaHNSW(dimension)
    hnsw.build(clusters_adjusted)
    return hnsw

def find(hnsw: MetaHNSW, v: Vector, k: int) -> List[VectorId]:
    return [cluster_id for cluster_id, _ in hnsw.find_nearest_nodes(v, k)]