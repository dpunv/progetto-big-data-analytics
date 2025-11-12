import json
import os
import sys
import numpy as np
from typing import Tuple, List, Dict
import faiss
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import random
import math
import heapq
import time

def load_vectors(filename, max_vectors):
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        sys.exit(1)
    try:
        with open(filename, 'r') as f:
            vectors = [item['embedding'] for item in json.load(f)]
        if max_vectors > 0:
            vectors = vectors[:max_vectors]
        
        X = np.array(vectors)
        
        if len(X.shape) != 2 or X.shape[1] == 0:
            print(f"Error: Data in '{filename}' is not a valid 2D array.")
            sys.exit(1)
        return X
    except Exception as e:
        print(f"Error loading or processing '{filename}': {e}")
        sys.exit(1)

def find_k_and_run_kmeans(X, max_k, random_state): # using silhouette score
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
            max_score = score
            best_centroids = kmeans.centroids
            best_labels = labels
            best_k = k
    
    centroids_list = best_centroids.tolist()
    with open('centroids.json', 'w') as f:
        json.dump(centroids_list, f, indent=2)
    label_counts = Counter(best_labels)
    print("\n[Vector Count per Cluster]")
    for cluster_label, count in sorted(label_counts.items()):
        print(f"  Cluster {cluster_label:2}: {count} vectors")

    return best_k, best_centroids, label_counts

def find_assignment(clusters: List[Tuple[int, List[float]]], all_nodes, replication_factor, beam_width): # clusters è la lista di coppie 
    """
    Finds a high-quality assignment using Beam Search.
    """
    print(f"\n--- Running Beam Search (Beam Width: {beam_width}) ---")
    start_time = time.time()
    all_combos = list(itertools.combinations(all_nodes, replication_factor))
    
    beam = [(0.0, [], {id: 0.0 for id in all_nodes})] # State: (score, partial_assignment, node_loads)
    
    clusters_sorted = sorted(clusters, key=lambda x: x[0], reverse=True)

    for cluster_load, _ in clusters_sorted:
        potential_states = []
        
        for _, current_assignment, current_node_loads in beam:
            for combo in all_combos:
                new_node_loads = dict(current_node_loads)
                
                for node_idx in combo:
                    new_node_loads[node_idx] += cluster_load
                
                new_assignment = current_assignment + [combo]
                
                partial_score = sum(load**2 for load in new_node_loads)
                
                potential_states.append(
                    (partial_score, new_assignment, new_node_loads)
                )

        beam = heapq.nsmallest(beam_width, potential_states, key=lambda x: x[0]) # Prune: Keep only the B best new states
        
    _, best_assignment, _ = beam[0]
    end_time = time.time()
    print(f"Beam Search completed in {end_time - start_time:.4f} seconds.")
    
    assignment = {node_id: [] for node_id in all_nodes}

    for index, nodes_tuple in enumerate(best_assignment):
        for node in nodes_tuple:
            assignment[node].append(clusters_sorted[index][1])

    return assignment
