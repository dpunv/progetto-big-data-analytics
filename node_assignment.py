import json
import os
import sys
import numpy as np
import faiss
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import random
import math
import heapq
import time

# --- K-Means Analysis Functions ---

def load_vectors(filename, max_vectors):
    """
    Loads vectors from a JSON file.
    Expects a JSON file containing a list of objects,
    each with an 'embedding' key.
    e.g., [{'embedding': [1, 2]}, {'embedding': [3, 4]}, ...]
    """
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        print("Please create this file or change the EMBEDDINGS_FILE variable.")
        sys.exit(1)
        
    try:
        with open(filename, 'r') as f:
            # Assumes the structure is a list of objects with 'embedding' keys
            vectors = [item['embedding'] for item in json.load(f)]
            
        if max_vectors > 0:
             vectors = vectors[:max_vectors]
        
        X = np.array(vectors)
        
        if len(X.shape) != 2 or X.shape[1] == 0:
            print(f"Error: Data in '{filename}' is not a valid 2D array.")
            sys.exit(1)
            
        print(f"Successfully loaded {X.shape[0]} vectors with {X.shape[1]} dimensions.")
        return X
        
    except Exception as e:
        print(f"Error loading or processing '{filename}': {e}")
        sys.exit(1)

def plot_elbow_method(X, max_k, random_state):
    """
    Calculates inertia for k=1 to max_k and plots the elbow curve.
    Saves the plot as 'elbow_plot.png'.
    """
    print(f"\n--- Calculating Elbow Method (k=1 to {max_k}) ---")
    inertia_values = []
    k_range = range(1, max_k + 1)
    
    # Convert to float32 for FAISS
    X_faiss = X.astype(np.float32)
    n, d = X_faiss.shape
    
    for k in k_range:
        if k % (max_k // 10 if max_k > 10 else 1) == 0:
            print(f"  Calculating for k={k}...")
        
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
        
        # Calculate inertia manually
        _, labels = kmeans.index.search(X_faiss, 1)
        labels = labels.flatten()
        
        inertia = 0.0
        for i in range(n):
            diff = X_faiss[i] - kmeans.centroids[labels[i]]
            inertia += np.dot(diff, diff)
        
        inertia_values.append(inertia)
        
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertia_values, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k (FAISS)')
    plt.xticks(np.arange(1, max_k + 1, max(1, max_k // 20)))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('elbow_plot.png')
    print("Elbow method plot saved to 'elbow_plot.png'")
    print("--> Look for the 'elbow' (point of sharpest bend) in this plot.")
    plt.close()

def plot_silhouette_scores(X, max_k, random_state):
    """
    Calculates silhouette scores for k=2 to max_k and plots them.
    Saves the plot as 'silhouette_plot.png'.
    Returns the k with the highest score.
    """
    print(f"\n--- Calculating Silhouette Scores (k=2 to {max_k}) ---")
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    if X.shape[0] <= max_k:
        print(f"Warning: Number of samples ({X.shape[0]}) is <= max_k ({max_k}).")
        print(f"Adjusting max_k to {X.shape[0] - 1}.")
        k_range = range(2, X.shape[0])

    # Convert to float32 for FAISS
    X_faiss = X.astype(np.float32)
    n, d = X_faiss.shape

    for k in k_range:
        if k % (max_k // 10 if max_k > 10 else 1) == 0:
            print(f"  Calculating for k={k}...")
        
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
        
        # Get labels
        _, labels = kmeans.index.search(X_faiss, 1)
        labels = labels.flatten()
        
        # Calculate silhouette score
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Score for Optimal k (FAISS)')
    plt.xticks(np.arange(2, max_k + 1, max(1, max_k // 20)))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('silhouette_plot.png')
    print("Silhouette score plot saved to 'silhouette_plot.png'")
    
    if not silhouette_scores:
        print("Error: Could not calculate any silhouette scores.")
        return None
        
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"--> The highest silhouette score is {max(silhouette_scores):.4f} at k={best_k}.")
    return best_k

def run_final_clustering(X, k, random_state):
    """
    Runs K-Means with the optimal k, saves centroids, and returns
    the count of vectors associated with each cluster.
    """
    print(f"\n--- Running Final Clustering with k={k} (FAISS) ---")
    
    # Convert to float32 for FAISS
    X_faiss = X.astype(np.float32)
    n, d = X_faiss.shape
    
    print(f"Training FAISS K-means with {n} vectors, {d} dimensions, {k} clusters...")
    start_time = time.time()
    
    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=300,
        nredo=1,
        verbose=True,
        seed=random_state,
        gpu=False
    )
    kmeans.train(X_faiss)
    
    training_time = time.time() - start_time
    print(f"FAISS K-means training completed in {training_time:.2f}s")
    
    # Get centroids and labels
    centroids = kmeans.centroids
    _, labels = kmeans.index.search(X_faiss, 1)
    labels = labels.flatten()
    
    # Count vectors per cluster
    label_counts = Counter(labels)
    
    print("\n[Final Centroids]")
    # Save centroids to a clean JSON file
    centroids_list = centroids.tolist()
    with open('centroids.json', 'w') as f:
        json.dump(centroids_list, f, indent=2)
    print(f"Saved {len(centroids_list)} centroids to 'centroids.json'")

    print("\n[Vector Count per Cluster]")
    for cluster_label, count in sorted(label_counts.items()):
        print(f"  Cluster {cluster_label:2}: {count} vectors")
        
    return label_counts, centroids_list

# --- Load Balancing Functions ---

def calculate_assignment_score(assignment, clusters, num_nodes, wanted_load):
    """
    Calculates the 'cost' (Sum of Squared Errors) of a full assignment.
    A lower score is better.
    """
    node_loads = [0.0] * num_nodes
    
    for cluster_load, combo in zip(clusters, assignment):
        for node_index in combo:
            node_loads[node_index] += cluster_load
            
    sse = sum((load - wanted_load)**2 for load in node_loads)
    
    mean_load = sum(node_loads) / num_nodes
    variance = sum((load - mean_load)**2 for load in node_loads) / num_nodes
    std_dev = math.sqrt(variance)

    return sse, std_dev, node_loads

def run_beam_search(clusters_sorted, all_combos, num_nodes, beam_width):
    """
    Finds a high-quality assignment using Beam Search.
    """
    print(f"\n--- Running Beam Search (Beam Width: {beam_width}) ---")
    start_time = time.time()
    
    # State: (score, partial_assignment, node_loads)
    beam = [(0.0, [], [0.0] * num_nodes)]
    
    for i, cluster_load in enumerate(clusters_sorted):
        potential_states = []
        
        for current_sse, current_assignment, current_node_loads in beam:
            for combo in all_combos:
                new_node_loads = list(current_node_loads)
                
                for node_idx in combo:
                    new_node_loads[node_idx] += cluster_load
                
                new_assignment = current_assignment + [combo]
                
                partial_score = sum(load**2 for load in new_node_loads)
                
                potential_states.append(
                    (partial_score, new_assignment, new_node_loads)
                )

        # Prune: Keep only the B best new states
        beam = heapq.nsmallest(beam_width, potential_states, key=lambda x: x[0])
        
    best_score, best_assignment, best_node_loads = beam[0]
    end_time = time.time()
    print(f"Beam Search completed in {end_time - start_time:.4f} seconds.")
    
    return best_assignment

def get_neighbor_assignment(current_assignment, all_combos):
    """
    Creates a "neighbor" solution by making one random change.
    """
    neighbor = list(current_assignment)
    cluster_to_change = random.randrange(len(neighbor))
    
    new_combo = random.choice(all_combos)
    while new_combo == neighbor[cluster_to_change]:
        new_combo = random.choice(all_combos)
        
    neighbor[cluster_to_change] = new_combo
    return neighbor

def run_simulated_annealing(initial_assignment, clusters, all_combos, num_nodes, 
                            wanted_load, initial_temp, cooling_rate, max_iterations):
    """
    Improves an existing solution using Simulated Annealing.
    """
    print(f"\n--- Running Simulated Annealing ({max_iterations} iterations) ---")
    start_time = time.time()

    current_assignment = initial_assignment
    current_cost, _, _ = calculate_assignment_score(
        current_assignment, clusters, num_nodes, wanted_load
    )
    
    best_assignment = current_assignment
    best_cost = current_cost
    temp = initial_temp

    for i in range(max_iterations):
        neighbor = get_neighbor_assignment(current_assignment, all_combos)
        neighbor_cost, _, _ = calculate_assignment_score(
            neighbor, clusters, num_nodes, wanted_load
        )
        
        cost_diff = neighbor_cost - current_cost

        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
            current_assignment = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_assignment = current_assignment
                best_cost = current_cost
        
        temp *= cooling_rate
        
        if (i + 1) % (max_iterations // 10) == 0:
            print(f"  Iter {i+1:6}: Temp={temp:8.2f}, Current SSE={current_cost:,.2f}, Best SSE={best_cost:,.2f}")

    end_time = time.time()
    print(f"Simulated Annealing completed in {end_time - start_time:.4f} seconds.")
    return best_assignment

def print_final_loads(node_loads, wanted_load):
    """Helper function to print the load balancing results."""
    print("  Node Loads:")
    max_load = max(node_loads)
    min_load = min(node_loads)
    
    for i, load in enumerate(node_loads):
        dev = load - wanted_load
        dev_percent = (dev / wanted_load) * 100
        print(f"    Node {i:2}: {load:,.2f} (Dev: {dev:+.2f} / {dev_percent:+.2f}%)")
    
    print(f"\n  Min Load: {min_load:,.2f}")
    print(f"  Max Load: {max_load:,.2f}")
    print(f"  Range (Delta): {max_load - min_load:,.2f}")


# --- NEW: API Function for External Use ---

def compute_node_assignments(embeddings_file, num_nodes, max_k_to_test=30, 
                             random_state=42, max_vectors=10000,
                             rep_factor=None, beam_width=5,
                             use_simulated_annealing=True):
    """
    Complete pipeline: clustering + load balancing.
    
    Args:
        embeddings_file: Path to JSON file with embeddings
        num_nodes: Number of nodes in the system
        max_k_to_test: Maximum k for cluster analysis
        random_state: Random seed for reproducibility
        max_vectors: Maximum vectors to use for training (0 = all)
        rep_factor: Replication factor (None = auto-calculate)
        beam_width: Beam width for search algorithm
        use_simulated_annealing: Whether to refine with SA
    
    Returns:
        dict with:
            - 'centroids': List of cluster centroids
            - 'cluster_counts': Dict of cluster sizes
            - 'node_assignments': Dict mapping node_id -> list of cluster indices
            - 'assignment_details': List of (cluster_idx, node_indices) tuples
            - 'stats': Performance statistics
    """
    print("="*60)
    print("COMPUTING OPTIMAL NODE ASSIGNMENTS")
    print("="*60)
    
    # Step 1: Load and cluster
    X = load_vectors(embeddings_file, max_vectors)
    
    if X.shape[0] <= max_k_to_test:
        max_k_to_test = X.shape[0] - 1
        if max_k_to_test < 2:
            raise ValueError("Not enough data points to cluster")
        print(f"MAX_K_TO_TEST adjusted to {max_k_to_test}")

    plot_elbow_method(X, max_k_to_test, random_state)
    recommended_k = plot_silhouette_scores(X, max_k_to_test, random_state)
    
    if recommended_k is None:
        raise ValueError("Could not determine optimal k")
        
    print(f"\n[Recommendation] Using k = {recommended_k}")
    
    cluster_counts, centroids_list = run_final_clustering(X, recommended_k, random_state)
    clusters = list(cluster_counts.values())
    
    # Step 2: Load balance
    print("\n" + "="*60)
    print("LOAD BALANCING CLUSTERS ACROSS NODES")
    print("="*60)
    
    if rep_factor is None:
        rep_factor = max(3, int(num_nodes / len(clusters)))
    
    clusters_sorted = sorted(clusters, reverse=True)
    total_load = sum(clusters)
    wanted_load = rep_factor * total_load / num_nodes
    
    node_indices = list(range(num_nodes))
    all_combos = list(itertools.combinations(node_indices, rep_factor))

    print(f"Balancing {len(clusters)} clusters (Total Load: {total_load})")
    print(f"Across {num_nodes} nodes with Replication Factor {rep_factor}")
    print(f"Target load per node: {wanted_load:,.2f}")

    # Beam search
    bs_assignment = run_beam_search(clusters_sorted, all_combos, num_nodes, beam_width)
    bs_sse, bs_std_dev, bs_loads = calculate_assignment_score(
        bs_assignment, clusters_sorted, num_nodes, wanted_load
    )
    
    print("\n--- Beam Search Result ---")
    print(f"  Score (SSE): {bs_sse:,.2f}")
    print(f"  Std Deviation: {bs_std_dev:,.2f}")
    print_final_loads(bs_loads, wanted_load)

    # Optional: Simulated annealing
    final_assignment = bs_assignment
    final_sse = bs_sse
    
    if use_simulated_annealing:
        sa_assignment = run_simulated_annealing(
            bs_assignment, clusters_sorted, all_combos, num_nodes,
            wanted_load, 10000.0, 0.999, 20000
        )
        
        sa_sse, sa_std_dev, sa_loads = calculate_assignment_score(
            sa_assignment, clusters_sorted, num_nodes, wanted_load
        )
        
        print("\n--- Simulated Annealing Result ---")
        print(f"  Score (SSE): {sa_sse:,.2f}")
        print(f"  Std Deviation: {sa_std_dev:,.2f}")
        print_final_loads(sa_loads, wanted_load)
        
        if sa_sse < bs_sse:
            final_assignment = sa_assignment
            final_sse = sa_sse
            print(f"\nUsing SA solution (improvement: {bs_sse - sa_sse:,.2f})")
        else:
            print("\nUsing Beam Search solution (SA did not improve)")

    # Step 3: Format results
    # Map sorted cluster indices back to original cluster indices
    cluster_sort_map = sorted(range(len(clusters)), 
                              key=lambda i: clusters[i], 
                              reverse=True)
    
    # Create node assignments
    node_assignments = {i: [] for i in range(num_nodes)}
    assignment_details = []
    
    for sorted_idx, node_combo in enumerate(final_assignment):
        original_cluster_idx = cluster_sort_map[sorted_idx]
        assignment_details.append((int(original_cluster_idx), [int(n) for n in node_combo]))
        
        for node_idx in node_combo:
            node_assignments[node_idx].append(int(original_cluster_idx))
    
    # Convert all numpy types to native Python types for JSON serialization
    cluster_counts_json = {int(k): int(v) for k, v in cluster_counts.items()}
    node_assignments_json = {int(k): [int(x) for x in v] for k, v in node_assignments.items()}
    
    result = {
        'centroids': centroids_list,
        'cluster_counts': cluster_counts_json,
        'node_assignments': node_assignments_json,
        'assignment_details': assignment_details,
        'stats': {
            'num_clusters': int(len(clusters)),
            'num_nodes': int(num_nodes),
            'replication_factor': int(rep_factor),
            'final_sse': float(final_sse),
            'target_load_per_node': float(wanted_load)
        }
    }
    
    # Save to file
    with open('node_assignments.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved complete assignment to 'node_assignments.json'")
    
    return result


# --- Main Execution (Standalone Mode) ---

def main():
    """
    Main function for standalone execution.
    """
    EMBEDDINGS_FILE = 'embeddings.json'
    NUM_NODES = 20
    MAX_K_TO_TEST = 30
    RANDOM_STATE = 42
    MAX_VECTORS = 10000
    
    result = compute_node_assignments(
        embeddings_file=EMBEDDINGS_FILE,
        num_nodes=NUM_NODES,
        max_k_to_test=MAX_K_TO_TEST,
        random_state=RANDOM_STATE,
        max_vectors=MAX_VECTORS
    )
    
    print("\n" + "="*60)
    print("ASSIGNMENT SUMMARY")
    print("="*60)
    for node_idx, cluster_indices in result['node_assignments'].items():
        print(f"Node {node_idx}: {len(cluster_indices)} clusters -> {cluster_indices}")

if __name__ == "__main__":
    main()