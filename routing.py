# routing.py
import threading
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

class RoutingTable:
    """Gestisce mappatura cluster → nodi con ordinamento semantico."""
    
    def __init__(self, node_names: list[str]):
        self._node_names = node_names
        self._routing_map = {}
        self._lock = threading.Lock()
        self._cluster_order = []
    
    def get_nodes(self, cluster_id: str) -> list[str]:
        """Restituisce nodi contenenti cluster_id."""
        with self._lock:
            if cluster_id in self._routing_map:
                return self._routing_map[cluster_id]
            raise KeyError(f"Cluster {cluster_id} not found")
    
    def assign_clusters_by_semantic_similarity(self, n_clusters: int, centroids: np.ndarray):
        """
        Assegna cluster ai nodi raggruppando per similarità semantica.
        
        STRATEGIA:
        1. Calcola matrice distanze tra centroidi
        2. Costruisce Minimum Spanning Tree (MST)
        3. Ordina cluster via DFS sul MST
        4. Divide sequenza ordinata tra nodi
        
        Risultato: cluster simili → stesso nodo → query efficienti
        """
        with self._lock:
            n_nodes = len(self._node_names)
            
            print(f"\n{'='*70}")
            print(f"SEMANTIC CLUSTERING ASSIGNMENT")
            print(f"{'='*70}")
            print(f"Clusters: {n_clusters} | Nodes: {n_nodes}")
            print(f"Strategy: Group similar clusters on same node\n")
            
            # Step 1: Distanze
            print("Step 1: Computing distances...")
            distances_condensed = pdist(centroids, metric='euclidean')
            distance_matrix = squareform(distances_condensed)
            
            print(f"  Min: {np.min(distance_matrix[distance_matrix > 0]):.4f}")
            print(f"  Max: {np.max(distance_matrix):.4f}")
            print(f"  Mean: {np.mean(distance_matrix[distance_matrix > 0]):.4f}\n")
            
            # Step 2: MST
            print("Step 2: Building MST...")
            mst = minimum_spanning_tree(distance_matrix)
            mst_array = mst.toarray()
            
            print(f"  Edges: {np.count_nonzero(mst_array)}")
            print(f"  Weight: {mst_array.sum():.4f}\n")
            
            # Step 3: DFS ordering
            print("Step 3: Ordering clusters...")
            cluster_order = self._dfs_mst_traversal(mst_array, n_clusters)
            self._cluster_order = cluster_order
            
            print(f"  Order: {cluster_order}\n")
            
            # Verifica distanze consecutive
            print("  Consecutive distances:")
            for i in range(min(5, len(cluster_order) - 1)):
                c1, c2 = cluster_order[i], cluster_order[i+1]
                dist = distance_matrix[c1, c2]
                print(f"    {c1} → {c2}: {dist:.4f}")
            print()
            
            # Step 4: Partizionamento
            print("Step 4: Partitioning...")
            
            clusters_per_node = n_clusters // n_nodes
            remainder = n_clusters % n_nodes
            
            print(f"  Base: {clusters_per_node} clusters/node")
            if remainder > 0:
                print(f"  Extra: {remainder} cluster(s)\n")
            
            current_idx = 0
            
            for node_idx, node_name in enumerate(self._node_names):
                n_clusters_for_node = clusters_per_node + (1 if node_idx < remainder else 0)
                node_clusters = cluster_order[current_idx : current_idx + n_clusters_for_node]
                
                for cluster_id in node_clusters:
                    self._routing_map[str(cluster_id)] = [node_name]
                
                # Coesione intra-nodo
                if len(node_clusters) > 1:
                    avg_dist = np.mean([
                        distance_matrix[node_clusters[i], node_clusters[j]]
                        for i in range(len(node_clusters))
                        for j in range(i+1, len(node_clusters))
                    ])
                else:
                    avg_dist = 0
                
                print(f"  {node_name}: {len(node_clusters)} clusters {node_clusters}")
                print(f"    Avg intra-distance: {avg_dist:.4f}")
                
                current_idx += n_clusters_for_node
            
            print(f"\n{'='*70}")
            print("✅ Assignment complete")
            print(f"{'='*70}\n")
    
    def _dfs_mst_traversal(self, mst_array: np.ndarray, n_clusters: int) -> list[int]:
        """DFS sul MST per ordinamento semantico."""
        visited = set()
        order = []
        
        def dfs(node):
            visited.add(node)
            order.append(node)
            
            neighbors = np.where(mst_array[node] > 0)[0].tolist()
            neighbors += np.where(mst_array[:, node] > 0)[0].tolist()
            neighbors = list(set(neighbors))
            
            neighbors.sort(key=lambda n: mst_array[node, n] + mst_array[n, node])
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(0)
        
        for cluster_id in range(n_clusters):
            if cluster_id not in visited:
                dfs(cluster_id)
        
        return order