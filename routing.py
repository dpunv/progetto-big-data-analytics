# routing.py
import threading
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

class RoutingTable:
    """Gestisce mappatura cluster → nodi con ordinamento semantico."""
    
    def __init__(self, node_names: list[str]):
        self.routing_table = {node: [] for node in node_names}
        self._node_names = node_names
        self._routing_map = {}
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Gestione repliche
        # ═══════════════════════════════════════════════════════════════
        self.replicas = {}  # {cluster_id: [primary_node, replica_node1, ...]}
    
    def get_nodes(self, cluster_id: str) -> list[str]:
        """Restituisce nodi contenenti cluster_id."""
        cluster_id_str = str(cluster_id)
        for node_name, clusters in self.routing_table.items():
            if cluster_id_str in [str(c) for c in clusters]:
                return [node_name]
        raise KeyError(f"Cluster {cluster_id} not found in routing table")
    
    def assign_clusters_by_semantic_similarity(self, n_clusters: int, centroids: np.ndarray):
        """
        Assegna cluster ai nodi in modo round-robin semplice.
        """
        nodes_list = list(self.routing_table.keys())
        
        for cluster_id in range(n_clusters):
            assigned_node = nodes_list[cluster_id % len(nodes_list)]
            self.routing_table[assigned_node].append(cluster_id)
    
    def get_clusters_for_node(self, node_name: str) -> list:
        """Ritorna la lista di cluster ID assegnati a un nodo specifico."""
        return self.routing_table.get(node_name, [])
    
    def assign_cluster_to_node(self, cluster_id: str, node_name: str):
        """Assegna un cluster a un nodo specifico (utile per rebalancing)."""
        cluster_id_int = int(cluster_id)
        
        # Rimuovi da tutti i nodi
        for node in self.routing_table:
            self.routing_table[node] = [c for c in self.routing_table[node] if c != cluster_id_int]
        
        # Assegna al nuovo nodo
        if cluster_id_int not in self.routing_table[node_name]:
            self.routing_table[node_name].append(cluster_id_int)
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVI METODI: Gestione repliche
    # ═══════════════════════════════════════════════════════════════
    
    def add_replica(self, cluster_id: str, replica_node: str):
        """
        Aggiungi replica di un cluster su un nodo.
        
        Args:
            cluster_id: ID cluster da replicare
            replica_node: Nodo dove creare replica
        """
        if cluster_id not in self.replicas:
            # Prima volta: inizializza con primary
            primary = self.get_nodes(cluster_id)[0]
            self.replicas[cluster_id] = [primary]
        
        # Aggiungi replica se non già presente
        if replica_node not in self.replicas[cluster_id]:
            self.replicas[cluster_id].append(replica_node)
    
    def get_all_nodes_for_cluster(self, cluster_id: str) -> list[str]:
        """
        Restituisce TUTTI i nodi per cluster (primary + repliche).
        
        COMPLESSITÀ: O(1) - hash table lookup
        
        Args:
            cluster_id: ID cluster
            
        Returns:
            Lista nodi [primary, replica1, replica2, ...]
        """
        # Se ha repliche registrate, ritorna lista completa
        if cluster_id in self.replicas:
            return self.replicas[cluster_id]
        
        # Altrimenti fallback: solo primary
        try:
            return [self.get_nodes(cluster_id)[0]]
        except (KeyError, IndexError):
            return []
    
    def get_primary_node(self, cluster_id: str) -> str:
        """
        Restituisce nodo primary per cluster (ignora repliche).
        
        Args:
            cluster_id: ID cluster
            
        Returns:
            Nome nodo primary
        """
        if cluster_id in self.replicas:
            return self.replicas[cluster_id][0]
        
        return self.get_nodes(cluster_id)[0]
    
    def has_replicas(self, cluster_id: str) -> bool:
        """Check se cluster ha repliche."""
        return cluster_id in self.replicas and len(self.replicas[cluster_id]) > 1
    
    def get_replication_stats(self) -> dict:
        """Statistiche repliche."""
        total_clusters = len(self.replicas)
        total_replicas = sum(len(nodes) - 1 for nodes in self.replicas.values())
        
        return {
            'total_clusters_with_replicas': total_clusters,
            'total_replicas': total_replicas,
            'avg_replicas_per_cluster': total_replicas / total_clusters if total_clusters > 0 else 0,
            'clusters': {cid: len(nodes) for cid, nodes in self.replicas.items()}
        }