# routing.py
import threading

class RoutingTable:
    """
    Gestisce la mappatura tra cluster_id e nodi Qdrant.
    È thread-safe per gestire aggiornamenti concorrenti.
    """
    def __init__(self, node_names: list[str]):
        self._node_names = node_names
        self._routing_map = {}
        self._lock = threading.Lock()

    def assign_initial_clusters(self, n_clusters: int):
        """Assegna i cluster iniziali ai nodi in modo round-robin."""
        with self._lock:
            for i in range(n_clusters):
                node = self._node_names[i % len(self._node_names)]
                self._routing_map[str(i)] = [node] # Mappa a una lista per supportare futuri split
            print("Initial routing table created.")
            # print(self._routing_map)

    def get_nodes(self, cluster_id: str) -> list[str]:
        """Ottiene i nodi target per un dato cluster_id."""
        with self._lock:
            # cluster_id potrebbe essere "5" o "5.0", "5.1" dopo uno split.
            # Cerchiamo la corrispondenza più specifica.
            if cluster_id in self._routing_map:
                return self._routing_map[cluster_id]
            
            # Se è un sub-cluster, controlliamo la radice (es. "5" per "5.0")
            base_cluster = cluster_id.split('.')[0]
            if base_cluster in self._routing_map:
                return self._routing_map[base_cluster]
                
            raise KeyError(f"Cluster ID {cluster_id} not found in routing table.")

    def split_cluster_mapping(self, old_cluster_id: int, new_sub_cluster_ids: list[str], node_assignments: dict[str, str]):
        """
        Aggiorna la tabella di routing per riflettere uno split.
        Questa operazione è atomica dal punto di vista logico.
        """
        with self._lock:
            print(f"Updating routing table: Splitting cluster {old_cluster_id}...")
            
            # Rimuoviamo la vecchia mappatura del cluster padre
            if str(old_cluster_id) in self._routing_map:
                del self._routing_map[str(old_cluster_id)]

            # Aggiungiamo le nuove mappature per i sub-cluster
            for sub_id in new_sub_cluster_ids:
                node = node_assignments[sub_id]
                self._routing_map[sub_id] = [node]
                print(f"  -> Sub-cluster {sub_id} assigned to {node}")
        
        print(f"Routing table updated for split of cluster {old_cluster_id}.")