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
        Assegna cluster ai nodi usando Minimum Spanning Tree (MST) semantico.
        
        STRATEGIA:
        1. Calcola matrice distanze tra centroidi cluster
        2. Costruisce MST (Minimum Spanning Tree)
        3. Taglia MST in N sottografi (N = numero nodi)
        4. Ogni sottografo → 1 nodo
        
        VANTAGGI:
        - Cluster semanticamente simili → stesso nodo
        - Migliora locality query
        - Riduce cross-node queries
        
        Args:
            n_clusters: Numero totale cluster
            centroids: Centroidi K-means [n_clusters, dimension]
        """
        # Calcola matrice distanze tra centroidi (cosine distance)
        distances = squareform(pdist(centroids, metric='cosine'))
        
        # Costruisci MST
        mst = minimum_spanning_tree(distances)
        
        # Converti MST in grafo non orientato
        mst_undirected = mst + mst.T
        
        # Taglia MST in N sottografi (dove N = numero nodi)
        # Strategia: rimuovi N-1 edge più lunghi
        edges = []
        rows, cols = mst_undirected.nonzero()
        for i, j in zip(rows, cols):
            if i < j:  # Evita duplicati (edge i→j e j→i)
                weight = mst_undirected[i, j]
                edges.append((weight, i, j))
        
        # Ordina edge per peso (decrescente)
        edges.sort(reverse=True)
        
        # Rimuovi top N-1 edge per creare N sottografi
        num_nodes = len(self._node_names)
        edges_to_remove = edges[:num_nodes - 1]
        
        # Crea grafo senza edge rimossi
        cluster_groups = list(range(n_clusters))  # Inizialmente ogni cluster è gruppo separato
        
        # Union-Find per raggruppare cluster connessi
        parent = list(range(n_clusters))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Unisci cluster connessi (escludendo edge rimossi)
        removed_edges_set = {(min(i, j), max(i, j)) for _, i, j in edges_to_remove}
        
        for i, j in zip(rows, cols):
            if i < j and (i, j) not in removed_edges_set:
                union(i, j)
        
        # Raggruppa cluster per componente connessa
        groups = {}
        for cluster_id in range(n_clusters):
            root = find(cluster_id)
            if root not in groups:
                groups[root] = []
            groups[root].append(cluster_id)
        
        # Assegna gruppi ai nodi (round-robin se gruppi < nodi)
        nodes_list = list(self._node_names)
        group_list = list(groups.values())
        
        for idx, cluster_group in enumerate(group_list):
            assigned_node = nodes_list[idx % len(nodes_list)]
            for cluster_id in cluster_group:
                self.routing_table[assigned_node].append(cluster_id)
        
        print(f"✓ Assigned {n_clusters} clusters to {len(nodes_list)} nodes (semantic MST)")
        
        # Stampa distribuzione
        for node, clusters in self.routing_table.items():
            print(f"  {node}: {len(clusters)} clusters → {clusters}")
    
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
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Gestione nodi non disponibili
    # ═══════════════════════════════════════════════════════════════
    def mark_node_unavailable(self, node_name: str):
        """
        Marca node_name come non disponibile: rimuove il nodo dalla routing_table
        e rialloca i suoi cluster agli altri nodi disponibili (round-robin).
        Rimuove inoltre il nodo dalle liste di repliche.
        """
        if node_name not in self._node_names:
            return
        
        # Rimuovi e salva cluster da riallocare
        clusters_to_move = self.routing_table.pop(node_name, [])
        
        # Aggiorna lista nodi disponibili
        remaining_nodes = [n for n in self._node_names if n != node_name]
        self._node_names = remaining_nodes
        
        # Se non ci sono nodi rimasti, lascia i cluster non assegnati (caller deve gestire)
        if not remaining_nodes:
            print(f"⚠️  All nodes removed; cannot reassign clusters from {node_name}")
            return
        
        # Rialloca clusters round-robin
        for i, cid in enumerate(clusters_to_move):
            target = remaining_nodes[i % len(remaining_nodes)]
            if cid not in self.routing_table[target]:
                self.routing_table[target].append(cid)
        
        # Rimuovi nodo dalle repliche dove presente
        for cid, nodes in list(self.replicas.items()):
            if node_name in nodes:
                nodes = [n for n in nodes if n != node_name]
                if not nodes:
                    # se non rimane neanche il primary, elimina entry
                    self.replicas.pop(cid, None)
                else:
                    self.replicas[cid] = nodes

    def get_available_nodes(self) -> list[str]:
        """Ritorna lista nodi attualmente considerati disponibili per routing."""
        return list(self._node_names)