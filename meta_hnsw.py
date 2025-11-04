"""
Meta-HNSW: Routing intelligente con indice HNSW globale dei centroidi dei nodi.

AGGIORNATO: Multi-Cluster Support
- Ogni nodo può avere MULTIPLI cluster rappresentativi
- HNSW indicizza TUTTI i cluster (flat storage)
- Query aggrega risultati cluster → nodo parent
- Esempio: 10 nodi × 5 cluster = 50 entry HNSW totali

STRUTTURA:
- cluster_centroids: [c1_n1, c2_n1, c1_n2, c2_n2, ...] (flat list)
- cluster_to_node: [node1, node1, node2, node2, ...] (mapping)
- node_to_clusters: {node1: [0,1], node2: [2,3], ...} (reverse mapping)
"""

import numpy as np
import hnswlib
import joblib
from typing import List, Tuple, Dict
import os
from collections import defaultdict


class MetaHNSW:
    """
    HNSW globale per routing multi-cluster.
    
    MULTI-CLUSTER WORKFLOW:
    1. add_node_clusters(node, [cluster1, cluster2, ...]) → aggiungi N cluster per nodo
    2. find_nearest_nodes(query, k_nodes) → cerca cluster, aggrega per nodo
    3. Rebuild automatico quando necessario
    """
    
    def __init__(self, dimension: int, max_clusters: int = 500, ef_construction: int = 200, M: int = 16):
        """
        Inizializza meta-HNSW multi-cluster.
        
        Args:
            dimension: Dimensione vettori (es. 384)
            max_clusters: Massimo numero di CLUSTER totali (non nodi!)
            ef_construction: Parametro HNSW costruzione
            M: Connessioni per layer HNSW
        """
        self.dimension = dimension
        self.max_clusters = max_clusters
        self.ef_construction = ef_construction
        self.M = M
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Flat Multi-Cluster Storage
        # ═══════════════════════════════════════════════════════════════
        
        # Flat list di TUTTI i cluster (di tutti i nodi)
        self.cluster_centroids: List[np.ndarray] = []
        
        # Mapping: cluster_id → node_name
        self.cluster_to_node: List[str] = []
        
        # Reverse mapping: node_name → [cluster_ids]
        self.node_to_clusters: Dict[str, List[int]] = {}
        
        # ═══════════════════════════════════════════════════════════════
        # RIMOSSO (old single-centroid structure):
        # self.node_names = []
        # self.node_centroids = {}
        # ═══════════════════════════════════════════════════════════════
        
        # Crea indice HNSW
        self.hnsw_index = hnswlib.Index(space='cosine', dim=dimension)
        self.hnsw_index.init_index(
            max_elements=max_clusters,
            ef_construction=ef_construction,
            M=M
        )
        self.hnsw_index.set_ef(50)
        
        # Rebuild management
        self.needs_rebuild = False
        self.updates_since_rebuild = 0
        self.rebuild_drift_threshold = 0.1
        self.rebuild_every_n_updates = 10
        
        # Statistics for incremental update (per-cluster)
        self.cluster_vector_counts: Dict[int, int] = {}  # {cluster_id: count}
        self.cluster_vector_sums: Dict[int, np.ndarray] = {}  # {cluster_id: sum_vector}
        
        print(f"✓ Meta-HNSW (Multi-Cluster) initialized: {dimension}-dim, max {max_clusters} clusters")
        print(f"✓ Supports multiple clusters per node (flat HNSW storage)")
    
    def add_node_clusters(self, node_name: str, cluster_vectors: List[np.ndarray]):
        """
        Aggiungi multipli cluster per un nodo.
        
        ESEMPIO:
        node1 ha 3 cluster:
        - add_node_clusters("node1", [cluster1_vec, cluster2_vec, cluster3_vec])
        - HNSW avrà 3 entry: [c1_node1, c2_node1, c3_node1]
        
        Args:
            node_name: Nome nodo (es. "node1")
            cluster_vectors: Lista di centroidi cluster [K, dimension]
        """
        if len(cluster_vectors) == 0:
            print(f"⚠️  Nodo {node_name}: nessun cluster, skip")
            return
        
        # Normalizza tutti i cluster
        normalized_clusters = []
        for cluster_vec in cluster_vectors:
            cluster_array = np.array(cluster_vec, dtype=np.float32)
            norm = np.linalg.norm(cluster_array)
            if norm > 0:
                cluster_array = cluster_array / norm
            normalized_clusters.append(cluster_array)
        
        # Aggiungi cluster all'indice flat
        cluster_ids = []
        for cluster_array in normalized_clusters:
            cluster_id = len(self.cluster_centroids)
            
            self.cluster_centroids.append(cluster_array)
            self.cluster_to_node.append(node_name)
            cluster_ids.append(cluster_id)
            
            # Inizializza statistiche per update incrementale
            self.cluster_vector_counts[cluster_id] = 0
            self.cluster_vector_sums[cluster_id] = np.zeros(self.dimension, dtype=np.float32)
        
        # Salva mapping nodo → cluster
        if node_name in self.node_to_clusters:
            # Nodo già esiste, aggiungi cluster
            self.node_to_clusters[node_name].extend(cluster_ids)
        else:
            self.node_to_clusters[node_name] = cluster_ids
        
        # Marca rebuild necessario
        self.needs_rebuild = True
        
        print(f"✓ Nodo {node_name}: aggiunti {len(cluster_vectors)} cluster (IDs: {cluster_ids})")
    
    def _calculate_centroid_drift(self) -> dict:
        """
        Calcola drift (distanza) tra centroidi attuali e precedenti.
        
        NOTA: Nel multi-cluster, non abbiamo "previous centroids" globali.
        Questa funzione è mantenuta per compatibilità ma sempre restituisce 0.
        
        Returns:
            Dictionary {cluster_id: drift_distance} (sempre 0 per ora)
        """
        # Nel nuovo sistema multi-cluster, drift tracking non è implementato
        # perché trackiamo cluster individuali, non nodi aggregati
        return {}
    
    def _should_rebuild(self) -> tuple:
        """
        Determina se è necessario rebuild HNSW.
        
        CRITERI:
        1. needs_rebuild flag è True (rebuild esplicito richiesto)
        2. Numero update batch > intervallo rebuild
        
        Returns:
            (should_rebuild: bool, reason: str)
        """
        # Criterio 1: flag esplicito
        if self.needs_rebuild:
            return (True, "explicit rebuild flag")
        
        # Criterio 2: intervallo batch
        if self.updates_since_rebuild >= self.rebuild_every_n_updates:
            return (True, f"batch interval reached ({self.updates_since_rebuild} >= {self.rebuild_every_n_updates})")
        
        return (False, "")
    
    def force_rebuild(self):
        """
        Forza rebuild immediato dell'indice HNSW.
        
        USO:
        - Dopo ingestion massiva
        - Prima di benchmark
        - Dopo cambio configurazione
        """
        print("🔧 Forcing HNSW rebuild...")
        self._rebuild_hnsw_index()
    
    def recalculate_centroid_from_scratch(self, node_name: str, vectors: np.ndarray, method: str = 'mean') -> bool:
        """
        Ricalcola cluster di un nodo completamente da zero.
        
        NOTA: Nel multi-cluster, questo sostituisce TUTTI i cluster di un nodo
        con un SINGOLO nuovo cluster (usato per update coordinatore).
        
        Args:
            node_name: Nome del nodo
            vectors: Array di vettori [N, dimension] o [1, dimension] per singolo cluster
            method: Metodo calcolo ('mean', 'median')
            
        Returns:
            True se update riuscito
        """
        if node_name not in self.node_to_clusters:
            print(f"⚠️  Nodo {node_name} non trovato in meta-HNSW")
            return False
        
        if len(vectors.shape) == 1:
            # Singolo vettore, wrappa in array 2D
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[0] == 0:
            print(f"⚠️  Nessun vettore fornito per {node_name}")
            return False
        
        # Calcola nuovo centroide
        if method == 'mean':
            new_centroid = np.mean(vectors, axis=0)
        elif method == 'median':
            new_centroid = np.median(vectors, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalizza
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
        
        # Sostituisci TUTTI i cluster del nodo con questo singolo nuovo cluster
        old_cluster_ids = self.node_to_clusters[node_name]
        
        # Rimuovi vecchi cluster (segna come "stale", rebuild li rimuoverà)
        for old_id in old_cluster_ids:
            if old_id < len(self.cluster_to_node):
                self.cluster_to_node[old_id] = None  # Mark as deleted
        
        # Aggiungi nuovo cluster
        new_cluster_id = len(self.cluster_centroids)
        self.cluster_centroids.append(new_centroid)
        self.cluster_to_node.append(node_name)
        
        # Update mapping
        self.node_to_clusters[node_name] = [new_cluster_id]
        
        # Reset statistiche per il nuovo cluster
        self.cluster_vector_counts[new_cluster_id] = vectors.shape[0]
        self.cluster_vector_sums[new_cluster_id] = np.sum(vectors, axis=0)
        
        # Forza rebuild (necessario per rimuovere cluster "stale")
        self.needs_rebuild = True
        
        print(f"✓ Centroide {node_name} ricalcolato da {vectors.shape[0]} vettori (nuovo cluster_id: {new_cluster_id})")
        
        # Rebuild immediato per ricalcolo da scratch
        self._rebuild_hnsw_index()
        
        return True
    
    def _rebuild_hnsw_index(self):
        """Ricostruisci HNSW con TUTTI i cluster (flat), rimuovendo cluster "stale"."""
        # Filtra cluster validi (quelli con node_name non None)
        valid_clusters = []
        valid_cluster_to_node = []
        old_to_new_id = {}
        
        for old_id, (centroid, node_name) in enumerate(zip(self.cluster_centroids, self.cluster_to_node)):
            if node_name is not None:  # Cluster valido
                new_id = len(valid_clusters)
                old_to_new_id[old_id] = new_id
                valid_clusters.append(centroid)
                valid_cluster_to_node.append(node_name)
        
        # Update node_to_clusters con nuovi ID
        new_node_to_clusters = {}
        for node_name, old_cluster_ids in self.node_to_clusters.items():
            new_cluster_ids = []
            for old_id in old_cluster_ids:
                if old_id in old_to_new_id:
                    new_cluster_ids.append(old_to_new_id[old_id])
            if new_cluster_ids:  # Solo se nodo ha cluster validi
                new_node_to_clusters[node_name] = new_cluster_ids
        
        print(f"  🔧 Rebuilding HNSW: {len(valid_clusters)} valid clusters (removed {len(self.cluster_centroids) - len(valid_clusters)} stale) across {len(new_node_to_clusters)} nodes...")
        
        # Sostituisci con cluster puliti
        self.cluster_centroids = valid_clusters
        self.cluster_to_node = valid_cluster_to_node
        self.node_to_clusters = new_node_to_clusters
        
        # Crea nuovo indice HNSW
        new_index = hnswlib.Index(space='cosine', dim=self.dimension)
        new_index.init_index(
            max_elements=max(len(self.cluster_centroids), self.max_clusters),
            ef_construction=self.ef_construction,
            M=self.M
        )
        new_index.set_ef(50)
        
        # Inserisci TUTTI i cluster validi
        if self.cluster_centroids:
            centroids_array = np.array(self.cluster_centroids)
            indices = np.arange(len(self.cluster_centroids))
            new_index.add_items(centroids_array, indices)
        
        self.hnsw_index = new_index
        self.needs_rebuild = False
        self.updates_since_rebuild = 0
        
        print(f"  ✓ HNSW index rebuilt")
    
    def find_nearest_nodes(self, query_vector: np.ndarray, k: int = None, k_nodes: int = None, k_clusters: int = 10) -> List[Tuple[str, float]]:
        """
        Trova i k nodi più vicini aggregando risultati dei cluster.
        
        ALGORITMO:
        1. Query HNSW: trova top-k_clusters cluster più vicini
        2. Aggrega cluster → nodi (prendi MIN distance per nodo)
        3. Restituisci top-k_nodes nodi
        
        Args:
            query_vector: Vettore query [dimension]
            k: Alias per k_nodes (backward compatibility)
            k_nodes: Numero nodi da restituire (default: 3)
            k_clusters: Numero cluster da considerare (default: 10)
            
        Returns:
            Lista di (node_name, distance) ordinata per vicinanza
            
        Example:
            query = embedding("machine learning")
            nodes = meta_hnsw.find_nearest_nodes(query, k=3, k_clusters=10)
            # Risultato: [("node1", 0.12), ("node3", 0.15), ("node5", 0.18)]
        """
        # Backward compatibility: k è alias per k_nodes
        if k is not None and k_nodes is None:
            k_nodes = k
        elif k_nodes is None:
            k_nodes = 3  # Default
        
        if len(self.cluster_centroids) == 0:
            raise ValueError("Meta-HNSW vuoto! Aggiungi cluster prima di query.")
        
        # Lazy rebuild prima di query
        should_rebuild, reason = self._should_rebuild()
        if should_rebuild:
            print(f"  📌 Lazy rebuild before query: {reason}")
            self._rebuild_hnsw_index()
        
        # Normalizza query
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Limita k_clusters al numero totale di cluster
        k_clusters = min(k_clusters, len(self.cluster_centroids))
        
        # Query HNSW: trova top-k CLUSTER
        indices, distances = self.hnsw_index.knn_query(query_norm.reshape(1, -1), k=k_clusters)
        
        # Aggrega cluster → nodi
        node_scores = defaultdict(lambda: float('inf'))
        
        for cluster_id, dist in zip(indices[0], distances[0]):
            node_name = self.cluster_to_node[cluster_id]
            # Prendi MIN distance tra tutti i cluster di un nodo
            node_scores[node_name] = min(node_scores[node_name], dist)
        
        # Ordina nodi per best score
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1])[:k_nodes]
        
        return top_nodes
    
    def update_cluster_incremental(self, cluster_id: int, new_vector: np.ndarray) -> bool:
        """
        Aggiorna un cluster specifico in modo incrementale.
        
        Args:
            cluster_id: ID del cluster da aggiornare
            new_vector: Nuovo vettore inserito nel cluster
            
        Returns:
            True se update riuscito
        """
        if cluster_id >= len(self.cluster_centroids):
            print(f"⚠️  Cluster ID {cluster_id} non trovato")
            return False
        
        # Update running statistics
        old_count = self.cluster_vector_counts.get(cluster_id, 0)
        old_sum = self.cluster_vector_sums.get(cluster_id, np.zeros(self.dimension, dtype=np.float32))
        
        new_count = old_count + 1
        new_sum = old_sum + new_vector
        
        # Calcola nuovo centroide
        new_centroid = new_sum / new_count
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
        
        # Update stored data
        self.cluster_centroids[cluster_id] = new_centroid
        self.cluster_vector_counts[cluster_id] = new_count
        self.cluster_vector_sums[cluster_id] = new_sum
        
        return True
    
    def save(self, path: str):
        """Salva meta-HNSW multi-cluster su disco."""
        if self.needs_rebuild or self.updates_since_rebuild > 0:
            print("  🔧 Final rebuild before save...")
            self._rebuild_hnsw_index()
        
        data = {
            'dimension': self.dimension,
            'max_clusters': self.max_clusters,
            # NUOVO: Salva flat multi-cluster structure
            'cluster_centroids': [c.tolist() for c in self.cluster_centroids],
            'cluster_to_node': self.cluster_to_node,
            'node_to_clusters': self.node_to_clusters,
            'cluster_vector_counts': self.cluster_vector_counts,
            'cluster_vector_sums': {k: v.tolist() for k, v in self.cluster_vector_sums.items()},
            'rebuild_drift_threshold': self.rebuild_drift_threshold,
            'rebuild_every_n_updates': self.rebuild_every_n_updates
        }
        
        # Salva indice HNSW
        hnsw_path = path + '.hnsw'
        self.hnsw_index.save_index(hnsw_path)
        
        # Salva metadata
        joblib.dump(data, path)
        
        print(f"✓ Meta-HNSW (multi-cluster) salvato: {path}")
        print(f"  - {len(self.cluster_centroids)} cluster totali")
        print(f"  - {len(self.node_to_clusters)} nodi")
    
    @classmethod
    def load(cls, path: str) -> 'MetaHNSW':
        """Carica meta-HNSW multi-cluster da disco."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Meta-HNSW non trovato: {path}")
        
        # Carica metadata
        data = joblib.load(path)
        
        # Crea oggetto
        meta_hnsw = cls(
            dimension=data['dimension'],
            max_clusters=data['max_clusters']
        )
        
        # Carica indice HNSW
        hnsw_path = path + '.hnsw'
        meta_hnsw.hnsw_index.load_index(hnsw_path, max_elements=data['max_clusters'])
        
        # NUOVO: Ripristina flat multi-cluster structure
        meta_hnsw.cluster_centroids = [np.array(c, dtype=np.float32) for c in data['cluster_centroids']]
        meta_hnsw.cluster_to_node = data['cluster_to_node']
        meta_hnsw.node_to_clusters = data['node_to_clusters']
        meta_hnsw.cluster_vector_counts = data.get('cluster_vector_counts', {})
        
        cluster_sums = data.get('cluster_vector_sums', {})
        meta_hnsw.cluster_vector_sums = {
            int(k): np.array(v, dtype=np.float32) for k, v in cluster_sums.items()
        }
        
        meta_hnsw.rebuild_drift_threshold = data.get('rebuild_drift_threshold', 0.1)
        meta_hnsw.rebuild_every_n_updates = data.get('rebuild_every_n_updates', 10)
        
        # Reset contatori
        meta_hnsw.needs_rebuild = False
        meta_hnsw.updates_since_rebuild = 0
        
        print(f"✓ Meta-HNSW (multi-cluster) caricato: {path}")
        print(f"  - {len(meta_hnsw.cluster_centroids)} cluster totali")
        print(f"  - {len(meta_hnsw.node_to_clusters)} nodi")
        
        return meta_hnsw
    
    def get_statistics(self) -> dict:
        """Restituisce statistiche meta-HNSW multi-cluster."""
        if not self.cluster_centroids:
            return {}
        
        # Calcola statistiche per NODI (aggregando cluster)
        from scipy.spatial.distance import pdist, squareform
        
        # Calcola centroide medio per ogni nodo (per statistiche)
        node_avg_centroids = {}
        for node_name, cluster_ids in self.node_to_clusters.items():
            node_clusters = [self.cluster_centroids[cid] for cid in cluster_ids]
            node_avg_centroids[node_name] = np.mean(node_clusters, axis=0)
        
        node_centroids_array = np.array(list(node_avg_centroids.values()))
        
        if len(node_centroids_array) > 1:
            node_distances = squareform(pdist(node_centroids_array, metric='cosine'))
            non_zero = node_distances[node_distances > 0]
            
            min_dist = float(non_zero.min()) if len(non_zero) > 0 else 0
            max_dist = float(non_zero.max()) if len(non_zero) > 0 else 0
            mean_dist = float(non_zero.mean()) if len(non_zero) > 0 else 0
        else:
            min_dist = max_dist = mean_dist = 0
        
        # Cluster per nodo statistics
        clusters_per_node = {
            node: len(cluster_ids) for node, cluster_ids in self.node_to_clusters.items()
        }
        
        total_vectors_tracked = sum(self.cluster_vector_counts.values())
        
        return {
            'num_nodes': len(self.node_to_clusters),
            'num_clusters_total': len(self.cluster_centroids),
            'clusters_per_node': clusters_per_node,
            'avg_clusters_per_node': len(self.cluster_centroids) / len(self.node_to_clusters) if self.node_to_clusters else 0,
            'dimension': self.dimension,
            'min_node_distance': min_dist,
            'max_node_distance': max_dist,
            'mean_node_distance': mean_dist,
            'nodes': list(self.node_to_clusters.keys()),
            'total_vectors_tracked': total_vectors_tracked,
            'needs_rebuild': self.needs_rebuild,
            'updates_since_rebuild': self.updates_since_rebuild,
            'rebuild_threshold': self.rebuild_drift_threshold
        }
