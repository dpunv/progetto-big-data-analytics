"""
Capacity-Constrained Cluster Rebalancer.

SCOPO:
- Monitora carico nodi durante ingestion
- Riassegna vettori quando nodo supera capacità
- Mantiene bilanciamento automatico del sistema

WORKFLOW:
1. check_node_capacity() → verifica se nodo è over-capacity
2. identify_vectors_to_reassign() → trova vettori da spostare
3. find_alternative_cluster() → trova cluster alternativo con spazio
4. execute_reassignment() → aggiorna routing table e metadata
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NodeCapacity:
    """Statistiche capacità di un nodo."""
    node_name: str
    current_vectors: int
    max_vectors: int
    current_gb: float
    max_gb: float
    utilization_pct: float
    is_over_capacity: bool


class CapacityConstrainedRebalancer:
    """
    Gestisce rebalancing automatico basato su capacità nodi.
    
    PROCESSO:
    1. Durante ingestion: conta vettori per nodo
    2. Quando nodo supera threshold → trigger rebalancing
    3. Identifica vettori da riassegnare (più lontani dal centroide)
    4. Trova cluster alternativo con spazio
    5. Aggiorna routing table
    
    ESEMPIO:
    node-1: 16K vettori (capacity 15K) → overflow 1K
    - Trova 1K vettori più lontani da centroid_cluster_0
    - Per ognuno: calcola secondo cluster più vicino
    - Riassegna a quel cluster (se ha spazio)
    """
    
    def __init__(self, 
                 max_capacity_vectors: int,
                 max_capacity_gb: float,
                 threshold: float = 0.9,
                 vector_dimension: int = 384):
        """
        Inizializza rebalancer.
        
        Args:
            max_capacity_vectors: Capacità massima in numero di vettori
            max_capacity_gb: Capacità massima in GB
            threshold: Soglia per trigger rebalancing (0.9 = 90%)
            vector_dimension: Dimensione vettori (per calcolo GB)
        """
        self.max_capacity_vectors = max_capacity_vectors
        self.max_capacity_gb = max_capacity_gb
        self.threshold = threshold
        self.vector_dimension = vector_dimension
        
        # 4 bytes per float32, poi converti in GB
        self.bytes_per_vector = vector_dimension * 4
        self.gb_per_vector = self.bytes_per_vector / (1024**3)
        
        # Statistiche rebalancing
        self.rebalancing_events = []
        self.total_vectors_reassigned = 0
        
        print(f"✓ Rebalancer initialized:")
        print(f"  Max capacity: {max_capacity_vectors:,} vectors or {max_capacity_gb:.2f} GB")
        print(f"  Rebalancing threshold: {threshold*100:.0f}%")
        print(f"  Vector size: {self.bytes_per_vector} bytes ({self.gb_per_vector*1e6:.2f} MB)")
    
    def check_node_capacity(self, 
                           node_name: str, 
                           current_vectors: int) -> NodeCapacity:
        """
        Verifica capacità di un nodo.
        
        Args:
            node_name: Nome del nodo
            current_vectors: Numero attuale di vettori nel nodo
            
        Returns:
            NodeCapacity object con statistiche
        """
        current_gb = current_vectors * self.gb_per_vector
        
        # Utilization come massimo tra vettori % e GB %
        utilization_vectors = current_vectors / self.max_capacity_vectors
        utilization_gb = current_gb / self.max_capacity_gb
        utilization_pct = max(utilization_vectors, utilization_gb)
        
        is_over_capacity = utilization_pct >= self.threshold
        
        return NodeCapacity(
            node_name=node_name,
            current_vectors=current_vectors,
            max_vectors=self.max_capacity_vectors,
            current_gb=current_gb,
            max_gb=self.max_capacity_gb,
            utilization_pct=utilization_pct,
            is_over_capacity=is_over_capacity
        )
    
    def identify_vectors_to_reassign(self,
                                     vectors: np.ndarray,
                                     cluster_centroid: np.ndarray,
                                     n_to_reassign: int,
                                     strategy: str = 'distance') -> np.ndarray:
        """
        Identifica quali vettori riassegnare da un cluster over-capacity.
        
        STRATEGIE:
        - 'distance': rimuovi i vettori più lontani dal centroide (mantiene coerenza semantica)
        - 'random': rimuovi vettori casuali (più veloce)
        - 'round_robin': rimuovi in modo equo
        
        Args:
            vectors: Array di vettori nel cluster [N, dim]
            cluster_centroid: Centroide del cluster [dim]
            n_to_reassign: Numero di vettori da riassegnare
            strategy: Strategia di selezione
            
        Returns:
            Indices dei vettori da riassegnare
        """
        if strategy == 'distance':
            # Calcola distanza di ogni vettore dal centroide
            distances = np.linalg.norm(vectors - cluster_centroid, axis=1)
            
            # Restituisci indici dei vettori più lontani
            farthest_indices = np.argsort(distances)[-n_to_reassign:]
            return farthest_indices
        
        elif strategy == 'random':
            # Selezione casuale
            return np.random.choice(len(vectors), n_to_reassign, replace=False)
        
        elif strategy == 'round_robin':
            # Prendi vettori spaziati uniformemente
            step = len(vectors) // n_to_reassign
            return np.arange(0, len(vectors), step)[:n_to_reassign]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def find_alternative_clusters(self,
                                  vector: np.ndarray,
                                  all_centroids: np.ndarray,
                                  current_cluster_id: int,
                                  node_capacities: Dict[int, NodeCapacity]) -> List[Tuple[int, float]]:
        """
        Trova cluster alternativi con spazio disponibile, ordinati per vicinanza.
        
        Args:
            vector: Vettore da riassegnare [dim]
            all_centroids: Tutti i centroidi K-means [n_clusters, dim]
            current_cluster_id: Cluster attuale del vettore
            node_capacities: Capacità di ogni nodo {cluster_id: NodeCapacity}
            
        Returns:
            Lista di (cluster_id, distance) ordinata per vicinanza, esclusi cluster pieni
        """
        # Calcola distanze da tutti i centroidi
        distances = np.linalg.norm(all_centroids - vector, axis=1)
        
        # Ordina per distanza crescente
        sorted_indices = np.argsort(distances)
        
        # Filtra: escludi cluster attuale e cluster pieni
        alternatives = []
        for cluster_id in sorted_indices:
            # Skip cluster attuale
            if cluster_id == current_cluster_id:
                continue
            
            # Skip se nodo target è pieno
            if cluster_id in node_capacities:
                capacity = node_capacities[cluster_id]
                if capacity.is_over_capacity:
                    continue
            
            alternatives.append((int(cluster_id), float(distances[cluster_id])))
        
        return alternatives
    
    def calculate_rebalancing_plan(self,
                                   node_loads: Dict[str, int],
                                   routing_table,
                                   cluster_centroids: np.ndarray,
                                   vectors_by_cluster: Dict[int, np.ndarray]) -> Dict:
        """
        Calcola piano di rebalancing per tutti i nodi over-capacity.
        
        Args:
            node_loads: Carico attuale per nodo {node_name: num_vectors}
            routing_table: RoutingTable object
            cluster_centroids: Centroidi K-means [n_clusters, dim]
            vectors_by_cluster: Vettori per ogni cluster {cluster_id: vectors_array}
            
        Returns:
            Piano di rebalancing: {
                'nodes_to_rebalance': [...],
                'reassignments': [(vector_idx, old_cluster, new_cluster), ...],
                'total_vectors_to_move': int
            }
        """
        plan = {
            'nodes_to_rebalance': [],
            'reassignments': [],
            'total_vectors_to_move': 0
        }
        
        # 1. Identifica nodi over-capacity
        node_capacities = {}
        for node_name, load in node_loads.items():
            capacity = self.check_node_capacity(node_name, load)
            
            if capacity.is_over_capacity:
                plan['nodes_to_rebalance'].append(node_name)
                print(f"\n⚠️  {node_name} is over-capacity:")
                print(f"    Current: {capacity.current_vectors:,} vectors ({capacity.utilization_pct*100:.1f}%)")
                print(f"    Max: {capacity.max_vectors:,} vectors")
                
                # Calcola overflow
                target_vectors = int(capacity.max_vectors * 0.85)  # Target 85% (safety margin)
                n_to_reassign = capacity.current_vectors - target_vectors
                
                print(f"    Need to reassign: {n_to_reassign:,} vectors")
                
                # 2. Per ogni cluster in questo nodo
                assigned_clusters = routing_table.get_clusters_for_node(node_name)
                
                for cluster_id in assigned_clusters:
                    if cluster_id not in vectors_by_cluster:
                        continue
                    
                    cluster_vectors = vectors_by_cluster[cluster_id]
                    cluster_centroid = cluster_centroids[cluster_id]
                    
                    # Numero di vettori da questo cluster da riassegnare
                    # (proporzionale alla dimensione del cluster)
                    cluster_share = len(cluster_vectors) / capacity.current_vectors
                    n_from_this_cluster = int(n_to_reassign * cluster_share)
                    
                    if n_from_this_cluster == 0:
                        continue
                    
                    # Identifica vettori da riassegnare
                    indices_to_reassign = self.identify_vectors_to_reassign(
                        vectors=cluster_vectors,
                        cluster_centroid=cluster_centroid,
                        n_to_reassign=n_from_this_cluster,
                        strategy='distance'
                    )
                    
                    # Per ogni vettore: trova cluster alternativo
                    for vec_idx in indices_to_reassign:
                        vector = cluster_vectors[vec_idx]
                        
                        # Trova cluster alternativi
                        alternatives = self.find_alternative_clusters(
                            vector=vector,
                            all_centroids=cluster_centroids,
                            current_cluster_id=cluster_id,
                            node_capacities={}  # TODO: passa capacità reali
                        )
                        
                        if alternatives:
                            # Prendi cluster più vicino con spazio
                            new_cluster_id, distance = alternatives[0]
                            
                            plan['reassignments'].append({
                                'vector_idx': int(vec_idx),
                                'old_cluster': int(cluster_id),
                                'new_cluster': int(new_cluster_id),
                                'distance_increase': float(distance),
                                'vector': vector
                            })
                            
                            plan['total_vectors_to_move'] += 1
        
        return plan
    
    def get_rebalancing_statistics(self) -> Dict:
        """Restituisce statistiche rebalancing."""
        return {
            'total_events': len(self.rebalancing_events),
            'total_vectors_reassigned': self.total_vectors_reassigned,
            'events': self.rebalancing_events
        }


def simulate_capacity_aware_assignment(vectors: np.ndarray,
                                       cluster_assignments: np.ndarray,
                                       cluster_centroids: np.ndarray,
                                       max_capacity_per_cluster: int) -> Tuple[np.ndarray, Dict]:
    """
    Simula assignment con capacity constraints (per testing/validation).
    
    Args:
        vectors: Tutti i vettori [N, dim]
        cluster_assignments: Assignmen iniziali [N] (da K-means)
        cluster_centroids: Centroidi K-means [n_clusters, dim]
        max_capacity_per_cluster: Capacità massima per cluster
        
    Returns:
        (new_assignments, statistics)
    """
    n_vectors = len(vectors)
    n_clusters = len(cluster_centroids)
    
    new_assignments = cluster_assignments.copy()
    cluster_counts = np.bincount(cluster_assignments, minlength=n_clusters)
    
    statistics = {
        'initial_max_load': int(cluster_counts.max()),
        'initial_min_load': int(cluster_counts.min()),
        'vectors_reassigned': 0,
        'overflow_clusters': []
    }
    
    # Identifica cluster over-capacity
    overflow_clusters = np.where(cluster_counts > max_capacity_per_cluster)[0]
    
    for cluster_id in overflow_clusters:
        overflow = int(cluster_counts[cluster_id] - max_capacity_per_cluster)
        statistics['overflow_clusters'].append({
            'cluster_id': int(cluster_id),
            'overflow': overflow
        })
        
        # Trova vettori in questo cluster
        cluster_mask = (cluster_assignments == cluster_id)
        cluster_vectors = vectors[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Calcola distanze dal centroide
        centroid = cluster_centroids[cluster_id]
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        # Prendi i più lontani
        farthest_local_indices = np.argsort(distances)[-overflow:]
        farthest_global_indices = cluster_indices[farthest_local_indices]
        
        # Riassegna a secondo cluster più vicino
        for vec_idx in farthest_global_indices:
            vector = vectors[vec_idx]
            
            # Calcola distanze da tutti i centroidi
            all_distances = np.linalg.norm(cluster_centroids - vector, axis=1)
            
            # Ordina per vicinanza (escludi cluster corrente)
            sorted_clusters = np.argsort(all_distances)
            
            # Trova primo cluster con spazio
            for alt_cluster_id in sorted_clusters:
                if alt_cluster_id == cluster_id:
                    continue
                
                if cluster_counts[alt_cluster_id] < max_capacity_per_cluster:
                    # Riassegna
                    new_assignments[vec_idx] = alt_cluster_id
                    cluster_counts[cluster_id] -= 1
                    cluster_counts[alt_cluster_id] += 1
                    statistics['vectors_reassigned'] += 1
                    break
    
    statistics['final_max_load'] = int(cluster_counts.max())
    statistics['final_min_load'] = int(cluster_counts.min())
    statistics['final_imbalance'] = statistics['final_max_load'] - statistics['final_min_load']
    
    return new_assignments, statistics
