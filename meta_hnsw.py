"""
Meta-HNSW: Routing intelligente con indice HNSW globale dei centroidi dei nodi.

IDEA:
- Ogni nodo Qdrant ha un centroide rappresentativo (media dei suoi vettori)
- Tutti i centroidi sono indicizzati in un piccolo HNSW globale
- Query/Insert: cerca nel meta-HNSW i top-k nodi più vicini
- Interroga solo quei nodi invece di fare broadcast

VANTAGGI:
- Riduce latenza: O(log N) invece di O(N) nodi da interrogare
- Scalabilità: funziona bene anche con 100+ nodi
- Precisione: routing semantico basato su similarità reale

ESEMPIO:
Query "machine learning" → meta-HNSW trova nodi tech (cluster 0,1,2)
→ interroga solo quei 3 nodi invece di tutti i 10
"""

import numpy as np
import hnswlib
import joblib
from typing import List, Tuple
import os


class MetaHNSW:
    """
    Indice HNSW globale per routing basato su centroidi dei nodi.
    
    GESTIONE UPDATE CENTROIDI:
    - HNSW non supporta update in-place sicuro
    - Soluzione: rebuild indice quando centroidi cambiano significativamente
    - Ottimizzazione: rebuild solo se drift > threshold
    
    STRATEGIE REBUILD:
    1. Lazy rebuild: rebuild solo quando serve (prima di query)
    2. Threshold-based: rebuild se centroidi cambiano > X%
    3. Periodic rebuild: ogni N update batch
    
    STRUTTURA:
    - self.node_centroids: dict {node_name: centroid_vector}
    - self.hnsw_index: hnswlib index con tutti i centroidi
    - self.node_names: mapping [index_id → node_name]
    
    WORKFLOW:
    1. add_node_centroid(node, vectors) → calcola centroide, aggiungi a HNSW
    2. find_nearest_nodes(query_vector, k) → cerca top-k nodi simili
    3. save/load → persistenza su disco
    """
    
    def __init__(self, dimension: int, max_nodes: int = 100, ef_construction: int = 200, M: int = 16):
        """
        Inizializza meta-HNSW.
        
        Args:
            dimension: Dimensione vettori (es. 384 per sentence-transformers)
            max_nodes: Massimo numero di nodi (capacità iniziale)
            ef_construction: Parametro HNSW per costruzione (più alto = più preciso ma più lento)
            M: Numero di connessioni per layer HNSW (default 16 è buon compromesso)
        """
        self.dimension = dimension
        self.max_nodes = max_nodes
        self.ef_construction = ef_construction
        self.M = M
        
        # Crea indice HNSW
        self.hnsw_index = hnswlib.Index(space='cosine', dim=dimension)
        self.hnsw_index.init_index(
            max_elements=max_nodes,
            ef_construction=ef_construction,
            M=M
        )
        self.hnsw_index.set_ef(50)
        
        # Mapping index_id → node_name
        self.node_names = []
        
        # Centroidi dei nodi {node_name: centroid_array}
        self.node_centroids = {}
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Gestione rebuild HNSW
        # ═══════════════════════════════════════════════════════════════
        
        # Flag: indica se HNSW deve essere ricostruito
        self.needs_rebuild = False
        
        # Conta update batch da ultimo rebuild
        self.updates_since_rebuild = 0
        
        # Centroidi precedenti (per calcolare drift)
        self.previous_centroids = {}
        
        # Threshold per rebuild basato su drift (distanza cosine)
        self.rebuild_drift_threshold = 0.1  # Se centroide si sposta > 0.1, rebuild
        
        # Rebuild ogni N update batch
        self.rebuild_every_n_updates = 10  # Rebuild ogni 10 batch update
        
        # Statistiche per update incrementale
        self.node_vector_counts = {}
        self.node_vector_sums = {}
        
        print(f"✓ Meta-HNSW initialized: {dimension}-dim, max {max_nodes} nodes")
        print(f"✓ Incremental update enabled")
        print(f"✓ Rebuild threshold: {self.rebuild_drift_threshold:.3f} cosine distance")
        print(f"✓ Rebuild interval: every {self.rebuild_every_n_updates} batch updates")
    
    def _calculate_centroid_drift(self) -> dict[str, float]:
        """
        Calcola drift (distanza) tra centroidi attuali e precedenti.
        
        SCOPO:
        - Determina se centroidi sono cambiati significativamente
        - Usa distanza cosine (più adatta per embeddings normalizzati)
        
        Returns:
            Dictionary {node_name: drift_distance}
        """
        drift = {}
        
        for node_name, current_centroid in self.node_centroids.items():
            if node_name not in self.previous_centroids:
                drift[node_name] = float('inf')  # Prima volta: rebuild necessario
                continue
            
            prev_centroid = self.previous_centroids[node_name]
            
            # Calcola distanza cosine
            norm_current = np.linalg.norm(current_centroid)
            norm_prev = np.linalg.norm(prev_centroid)
            
            if norm_current == 0 or norm_prev == 0:
                drift[node_name] = float('inf')
                continue
            
            # Cosine distance = 1 - cosine similarity
            cosine_sim = np.dot(current_centroid, prev_centroid) / (norm_current * norm_prev)
            cosine_dist = 1 - cosine_sim
            
            drift[node_name] = float(cosine_dist)
        
        return drift
    
    def _should_rebuild(self) -> Tuple[bool, str]:
        """
        Determina se è necessario rebuild HNSW.
        
        CRITERI:
        1. needs_rebuild flag è True (rebuild esplicito richiesto)
        2. Drift centroidi > threshold
        3. Numero update batch > intervallo rebuild
        
        Returns:
            (should_rebuild, reason)
        """
        # Criterio 1: flag esplicito
        if self.needs_rebuild:
            return (True, "explicit rebuild flag")
        
        # Criterio 2: drift threshold
        drift = self._calculate_centroid_drift()
        if drift:
            max_drift = max(drift.values())
            max_drift_node = max(drift.items(), key=lambda x: x[1])[0]
            
            if max_drift > self.rebuild_drift_threshold:
                return (True, f"centroid drift {max_drift:.4f} > threshold {self.rebuild_drift_threshold} (node: {max_drift_node})")
        
        # Criterio 3: intervallo batch
        if self.updates_since_rebuild >= self.rebuild_every_n_updates:
            return (True, f"batch interval reached ({self.updates_since_rebuild} >= {self.rebuild_every_n_updates})")
        
        return (False, "")
    
    def _rebuild_hnsw_index(self):
        """
        Ricostruisce completamente l'indice HNSW.
        
        PROCESSO:
        1. Crea nuovo indice HNSW vuoto
        2. Inserisci tutti i centroidi attuali
        3. Aggiorna previous_centroids
        4. Reset contatori
        
        PERCHÉ È NECESSARIO:
        - hnswlib.add_items() con stesso ID può creare duplicati
        - Rebuild garantisce coerenza: 1 centroide = 1 entry
        
        COSTO:
        - O(N log N) dove N = numero nodi
        - Con 10-100 nodi: ~10-100ms (accettabile)
        - Con 1000+ nodi: ~1-5s (fare rebuild meno frequente)
        """
        print(f"  🔧 Rebuilding HNSW index ({len(self.node_centroids)} nodes)...")
        
        # Crea nuovo indice
        new_index = hnswlib.Index(space='cosine', dim=self.dimension)
        new_index.init_index(
            max_elements=self.max_nodes,
            ef_construction=self.ef_construction,
            M=self.M
        )
        new_index.set_ef(50)
        
        # Inserisci tutti i centroidi
        if self.node_centroids:
            centroids_array = np.array([
                self.node_centroids[node] for node in self.node_names
            ])
            indices = np.arange(len(self.node_names))
            
            new_index.add_items(centroids_array, indices)
        
        # Sostituisci vecchio indice
        self.hnsw_index = new_index
        
        # Salva snapshot centroidi attuali
        self.previous_centroids = {
            name: centroid.copy() for name, centroid in self.node_centroids.items()
        }
        
        # Reset contatori
        self.needs_rebuild = False
        self.updates_since_rebuild = 0
        
        print(f"  ✓ HNSW index rebuilt")
    
    def add_node_centroid(self, node_name: str, vectors: np.ndarray, method: str = 'mean'):
        """
        Calcola centroide di un nodo e aggiungilo al meta-HNSW.
        
        METODI DI CALCOLO CENTROIDE:
        - 'mean': media aritmetica (veloce, sensibile a outlier)
        - 'median': mediana (più robusto, ma più lento)
        - 'weighted': media ponderata per densità locale (avanzato)
        
        Args:
            node_name: Nome del nodo (es. "node-1")
            vectors: Array di vettori contenuti nel nodo [N, dim]
            method: Metodo di calcolo centroide
        """
        if len(vectors) == 0:
            print(f"⚠️  Nodo {node_name}: nessun vettore, skip centroide")
            return
        
        # Calcola centroide
        if method == 'mean':
            centroid = np.mean(vectors, axis=0)
        elif method == 'median':
            centroid = np.median(vectors, axis=0)
        elif method == 'weighted':
            from scipy.spatial.distance import pdist, squareform
            
            if len(vectors) > 1000:
                sample_indices = np.random.choice(len(vectors), 1000, replace=False)
                sample = vectors[sample_indices]
            else:
                sample = vectors
            
            distances = squareform(pdist(sample, metric='euclidean'))
            avg_distances = distances.mean(axis=1)
            weights = 1.0 / (avg_distances + 1e-6)
            weights /= weights.sum()
            centroid = np.average(sample, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalizza centroide
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        # Salva mapping
        if node_name not in self.node_names:
            self.node_names.append(node_name)
        
        self.node_centroids[node_name] = centroid
        
        # Inizializza statistiche per update incrementale
        self.node_vector_counts[node_name] = len(vectors)
        self.node_vector_sums[node_name] = np.sum(vectors, axis=0)
        
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Segna rebuild necessario invece di add_items diretto
        # ═══════════════════════════════════════════════════════════════
        self.needs_rebuild = True
        
        print(f"✓ Nodo {node_name}: centroide calcolato ({len(vectors)} vettori, method={method})")
    
    def update_centroid_incremental(self, node_name: str, new_vector: np.ndarray) -> bool:
        """
        Aggiorna centroide di un nodo in modo incrementale (running average).
        
        VANTAGGI:
        - O(1) complexity: solo somma di vettori
        - Nessun fetch da Qdrant
        - Centroidi sempre aggiornati
        
        FORMULA:
        new_centroid = (old_sum + new_vector) / (old_count + 1)
        
        ESEMPIO:
        Nodo ha 1000 vettori, centroide = media di 1000 vettori
        Inserisci vettore 1001-esimo:
        - new_sum = old_sum + vector_1001
        - new_count = 1001
        - new_centroid = new_sum / 1001
        
        QUANDO USARE:
        - Ad ogni inserimento: overhead minimo (~0.1ms)
        - Oppure ogni N inserimenti (batch update)
        
        Args:
            node_name: Nome del nodo da aggiornare
            new_vector: Nuovo vettore inserito nel nodo
            
        Returns:
            True se update riuscito, False se nodo non esiste
        """
        if node_name not in self.node_centroids:
            print(f"⚠️  Nodo {node_name} non trovato in meta-HNSW")
            return False
        
        # Update running statistics
        old_count = self.node_vector_counts[node_name]
        old_sum = self.node_vector_sums[node_name]
        
        new_count = old_count + 1
        new_sum = old_sum + new_vector
        
        # Calcola nuovo centroide (running average)
        new_centroid = new_sum / new_count
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
        
        # Update stored data
        self.node_centroids[node_name] = new_centroid
        self.node_vector_counts[node_name] = new_count
        self.node_vector_sums[node_name] = new_sum
        
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Marca rebuild se necessario (lazy rebuild)
        # ═══════════════════════════════════════════════════════════════
        # Non rebuild subito (troppo costoso), lo faremo prima della prossima query
        # o quando drift supera threshold
        
        return True
    
    def update_centroid_batch(self, node_name: str, new_vectors: np.ndarray) -> bool:
        """
        Aggiorna centroide con batch di vettori (più efficiente di N chiamate incrementali).
        
        VANTAGGI:
        - Più efficiente: un solo update HNSW invece di N
        - Meno overhead: aggregazione in memoria
        
        QUANDO USARE:
        - Ogni K inserimenti (es. ogni 1000)
        - Periodicamente (es. ogni 60 secondi)
        
        Args:
            node_name: Nome del nodo
            new_vectors: Array di nuovi vettori [N, dimension]
            
        Returns:
            True se update riuscito
        """
        if node_name not in self.node_centroids:
            print(f"⚠️  Nodo {node_name} non trovato in meta-HNSW")
            return False
        
        if len(new_vectors) == 0:
            return True
        
        # Update running statistics
        old_count = self.node_vector_counts[node_name]
        old_sum = self.node_vector_sums[node_name]
        
        new_count = old_count + len(new_vectors)
        new_sum = old_sum + np.sum(new_vectors, axis=0)
        
        # Calcola nuovo centroide
        new_centroid = new_sum / new_count
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
        
        # Update stored data
        self.node_centroids[node_name] = new_centroid
        self.node_vector_counts[node_name] = new_count
        self.node_vector_sums[node_name] = new_sum
        
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Increment batch counter e controlla se serve rebuild
        # ═══════════════════════════════════════════════════════════════
        self.updates_since_rebuild += 1
        
        # Controlla se è tempo di rebuild
        should_rebuild, reason = self._should_rebuild()
        if should_rebuild:
            print(f"  📌 Triggering HNSW rebuild: {reason}")
            self._rebuild_hnsw_index()
        
        return True
    
    def recalculate_centroid_from_scratch(self, node_name: str, vectors: np.ndarray, method: str = 'mean') -> bool:
        """
        Ricalcola centroide completamente da zero (fallback se running average ha troppo drift).
        
        QUANDO USARE:
        - Periodicamente (es. ogni 100K inserimenti) per correggere drift
        - Se distribuzione vettori cambia radicalmente
        
        COSTO:
        - Richiede fetch di tutti i vettori dal nodo
        - O(N) dove N = numero vettori nel nodo
        - Più costoso ma più preciso
        
        Args:
            node_name: Nome del nodo
            vectors: Tutti i vettori del nodo (fetch da Qdrant)
            method: Metodo calcolo ('mean', 'median', 'weighted')
            
        Returns:
            True se update riuscito
        """
        if node_name not in self.node_centroids:
            print(f"⚠️  Nodo {node_name} non trovato in meta-HNSW")
            return False
        
        if len(vectors) == 0:
            return False
        
        # Calcola centroide da zero (stesso codice di add_node_centroid)
        if method == 'mean':
            new_centroid = np.mean(vectors, axis=0)
        elif method == 'median':
            new_centroid = np.median(vectors, axis=0)
        elif method == 'weighted':
            from scipy.spatial.distance import pdist, squareform
            
            if len(vectors) > 1000:
                sample_indices = np.random.choice(len(vectors), 1000, replace=False)
                sample = vectors[sample_indices]
            else:
                sample = vectors
            
            distances = squareform(pdist(sample, metric='euclidean'))
            avg_distances = distances.mean(axis=1)
            weights = 1.0 / (avg_distances + 1e-6)
            weights /= weights.sum()
            new_centroid = np.average(sample, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalizza
        new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
        
        # Reset statistiche
        self.node_centroids[node_name] = new_centroid
        self.node_vector_counts[node_name] = len(vectors)
        self.node_vector_sums[node_name] = np.sum(vectors, axis=0)
        
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Forza rebuild (ricalcolo da scratch = cambio significativo)
        # ═══════════════════════════════════════════════════════════════
        self.needs_rebuild = True
        
        print(f"✓ Centroide {node_name} ricalcolato da {len(vectors)} vettori")
        
        # Rebuild immediato per ricalcolo da scratch
        self._rebuild_hnsw_index()
        
        return True
    
    def find_nearest_nodes(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Trova i k nodi più vicini al vettore query.
        
        MODIFICATO: Lazy rebuild prima di query.
        
        Args:
            query_vector: Vettore query [dimension]
            k: Numero di nodi da restituire
            
        Returns:
            Lista di (node_name, distance) ordinata per vicinanza
            
        Example:
            query = embedding("machine learning")
            nodes = meta_hnsw.find_nearest_nodes(query, k=3)
            # [("node-1", 0.12), ("node-3", 0.18), ("node-5", 0.25)]
        """
        if len(self.node_names) == 0:
            raise ValueError("Meta-HNSW vuoto! Aggiungi nodi prima di query.")
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Lazy rebuild prima di query (se necessario)
        # ═══════════════════════════════════════════════════════════════
        should_rebuild, reason = self._should_rebuild()
        if should_rebuild:
            print(f"  📌 Lazy rebuild before query: {reason}")
            self._rebuild_hnsw_index()
        
        # Normalizza query
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Limita k
        k = min(k, len(self.node_names))
        
        # Query HNSW
        indices, distances = self.hnsw_index.knn_query(query_norm.reshape(1, -1), k=k)
        
        # Mappa indici → nomi nodi
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            node_name = self.node_names[idx]
            results.append((node_name, float(dist)))
        
        return results
    
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
    
    def save(self, path: str):
        """Salva meta-HNSW su disco."""
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Rebuild finale prima di salvare (garantisce coerenza)
        # ═══════════════════════════════════════════════════════════════
        if self.needs_rebuild or self.updates_since_rebuild > 0:
            print("  🔧 Final rebuild before save...")
            self._rebuild_hnsw_index()
        
        data = {
            'dimension': self.dimension,
            'max_nodes': self.max_nodes,
            'node_names': self.node_names,
            'node_centroids': self.node_centroids,
            'node_vector_counts': self.node_vector_counts,
            'node_vector_sums': self.node_vector_sums,
            'rebuild_drift_threshold': self.rebuild_drift_threshold,
            'rebuild_every_n_updates': self.rebuild_every_n_updates,
            'previous_centroids': self.previous_centroids
        }
        
        # Salva indice HNSW
        hnsw_path = path + '.hnsw'
        self.hnsw_index.save_index(hnsw_path)
        
        # Salva metadata
        joblib.dump(data, path)
        
        print(f"✓ Meta-HNSW salvato: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MetaHNSW':
        """Carica meta-HNSW da disco."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Meta-HNSW non trovato: {path}")
        
        # Carica metadata
        data = joblib.load(path)
        
        # Crea oggetto
        meta_hnsw = cls(
            dimension=data['dimension'],
            max_nodes=data['max_nodes']
        )
        
        # Carica indice HNSW
        hnsw_path = path + '.hnsw'
        meta_hnsw.hnsw_index.load_index(hnsw_path, max_elements=data['max_nodes'])
        
        # Ripristina mapping
        meta_hnsw.node_names = data['node_names']
        meta_hnsw.node_centroids = data['node_centroids']
        meta_hnsw.node_vector_counts = data.get('node_vector_counts', {})
        meta_hnsw.node_vector_sums = data.get('node_vector_sums', {})
        meta_hnsw.rebuild_drift_threshold = data.get('rebuild_drift_threshold', 0.1)
        meta_hnsw.rebuild_every_n_updates = data.get('rebuild_every_n_updates', 10)
        meta_hnsw.previous_centroids = data.get('previous_centroids', {})
        
        # Reset contatori (appena caricato, nessun update pending)
        meta_hnsw.needs_rebuild = False
        meta_hnsw.updates_since_rebuild = 0
        
        print(f"✓ Meta-HNSW caricato: {path} ({len(meta_hnsw.node_names)} nodi)")
        
        return meta_hnsw
    
    def get_statistics(self) -> dict:
        """Restituisce statistiche meta-HNSW."""
        if not self.node_centroids:
            return {}
        
        centroids_array = np.array(list(self.node_centroids.values()))
        
        from scipy.spatial.distance import pdist, squareform
        centroid_distances = squareform(pdist(centroids_array, metric='cosine'))
        non_zero = centroid_distances[centroid_distances > 0]
        
        # Calcola drift attuale
        drift = self._calculate_centroid_drift()
        max_drift = max(drift.values()) if drift else 0.0
        
        return {
            'num_nodes': len(self.node_names),
            'dimension': self.dimension,
            'min_distance': float(non_zero.min()) if len(non_zero) > 0 else 0,
            'max_distance': float(non_zero.max()) if len(non_zero) > 0 else 0,
            'mean_distance': float(non_zero.mean()) if len(non_zero) > 0 else 0,
            'nodes': self.node_names,
            'total_vectors_tracked': sum(self.node_vector_counts.values()),
            'vectors_per_node': self.node_vector_counts,
            # Nuove statistiche rebuild
            'needs_rebuild': self.needs_rebuild,
            'updates_since_rebuild': self.updates_since_rebuild,
            'max_centroid_drift': max_drift,
            'rebuild_threshold': self.rebuild_drift_threshold
        }
