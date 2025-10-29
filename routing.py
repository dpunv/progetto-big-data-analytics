# routing.py
import threading
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

class RoutingTable:
    """
    Gestisce la mappatura tra cluster_id e nodi Qdrant.
    È thread-safe per gestire aggiornamenti concorrenti.
    
    NUOVO APPROCCIO:
    - Ordina cluster per similarità semantica (centroidi vicini nello spazio)
    - Assegna cluster contigui allo stesso nodo
    - Preserva località semantica senza LSH
    """
    def __init__(self, node_names: list[str]):
        """
        Inizializza la tabella di routing.
        
        Args:
            node_names: Lista dei nomi dei nodi Qdrant disponibili
        """
        self._node_names = node_names
        self._routing_map = {}
        self._lock = threading.Lock()
        self._cluster_order = []  # NUOVO: ordine semantico dei cluster

    def assign_initial_clusters(self, n_clusters: int):
        """
        DEPRECATO: Usare assign_clusters_by_semantic_similarity invece.
        """
        raise NotImplementedError("Use assign_clusters_by_semantic_similarity instead")

    def get_nodes(self, cluster_id: str) -> list[str]:
        """
        Ottiene i nodi target per un dato cluster_id.
        
        SCOPO:
        - Dato un cluster_id, restituisce su quale/i nodo/i si trova
        - Supporta sia cluster originali ("5") che sub-cluster dopo split ("5.0", "5.1")
        
        COME FUNZIONA:
        1. Prima cerca una corrispondenza esatta (es. "5.0" → ["node-2"])
        2. Se non trova, cerca il cluster padre (es. "5.0" cerca "5")
        3. Se non trova nemmeno quello, errore
        
        PERCHÉ QUESTA LOGICA:
        - Prima dello split: cluster "5" → ["node-1"]
        - Dopo lo split: 
          - "5.0" → ["node-2"]
          - "5.1" → ["node-3"]
          - "5" non esiste più
        - Durante la migrazione: potremmo avere dati vecchi che usano "5"
        
        USO PRATICO:
        - Per inserimento: "Dove metto questo vettore del cluster 3?" → "node-1"
        - Per query: "Dove cerco vettori del cluster 7.1?" → "node-3"
        
        Args:
            cluster_id: ID del cluster (es. "5", "5.0", "5.1")
            
        Returns:
            Lista di nodi contenenti quel cluster (di solito 1 elemento)
            
        Raises:
            KeyError: Se il cluster_id non esiste nella tabella
        """
        with self._lock:
            # cluster_id potrebbe essere "5" o "5.0", "5.1" dopo uno split.
            # Cerchiamo la corrispondenza più specifica.
            if cluster_id in self._routing_map:
                return self._routing_map[cluster_id]
            
            # Se è un sub-cluster, controlliamo la radice (es. "5" per "5.0")
            # Utile durante transizioni o per compatibilità
            base_cluster = cluster_id.split('.')[0]
            if base_cluster in self._routing_map:
                return self._routing_map[base_cluster]
                
            raise KeyError(f"Cluster ID {cluster_id} not found in routing table.")

    def split_cluster_mapping(self, old_cluster_id: int, new_sub_cluster_ids: list[str], node_assignments: dict[str, str]):
        """
        Aggiorna la tabella di routing per riflettere uno split.
        Questa operazione è atomica dal punto di vista logico.
        
        SCOPO:
        - Quando un cluster viene splittato in sub-cluster, aggiorna la mappa
        - Rimuove il cluster "padre" originale
        - Aggiunge i nuovi sub-cluster con le loro destinazioni
        
        COME FUNZIONA:
        PRIMA dello split:
        - routing_map = {"5": ["node-1"], "6": ["node-2"], ...}
        
        CHIAMATA:
        - split_cluster_mapping(5, ["5.0", "5.1", "5.2"], 
                                {"5.0": "node-1", "5.1": "node-2", "5.2": "node-3"})
        
        DOPO lo split:
        - routing_map = {"5.0": ["node-1"], "5.1": ["node-2"], "5.2": ["node-3"], "6": ["node-2"], ...}
        
        PERCHÉ È IMPORTANTE:
        - Senza questo aggiornamento, il sistema continuerebbe a cercare "cluster 5" che non esiste più
        - Garantisce che nuove query/inserimenti vadano sui nodi corretti
        - È atomico (con lock) per evitare stati inconsistenti durante aggiornamenti
        
        ESEMPIO PRATICO:
        - Cluster 5 ha 10K vettori (hotspot!) su node-1
        - Lo splittiamo in 3 parti:
          - 5.0 (3K vettori) → node-1 (resta lì)
          - 5.1 (3.5K vettori) → node-2 (migrato)
          - 5.2 (3.5K vettori) → node-3 (migrato)
        - Questo metodo aggiorna la routing table per riflettere la nuova struttura
        
        Args:
            old_cluster_id: ID del cluster originale da splittare (es. 5)
            new_sub_cluster_ids: Lista di ID dei nuovi sub-cluster (es. ["5.0", "5.1", "5.2"])
            node_assignments: Mappa sub_cluster_id → nodo destinazione
        """
        with self._lock:  # Operazione atomica
            print(f"Updating routing table: Splitting cluster {old_cluster_id}...")
            
            # Rimuoviamo la vecchia mappatura del cluster padre
            # Es. rimuove "5": ["node-1"]
            if str(old_cluster_id) in self._routing_map:
                del self._routing_map[str(old_cluster_id)]

            # Aggiungiamo le nuove mappature per i sub-cluster
            # Es. aggiungiamo "5.0": ["node-1"], "5.1": ["node-2"], "5.2": ["node-3"]
            for sub_id in new_sub_cluster_ids:
                node = node_assignments[sub_id]
                self._routing_map[sub_id] = [node]
                print(f"  -> Sub-cluster {sub_id} assigned to {node}")
        
        print(f"Routing table updated for split of cluster {old_cluster_id}.")

    def assign_clusters_by_semantic_similarity(self, n_clusters: int, centroids: np.ndarray):
        """
        Assegna cluster ai nodi raggruppando cluster semanticamente simili.
        
        STRATEGIA NUOVA:
        1. Calcola matrice di distanze tra tutti i centroidi
        2. Ordina cluster per similarità usando Minimum Spanning Tree (MST)
        3. Divide la sequenza ordinata in N parti uguali (N = numero nodi)
        4. Assegna ogni parte contigua a un nodo diverso
        
        PERCHÉ QUESTA STRATEGIA:
        - Cluster contigui sono semanticamente vicini
        - Ogni nodo diventa "esperto" di un'area semantica
        - Query su topic simili → singolo nodo (no broadcast)
        
        ESEMPIO CON 9 CLUSTER E 3 NODI:
        
        Step 1 - Centroidi nello spazio:
        ```
        Cluster 0: [tech, AI]
        Cluster 1: [tech, ML]
        Cluster 2: [tech, data]
        Cluster 3: [sport, football]
        Cluster 4: [sport, basketball]
        Cluster 5: [sport, tennis]
        Cluster 6: [food, italian]
        Cluster 7: [food, asian]
        Cluster 8: [food, desserts]
        ```
        
        Step 2 - Matrice distanze (esempio semplificato):
        ```
        Distanze euclidee:
        0-1: 0.3 (molto simili, entrambi AI)
        0-2: 0.5 (simili, tech)
        0-3: 2.5 (distanti, tech vs sport)
        1-2: 0.4
        3-4: 0.6 (simili, entrambi sport)
        6-7: 0.7 (simili, entrambi food)
        ...
        ```
        
        Step 3 - MST per ordinare:
        ```
        MST trova il percorso che connette tutti i cluster minimizzando distanze:
        0 → 1 (0.3) → 2 (0.4) → 3 (2.5) → 4 (0.6) → 5 (0.8) → 6 (3.0) → 7 (0.7) → 8 (0.9)
        
        Ordine risultante: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        ```
        
        Step 4 - Divisione equa:
        ```
        9 cluster / 3 nodi = 3 cluster per nodo
        
        node-1: cluster [0, 1, 2]      ← area "tech"
        node-2: cluster [3, 4, 5]      ← area "sport"
        node-3: cluster [6, 7, 8]      ← area "food"
        ```
        
        BENEFICI:
        - Query "AI technology" → predice cluster 0 o 1 → cerca SOLO su node-1
        - Query "basketball stats" → predice cluster 4 → cerca SOLO su node-2
        - No broadcast, no overhead multi-nodo
        
        ALGORITMO DETTAGLIATO:
        
        1. CALCOLO DISTANZE:
        ```python
        distances[i][j] = ||centroid_i - centroid_j||₂
        ```
        
        2. MINIMUM SPANNING TREE:
        - Algoritmo: Kruskal o Prim
        - Input: grafo completo pesato (pesi = distanze)
        - Output: albero che connette tutti i nodi con peso minimo totale
        - Proprietà: percorre cluster in ordine di similarità
        
        3. DFS SUL MST:
        - Traversal depth-first dall'origine
        - Visita nodi in ordine che rispetta vicinanza
        - Risultato: lista ordinata di cluster ID
        
        4. PARTIZIONAMENTO:
        ```
        n_per_node = n_clusters / n_nodes
        node-1 riceve cluster [0 : n_per_node]
        node-2 riceve cluster [n_per_node : 2*n_per_node]
        ...
        ```
        
        Args:
            n_clusters: Numero totale di cluster K-means
            centroids: Array [n_clusters, dimensione] con centroidi
        """
        with self._lock:
            n_nodes = len(self._node_names)
            
            print(f"\n{'='*70}")
            print(f"SEMANTIC CLUSTERING ASSIGNMENT")
            print(f"{'='*70}")
            print(f"Total clusters: {n_clusters}")
            print(f"Available nodes: {n_nodes}")
            print(f"Strategy: Group semantically similar clusters on same node\n")
            
            # --- STEP 1: CALCOLA MATRICE DISTANZE ---
            print("Step 1: Computing pairwise centroid distances...")
            
            # Calcola distanze euclidee tra tutti i centroidi
            # pdist restituisce condensed distance matrix (vettore)
            distances_condensed = pdist(centroids, metric='euclidean')
            # squareform converte in matrice NxN simmetrica
            distance_matrix = squareform(distances_condensed)
            
            print(f"  Distance matrix shape: {distance_matrix.shape}")
            print(f"  Min distance: {np.min(distance_matrix[distance_matrix > 0]):.4f}")
            print(f"  Max distance: {np.max(distance_matrix):.4f}")
            print(f"  Mean distance: {np.mean(distance_matrix[distance_matrix > 0]):.4f}\n")
            
            # --- STEP 2: COSTRUISCI MINIMUM SPANNING TREE ---
            print("Step 2: Building Minimum Spanning Tree (MST)...")
            
            # MST trova l'albero che connette tutti i cluster minimizzando distanze totali
            mst = minimum_spanning_tree(distance_matrix)
            mst_array = mst.toarray()
            
            print(f"  MST edges: {np.count_nonzero(mst_array)}")
            print(f"  Total MST weight: {mst_array.sum():.4f}\n")
            
            # --- STEP 3: ORDINA CLUSTER CON DFS SUL MST ---
            print("Step 3: Ordering clusters via MST traversal...")
            
            cluster_order = self._dfs_mst_traversal(mst_array, n_clusters)
            self._cluster_order = cluster_order
            
            print(f"  Cluster order: {cluster_order}\n")
            
            # Stampa alcune distanze consecutive per verificare
            print("  Verification - Distances between consecutive clusters:")
            for i in range(min(5, len(cluster_order) - 1)):
                c1, c2 = cluster_order[i], cluster_order[i+1]
                dist = distance_matrix[c1, c2]
                print(f"    Cluster {c1} → {c2}: distance = {dist:.4f}")
            print()
            
            # --- STEP 4: DIVIDI EQUAMENTE TRA NODI ---
            print("Step 4: Partitioning clusters across nodes...")
            
            # Calcola quanti cluster per nodo
            clusters_per_node = n_clusters // n_nodes
            remainder = n_clusters % n_nodes
            
            print(f"  Base clusters per node: {clusters_per_node}")
            if remainder > 0:
                print(f"  Extra clusters to distribute: {remainder}\n")
            
            # Assegna cluster ai nodi
            current_idx = 0
            
            for node_idx, node_name in enumerate(self._node_names):
                # Primi nodi ricevono un cluster extra se c'è remainder
                n_clusters_for_node = clusters_per_node + (1 if node_idx < remainder else 0)
                
                # Prendi slice contigua dalla sequenza ordinata
                node_clusters = cluster_order[current_idx : current_idx + n_clusters_for_node]
                
                # Assegna alla routing map
                for cluster_id in node_clusters:
                    self._routing_map[str(cluster_id)] = [node_name]
                
                print(f"  {node_name}: {len(node_clusters)} clusters {node_clusters}")
                
                # Calcola centroide medio di questo nodo per vedere "tema" semantico
                node_centroid = np.mean(centroids[node_clusters], axis=0)
                avg_intra_distance = np.mean([
                    distance_matrix[node_clusters[i], node_clusters[j]]
                    for i in range(len(node_clusters))
                    for j in range(i+1, len(node_clusters))
                ]) if len(node_clusters) > 1 else 0
                
                print(f"    → Average intra-node distance: {avg_intra_distance:.4f}")
                
                current_idx += n_clusters_for_node
            
            print(f"\n{'='*70}")
            print("✅ Semantic assignment complete!")
            print(f"{'='*70}\n")

    def _dfs_mst_traversal(self, mst_array: np.ndarray, n_clusters: int) -> list[int]:
        """
        Attraversa il MST con DFS per ottenere ordine cluster semanticamente coerente.
        
        PERCHÉ DFS:
        - DFS visita l'albero in profondità
        - Garantisce che cluster vicini nel MST siano vicini nella sequenza
        - Preserva "percorso" semantico
        
        ESEMPIO MST:
        ```
              0
             / \
            1   2
           /     \
          3       4
        ```
        
        DFS traversal: [0, 1, 3, 2, 4]
        - Visita rami completamente prima di cambiare
        - Cluster vicini nell'albero → vicini nella lista
        
        Args:
            mst_array: Matrice NxN del MST (simmetrica, sparse)
            n_clusters: Numero totale di cluster
            
        Returns:
            Lista ordinata di cluster ID
        """
        visited = set()
        order = []
        
        def dfs(node):
            """Recursive DFS helper."""
            visited.add(node)
            order.append(node)
            
            # Trova vicini nel MST
            neighbors = np.where(mst_array[node] > 0)[0].tolist()
            # Aggiungi anche vicini nella direzione opposta (MST è simmetrico)
            neighbors += np.where(mst_array[:, node] > 0)[0].tolist()
            neighbors = list(set(neighbors))  # Rimuovi duplicati
            
            # Ordina vicini per distanza (visita prima i più vicini)
            neighbors.sort(key=lambda n: mst_array[node, n] + mst_array[n, node])
            
            # Visita vicini non ancora visitati
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor)
        
        # Inizia DFS dal cluster 0 (arbitrario, ma consistente)
        dfs(0)
        
        # Se ci sono cluster disconnessi (non dovrebbe succedere con MST), aggiungili
        for cluster_id in range(n_clusters):
            if cluster_id not in visited:
                dfs(cluster_id)
        
        return order

    def assign_initial_clusters_lsh(self, n_clusters: int, centroids: np.ndarray):
        """
        DEPRECATO: LSH rimosso, usare assign_clusters_by_semantic_similarity.
        """
        raise NotImplementedError(
            "LSH assignment removed. Use assign_clusters_by_semantic_similarity instead."
        )