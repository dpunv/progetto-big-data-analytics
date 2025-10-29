# routing.py
import threading
import numpy as np

class RoutingTable:
    """
    Gestisce la mappatura tra cluster_id e nodi Qdrant.
    È thread-safe per gestire aggiornamenti concorrenti.
    
    SCOPO GENERALE:
    - Mantiene la "mappa" che dice: "Il cluster X si trova sul nodo Y"
    - Permette di sapere dove cercare/inserire i dati per ogni cluster
    - Gestisce dinamicamente gli split dei cluster (quando vengono divisi)
    
    PERCHÉ SERVE:
    - È il "cervello" del routing: senza questa mappa non sapremmo dove sono i dati
    - Thread-safe: più processi possono leggere/aggiornare simultaneamente
    - Supporta cluster gerarchici (es. cluster "5" diventa "5.0", "5.1", "5.2")
    """
    def __init__(self, node_names: list[str]):
        """
        Inizializza la tabella di routing.
        
        COSA FA:
        - Memorizza i nomi dei nodi Qdrant disponibili (es. ["node-1", "node-2", "node-3"])
        - Crea una mappa vuota cluster→nodi
        - Inizializza un lock per thread-safety
        
        Args:
            node_names: Lista dei nomi dei nodi Qdrant disponibili
        """
        self._node_names = node_names  # Es. ["node-1", "node-2", "node-3"]
        self._routing_map = {}  # Mappa: cluster_id → [lista di nodi]
        self._lock = threading.Lock()  # Per operazioni atomiche thread-safe

    def assign_initial_clusters(self, n_clusters: int):
        """
        Assegna i cluster iniziali ai nodi in modo round-robin.
        
        SCOPO:
        - Distribuisce equamente i cluster iniziali sui nodi disponibili
        - È il setup iniziale del sistema prima di qualsiasi split
        
        COME FUNZIONA (ROUND-ROBIN):
        - Cluster 0 → node-1
        - Cluster 1 → node-2
        - Cluster 2 → node-3
        - Cluster 3 → node-1 (ricomincia)
        - Cluster 4 → node-2
        - etc.
        
        PERCHÉ ROUND-ROBIN:
        - Distribuzione bilanciata iniziale
        - Semplice e deterministico
        - Ogni nodo riceve circa lo stesso numero di cluster
        
        ESEMPIO:
        - Se hai 10 cluster e 3 nodi:
        - node-1: cluster 0, 3, 6, 9
        - node-2: cluster 1, 4, 7
        - node-3: cluster 2, 5, 8
        
        Args:
            n_clusters: Numero totale di cluster iniziali (dal K-means)
        """
        with self._lock:  # Lock per thread-safety
            for i in range(n_clusters):
                # Operatore modulo (%) per ciclare sui nodi: i % 3 → 0,1,2,0,1,2,...
                node = self._node_names[i % len(self._node_names)]
                
                # Mappa a una lista per supportare futuri split
                # Es. {"0": ["node-1"], "1": ["node-2"], ...}
                self._routing_map[str(i)] = [node]
            print("Initial routing table created.")

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

    def assign_initial_clusters_lsh(self, n_clusters: int, centroids: np.ndarray):
        """
        Assegna i cluster iniziali ai nodi usando Locality-Sensitive Hashing (LSH).
        
        SCOPO:
        - Distribuisce cluster PRESERVANDO la località semantica
        - Cluster semanticamente vicini vanno sullo stesso nodo
        - Migliore del round-robin per query semantiche
        
        COME FUNZIONA (LSH con Random Projection):
        1. Genera vettori casuali di proiezione (hyperplanes)
        2. Per ogni centroide, calcola su quale lato degli hyperplanes cade
        3. Questo genera un "hash" binario che preserva similarità
        4. Centroidi con hash simili → stesso nodo
        
        ESEMPIO MATEMATICO:
        - Hyperplane: vettore random [0.3, -0.7, 0.5, ...]
        - Centroide A: [1.2, 0.3, -0.5, ...] → dot product = 0.8 > 0 → bit 1
        - Centroide B: [1.1, 0.4, -0.6, ...] → dot product = 0.7 > 0 → bit 1 (stesso!)
        - Centroide C: [-0.5, 1.2, 0.3, ...] → dot product = -0.9 < 0 → bit 0 (diverso!)
        - A e B hanno hash simili → stesso nodo
        
        PERCHÉ LSH È MEGLIO:
        Round-robin:
        - node-1: cluster 0, 3, 6, 9 (semanticamente casuali)
        - Query su topic "tech" → cerca su tutti i nodi
        
        LSH:
        - node-1: cluster 0, 1, 2 (tutti topic "tech")
        - node-2: cluster 3, 4, 5 (tutti topic "lifestyle")
        - node-3: cluster 6, 7, 8, 9 (tutti topic "science")
        - Query su topic "tech" → cerca SOLO su node-1!
        
        ALGORITMO:
        1. Genera log2(n_nodi) hyperplanes random
        2. Per ogni centroide: calcola hash binario
        3. Hash modulo n_nodi → assegna nodo
        4. Centroidi vicini nello spazio → hash simili → stesso nodo
        
        Args:
            n_clusters: Numero totale di cluster iniziali
            centroids: Array numpy [n_clusters, dimensione] con i centroidi K-means
        """
        with self._lock:
            n_nodes = len(self._node_names)
            d = centroids.shape[1]  # Dimensione vettori
            
            # Numero di hyperplanes: log2(n_nodi) arrotondato per eccesso
            # Con 3 nodi: ceil(log2(3)) = 2 hyperplanes
            n_hyperplanes = int(np.ceil(np.log2(n_nodes)))
            
            print(f"LSH: Using {n_hyperplanes} random hyperplanes for {n_nodes} nodes")
            
            # Genera hyperplanes random (vettori normali unitari)
            np.random.seed(42)  # Seed fisso per riproducibilità
            hyperplanes = np.random.randn(n_hyperplanes, d)
            # Normalizza a vettori unitari
            hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1, keepdims=True)
            
            # Calcola hash LSH per ogni centroide
            for i in range(n_clusters):
                centroid = centroids[i]
                
                # Proietta il centroide su ogni hyperplane
                # Se dot product > 0 → bit 1, altrimenti bit 0
                projections = np.dot(hyperplanes, centroid)
                hash_bits = (projections > 0).astype(int)
                
                # Converti hash binario in intero
                # Es. [1, 0, 1] → 5 in decimale
                hash_value = int(''.join(map(str, hash_bits)), 2)
                
                # Assegna nodo: hash modulo numero_nodi
                # Questo garantisce distribuzione, mantenendo località
                node_idx = hash_value % n_nodes
                node = self._node_names[node_idx]
                
                self._routing_map[str(i)] = [node]
                print(f"  Cluster {i}: hash={hash_bits} ({hash_value}) → {node}")
            
            print("\nLSH routing table created with semantic locality preservation.")
            
            # Stampa distribuzione per verificare bilanciamento
            node_counts = {}
            for node_name in self._node_names:
                count = sum(1 for nodes in self._routing_map.values() if nodes[0] == node_name)
                node_counts[node_name] = count
            
            print("\nCluster distribution per node:")
            for node, count in node_counts.items():
                cluster_ids = [cid for cid, nodes in self._routing_map.items() if nodes[0] == node]
                print(f"  {node}: {count} clusters {sorted(cluster_ids)}")