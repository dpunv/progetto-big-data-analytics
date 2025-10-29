# rebalancer.py
"""
Modulo per il monitoraggio e rebalancing automatico dei cluster.

SCOPO GENERALE:
- Monitora il carico sui cluster (quanti vettori per cluster)
- Rileva hotspot (cluster sovraccarichi)
- Splitta cluster caldi in sub-cluster più piccoli
- Migra dati tra nodi per bilanciare il carico

COMPONENTI:
- Rebalancer: Classe principale che orchestra monitor + rebalancing
"""

import numpy as np
import faiss
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScrollRequest
from typing import Dict, List
from tqdm import tqdm

from config import *
from routing import RoutingTable


class Rebalancer:
    """
    Gestisce il monitoring e il rebalancing dei cluster distribuiti.
    
    RESPONSABILITÀ:
    1. Monitor: Conta quanti vettori ci sono in ogni cluster
    2. Detect: Identifica cluster che superano la soglia
    3. Split: Usa K-means per dividere cluster caldi in sub-cluster
    4. Migrate: Sposta dati da nodo sovraccarico ad altri nodi
    5. Update: Aggiorna la routing table dopo lo split
    
    PERCHÉ SERVE:
    - Previene hotspot permanenti che degradano performance
    - Mantiene distribuzione bilanciata tra nodi
    - Adatta dinamicamente alla distribuzione reale dei dati
    """
    
    def __init__(self, clients: Dict[str, QdrantClient], routing_table: RoutingTable):
        """
        Inizializza il Rebalancer.
        
        Args:
            clients: Dizionario {nome_nodo: QdrantClient} per comunicare con ogni nodo
            routing_table: Riferimento alla routing table per aggiornamenti
        """
        self.clients = clients
        self.routing_table = routing_table
    
    def monitor_clusters(self) -> Dict[str, int]:
        """
        Conta quanti vettori ci sono in ogni cluster su tutti i nodi.
        
        SCOPO:
        - Fornisce visibilità sulla distribuzione dei dati
        - Identifica cluster sbilanciati
        
        COME FUNZIONA:
        1. Per ogni nodo Qdrant:
           - Scorre tutti i punti nella collection
           - Legge il metadata "cluster_id"
           - Incrementa il contatore per quel cluster
        2. Aggrega i conteggi da tutti i nodi
        
        ESEMPIO OUTPUT:
        {
            "0": 1500,
            "1": 1600,
            "2": 1550,
            "3": 1480,
            "4": 1520,
            "5": 35000,  # <-- HOTSPOT!
            "6": 1490,
            "7": 1510,
            "8": 1540,
            "9": 1510
        }
        
        PERCHÉ SCROLL:
        - Qdrant scroll permette di iterare su tutti i punti
        - Efficiente anche con milioni di vettori
        - Paginato: non carica tutto in memoria
        
        Returns:
            Dizionario {cluster_id: numero_vettori}
        """
        cluster_counts = {}
        
        print("Monitoring cluster distribution across nodes...")
        
        # Itera su ogni nodo
        for node_name, client in self.clients.items():
            print(f"  Scanning {node_name}...")
            
            # Scroll attraverso tutti i punti nel nodo
            # offset=None inizia dall'inizio, limit=100 prende 100 punti alla volta
            offset = None
            while True:
                # Recupera batch di punti
                records, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,  # Batch size
                    offset=offset,
                    with_payload=True,  # Include metadata
                    with_vectors=False  # Non serve il vettore, solo metadata
                )
                
                # Se nessun record, abbiamo finito
                if not records:
                    break
                
                # Conta vettori per cluster
                for record in records:
                    cluster_id = record.payload.get("cluster_id", "unknown")
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                
                # Prossima pagina
                offset = next_offset
                if offset is None:
                    break
        
        # Stampa statistiche
        print("\nCluster Distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            status = "⚠️ HOTSPOT" if count > REBALANCER_THRESHOLD else "✓"
            print(f"  Cluster {cluster_id}: {count:,} vectors {status}")
        
        return cluster_counts
    
    def get_node_loads(self) -> Dict[str, int]:
        """
        Calcola il carico (numero di vettori) per ogni nodo.
        
        SCOPO:
        - Fornisce visibilità sul carico totale di ogni nodo
        - Usato per decisioni di assegnazione load-aware
        
        COME FUNZIONA:
        - Conta tutti i punti in ogni nodo (indipendentemente dal cluster)
        - Usa count API di Qdrant (molto veloce)
        
        PERCHÉ È IMPORTANTE:
        - Durante split, dobbiamo sapere quali nodi sono sotto-caricati
        - Evita di migrare su nodi già sovraccarichi
        
        Returns:
            Dizionario {nome_nodo: numero_vettori_totali}
            
        ESEMPIO OUTPUT:
        {
            "node-1": 0,
            "node-2": 50000,  # Sovraccarico!
            "node-3": 0
        }
        """
        node_loads = {}
        
        print("Calculating node loads...")
        for node_name, client in self.clients.items():
            try:
                # Usa count per ottenere numero totale punti nel nodo
                count_result = client.count(
                    collection_name=COLLECTION_NAME,
                    exact=True  # Count esatto (non stima)
                )
                node_loads[node_name] = count_result.count
                print(f"  {node_name}: {count_result.count:,} vectors")
            except Exception as e:
                print(f"  Warning: Could not get count for {node_name}: {e}")
                node_loads[node_name] = 0
        
        return node_loads
    
    def is_node_balanced(self, tolerance: float = 0.3) -> tuple[bool, dict[str, int]]:
        """
        Verifica se i nodi sono bilanciati in termini di carico totale.
        
        SCOPO:
        - Prima di fare rebalancing cluster-level, verifica se serve
        - Se nodi già bilanciati → skip rebalancing (non sprecare risorse)
        - Se nodi sbilanciati → procedi con split/migrazione
        
        COME FUNZIONA:
        - Calcola carico totale per ogni nodo
        - Calcola media e deviazione
        - Se nodo > media × (1 + tolerance) → sbilanciato
        
        ESEMPIO:
        Scenario 1 (BILANCIATO):
        - node-1: 16,500 vettori
        - node-2: 17,000 vettori  
        - node-3: 16,500 vettori
        - Media: 16,666
        - Max deviazione: 2% < 30% tolerance
        - Risultato: BILANCIATO ✅
        
        Scenario 2 (SBILANCIATO):
        - node-1: 0 vettori
        - node-2: 50,000 vettori ← 3x la media!
        - node-3: 0 vettori
        - Media: 16,666
        - Max deviazione: 200% > 30% tolerance
        - Risultato: SBILANCIATO ⚠️
        
        Args:
            tolerance: Tolleranza percentuale (0.3 = ±30% dalla media)
            
        Returns:
            (is_balanced, node_loads): 
                - is_balanced: True se sistema bilanciato
                - node_loads: Dizionario {nodo: carico}
        """
        node_loads = self.get_node_loads()
        
        # Calcola statistiche
        total_vectors = sum(node_loads.values())
        n_nodes = len(node_loads)
        avg_load = total_vectors / n_nodes if n_nodes > 0 else 0
        
        # Definisci range accettabile
        min_acceptable = avg_load * (1 - tolerance)
        max_acceptable = avg_load * (1 + tolerance)
        
        print(f"\n📊 Node Balance Check:")
        print(f"  Total vectors: {total_vectors:,}")
        print(f"  Average load per node: {avg_load:,.0f}")
        print(f"  Acceptable range: {min_acceptable:,.0f} - {max_acceptable:,.0f} (±{tolerance*100:.0f}%)")
        
        # Verifica ogni nodo
        is_balanced = True
        overloaded_nodes = []
        underloaded_nodes = []
        
        for node_name, load in node_loads.items():
            deviation_pct = ((load - avg_load) / avg_load * 100) if avg_load > 0 else 0
            
            if load > max_acceptable:
                status = "⚠️ OVERLOADED"
                is_balanced = False
                overloaded_nodes.append(node_name)
            elif load < min_acceptable:
                status = "📉 UNDERLOADED"
                underloaded_nodes.append(node_name)
            else:
                status = "✅ OK"
            
            print(f"  {node_name}: {load:,} vectors ({deviation_pct:+.1f}%) {status}")
        
        if is_balanced:
            print(f"\n✅ System is BALANCED (all nodes within ±{tolerance*100:.0f}% of average)")
        else:
            print(f"\n⚠️ System is UNBALANCED")
            print(f"  Overloaded nodes: {overloaded_nodes}")
            print(f"  Underloaded nodes: {underloaded_nodes}")
        
        return is_balanced, node_loads
    
    def identify_rebalancing_strategy(self, overloaded_node: str, node_loads: dict[str, int]) -> tuple[str, dict]:
        """
        Decide la strategia di rebalancing per un nodo sovraccarico.
        
        NUOVA STRATEGIA A TRE LIVELLI:
        1. Se nodo ha MULTIPLI cluster → MIGRATE cluster marginali (priorità)
        2. Se solo pochi cluster ma uno MOLTO denso → SPLIT cluster denso
        3. Altrimenti → MIGRATE cluster interi
        
        PERCHÉ QUESTA GERARCHIA:
        - MIGRATE marginali: preserva cluster denso intatto, sposta solo outlier
        - SPLIT: usato solo quando necessario (cluster troppo grande)
        - MIGRATE generale: fallback quando non ci sono hotspot evidenti
        
        Args:
            overloaded_node: Nome del nodo sovraccarico
            node_loads: Dizionario carico nodi
            
        Returns:
            (strategy, metadata): 
                - strategy: "MIGRATE_MARGINAL", "SPLIT", o "MIGRATE"
                - metadata: Info aggiuntive per eseguire la strategia
        """
        # Analizza cluster sul nodo sovraccarico
        cluster_info = self._analyze_node_clusters(overloaded_node)
        
        n_clusters = len(cluster_info)
        largest_cluster_size = max(cluster_info.values(), key=lambda x: x['count'])['count'] if cluster_info else 0
        
        print(f"\n🔍 Analyzing {overloaded_node}:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Largest cluster size: {largest_cluster_size:,} vectors")
        print(f"  Hotspot threshold: {REBALANCER_THRESHOLD:,} vectors")
        
        # STRATEGIA 1: Se nodo ha multipli cluster → prova migrazione marginali
        if n_clusters >= 3:
            print(f"  → Strategy: MIGRATE_MARGINAL (multiple clusters detected)")
            print(f"     Rationale: Prefer moving peripheral clusters over splitting")
            
            return "MIGRATE_MARGINAL", {
                'cluster_info': cluster_info,
                'source_node': overloaded_node
            }
        
        # STRATEGIA 2: Se cluster molto denso → split
        elif largest_cluster_size > REBALANCER_THRESHOLD:
            print(f"  → Strategy: SPLIT (hotspot detected)")
            print(f"     Rationale: Single cluster too large, must split")
            
            return "SPLIT", {
                'cluster_info': cluster_info
            }
        
        # STRATEGIA 3: Fallback → migrazione generale
        else:
            print(f"  → Strategy: MIGRATE (general rebalancing)")
            print(f"     Rationale: No clear hotspot, distribute evenly")
            
            return "MIGRATE", {
                'cluster_info': cluster_info
            }
    
    def _analyze_node_clusters(self, node_name: str) -> dict:
        """
        Analizza tutti i cluster presenti su un nodo.
        
        COSA RESTITUISCE:
        - Dizionario con info per ogni cluster:
          - count: numero di vettori
          - centroid: centroide calcolato dai vettori
          - vectors_sample: sample di vettori per calcolo similarità
        
        Returns:
            {
                "0": {"count": 1500, "centroid": array([...]), "vectors": [...]},
                "5": {"count": 36000, "centroid": array([...]), "vectors": [...]},
                ...
            }
        """
        cluster_info = {}
        client = self.clients[node_name]
        
        # Raggruppa punti per cluster
        cluster_vectors = {}
        
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not records:
                break
            
            for record in records:
                cluster_id = record.payload.get("cluster_id", "unknown")
                if cluster_id not in cluster_vectors:
                    cluster_vectors[cluster_id] = []
                cluster_vectors[cluster_id].append(record.vector)
            
            offset = next_offset
            if offset is None:
                break
        
        # Calcola statistiche per ogni cluster
        for cluster_id, vectors in cluster_vectors.items():
            vectors_array = np.array(vectors, dtype='float32')
            centroid = np.mean(vectors_array, axis=0)
            
            cluster_info[cluster_id] = {
                'count': len(vectors),
                'centroid': centroid,
                'vectors_sample': vectors_array[:100]  # Sample per efficienza
            }
        
        return cluster_info
    
    def migrate_marginal_clusters(self, source_node: str, cluster_info: dict, target_nodes: list[str], n_clusters_to_move: int):
        """
        Migra cluster MARGINALI (semanticamente isolati) verso altri nodi.
        
        STRATEGIA:
        1. Calcola centroide "medio" di tutti i cluster sul nodo
        2. Calcola distanza di ogni cluster da questo centroide medio
        3. I cluster più LONTANI sono "marginali" (outlier semantici)
        4. Migra questi cluster marginali, preservando cluster centrali
        
        PERCHÉ QUESTA STRATEGIA:
        - Cluster centrali = core semantico del nodo (es. tutti "tech")
        - Cluster marginali = outlier (es. "food" in mezzo a "tech")
        - Migrare marginali ha MENO impatto su query semantiche
        - Preserva località per cluster densi e centrali
        
        ESEMPIO:
        Nodo ha cluster: [0, 1, 2, 5, 7]
        - Cluster 0, 1, 2, 5: tutti "tech related" (vicini semanticamente)
        - Cluster 7: "food" (outlier)
        
        Centroide medio: punto centrale tra 0,1,2,5 (ignora 7 come outlier)
        Distanze:
        - Cluster 0: distanza 0.2 (centrale)
        - Cluster 1: distanza 0.1 (centrale)
        - Cluster 2: distanza 0.3 (centrale)
        - Cluster 5: distanza 0.4 (centrale, ma denso 36K vettori)
        - Cluster 7: distanza 2.5 (MARGINALE!)
        
        → Migra cluster 7 (piccolo E marginale)
        → Preserva cluster 5 (denso ma centrale)
        
        Args:
            source_node: Nodo sorgente
            cluster_info: Informazioni sui cluster (da _analyze_node_clusters)
            target_nodes: Nodi destinazione
            n_clusters_to_move: Quanti cluster spostare
        """
        print(f"\n🔄 Migrating marginal clusters from {source_node}...")
        
        # FASE 1: Calcola centroide medio (centro semantico del nodo)
        """
        Il centroide medio rappresenta il "core" semantico del nodo.
        Cluster vicini a questo sono centrali, lontani sono marginali.
        """
        all_centroids = np.array([info['centroid'] for info in cluster_info.values()])
        mean_centroid = np.mean(all_centroids, axis=0)
        
        print(f"  Calculated mean centroid (semantic center of node)")
        
        # FASE 2: Calcola "marginalità" di ogni cluster
        """
        Marginalità = distanza dal centroide medio × fattore dimensione
        
        FORMULA: marginality_score = distance × (1 / log(count + 2))
        
        PERCHÉ:
        - Cluster lontano E piccolo → score alto (priorità migrazione)
        - Cluster lontano MA grande → score medio (potrebbe essere sub-topic valido)
        - Cluster vicino (centrale) → score basso (mantieni)
        """
        cluster_marginality = []
        
        for cluster_id, info in cluster_info.items():
            centroid = info['centroid']
            count = info['count']
            
            # Distanza euclidea dal centroide medio
            distance_from_center = np.linalg.norm(centroid - mean_centroid)
            
            # Penalizza cluster grandi (preferiamo migrare piccoli)
            size_penalty = 1.0 / np.log(count + 2)  # Log per smooth scaling
            
            # Score finale: alto = marginale + piccolo
            marginality_score = distance_from_center * size_penalty
            
            cluster_marginality.append({
                'cluster_id': cluster_id,
                'count': count,
                'distance': distance_from_center,
                'marginality': marginality_score
            })
        
        # Ordina per marginalità (decrescente: più marginali prima)
        cluster_marginality.sort(key=lambda x: x['marginality'], reverse=True)
        
        print(f"\n  Cluster marginality ranking:")
        for i, cluster in enumerate(cluster_marginality[:5]):  # Mostra top 5
            status = "← MARGINAL" if i < n_clusters_to_move else ""
            print(f"    {i+1}. Cluster {cluster['cluster_id']}: "
                  f"count={cluster['count']:,}, "
                  f"distance={cluster['distance']:.3f}, "
                  f"marginality={cluster['marginality']:.4f} {status}")
        
        # FASE 3: Seleziona cluster da migrare
        """
        Sceglie i top N cluster più marginali.
        """
        clusters_to_migrate = cluster_marginality[:n_clusters_to_move]
        
        print(f"\n  Selected {len(clusters_to_migrate)} cluster(s) for migration:")
        for cluster in clusters_to_migrate:
            print(f"    - Cluster {cluster['cluster_id']}: {cluster['count']:,} vectors (marginality: {cluster['marginality']:.4f})")
        
        # FASE 4: Assegna cluster a nodi target con load-aware strategy
        """
        Usa stesso scoring semantico+carico come nello split.
        """
        from config import ALPHA_SEMANTIC, BETA_LOAD
        
        # Calcola carico nodi
        node_loads = self.get_node_loads()
        total_vectors = sum(node_loads.values())
        avg_load = total_vectors / len(node_loads) if len(node_loads) > 0 else 0
        
        # Per ogni cluster, trova miglior nodo target
        migration_plan = {}
        
        for cluster_data in clusters_to_migrate:
            cluster_id = cluster_data['cluster_id']
            cluster_centroid = cluster_info[cluster_id]['centroid']
            
            # Calcola score per ogni nodo target (esclude source)
            node_scores = {}
            
            for target_node in target_nodes:
                if target_node == source_node:
                    continue  # Salta nodo sorgente
                
                # SEMANTIC SCORE: basato su similarità con altri cluster già sul target
                # Per semplicità, usiamo carico come proxy (meno carico = più spazio semantico)
                semantic_score = 0.5  # Base score
                
                # LOAD SCORE
                node_load = node_loads[target_node]
                if avg_load > 0:
                    load_ratio = node_load / avg_load
                    load_score = 1.0 - min(load_ratio, 2.0) / 2.0
                else:
                    load_score = 1.0
                
                # COMBINED SCORE
                combined_score = ALPHA_SEMANTIC * semantic_score + BETA_LOAD * load_score
                
                node_scores[target_node] = {
                    'semantic': semantic_score,
                    'load': load_score,
                    'combined': combined_score
                }
            
            # Scegli nodo con score migliore
            if node_scores:
                best_node = max(node_scores.items(), key=lambda x: x[1]['combined'])[0]
                migration_plan[cluster_id] = best_node
                
                print(f"    Cluster {cluster_id} → {best_node} (score: {node_scores[best_node]['combined']:.3f})")
        
        # FASE 5: Esegui migrazione
        """
        Sposta i vettori dei cluster selezionati verso i nodi target.
        """
        print(f"\n  Executing migration...")
        
        from qdrant_client.http.models import PointStruct
        BATCH_SIZE = 100
        
        source_client = self.clients[source_node]
        
        for cluster_id, target_node in migration_plan.items():
            print(f"    Migrating cluster {cluster_id} to {target_node}...")
            
            # Recupera tutti i punti del cluster
            points_to_migrate = []
            offset = None
            
            while True:
                records, next_offset = source_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(
                        must=[FieldCondition(
                            key="cluster_id",
                            match=MatchValue(value=cluster_id)
                        )]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not records:
                    break
                
                for record in records:
                    points_to_migrate.append(PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload
                    ))
                
                offset = next_offset
                if offset is None:
                    break
            
            # Inserisci su target node in batch
            target_client = self.clients[target_node]
            
            for batch_start in tqdm(range(0, len(points_to_migrate), BATCH_SIZE),
                                   desc=f"      Uploading to {target_node}",
                                   leave=False):
                batch_end = min(batch_start + BATCH_SIZE, len(points_to_migrate))
                batch = points_to_migrate[batch_start:batch_end]
                
                target_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch,
                    wait=True
                )
            
            # Cancella da source node
            source_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(
                        key="cluster_id",
                        match=MatchValue(value=cluster_id)
                    )]
                ),
                wait=True
            )
            
            # Aggiorna routing table
            self.routing_table._routing_map[cluster_id] = [target_node]
            
            print(f"      ✓ Migrated {len(points_to_migrate)} vectors")
        
        print(f"\n✅ Marginal cluster migration complete!")
    
    def migrate_clusters(self, source_node: str, target_nodes: list[str], n_clusters_to_move: int):
        """
        Migra cluster interi dal nodo sorgente ai nodi target.
        
        STRATEGIA:
        - Identifica cluster meno simili tra loro sul source_node
        - Migra questi cluster verso nodi target (distribuiti equamente)
        - Mantiene cluster integri (non li splitta)
        
        PERCHÉ CLUSTER MENO SIMILI:
        - Cluster simili dovrebbero restare insieme (località semantica)
        - Cluster dissimili possono andare su nodi diversi senza impatto query
        
        ALGORITMO:
        1. Recupera tutti i cluster sul source_node
        2. Calcola similarità tra cluster (distanza centroidi)
        3. Seleziona N cluster più "isolati" semanticamente
        4. Migra verso target_nodes in round-robin
        
        Args:
            source_node: Nodo da cui prendere cluster
            target_nodes: Nodi destinazione
            n_clusters_to_move: Quanti cluster spostare
        """
        print(f"\n🔄 Migrating {n_clusters_to_move} clusters from {source_node}...")
        
        # TODO: Implementazione completa
        # Per ora, placeholder che spiega la logica
        
        print(f"  [Placeholder] Would migrate clusters to: {target_nodes}")
        print(f"  Strategy: Select least similar clusters and distribute")
    
    def split_and_migrate_cluster(self, hot_cluster_id: int):
        """
        Splitta un cluster hotspot e migra i dati tra nodi con strategia load-aware.
        
        MODIFICA PRINCIPALE:
        - ESCLUDE il nodo sorgente dalla selezione (evita migrazione circolare)
        - Usa scoring ibrido sui nodi alternativi: similarità semantica (LSH) + carico nodo
        - Formula: score = α × semantic_score + β × load_score
        
        PERCHÉ ESCLUDERE IL NODO SORGENTE:
        - Se tutti i cluster sono su node-2, LSH continuerà a preferire node-2
        - Con α alto (0.7), semantica domina → sub-cluster tornano su node-2
        - Escludendo node-2, FORZIAMO redistribuzione su node-1 e node-3
        - Tra i nodi alternativi, scegliamo quello con miglior combinazione semantica+carico
        
        Args:
            hot_cluster_id: ID del cluster da splittare (es. 5)
        """
        print(f"\n🔄 Splitting cluster {hot_cluster_id}...")
        
        # --- FASE 1: RECUPERA DATI ---
        """
        Scarica tutti i vettori del cluster caldo.
        
        PERCHÉ:
        - Serve i vettori per ri-clusterizzare con K-means
        - Dobbiamo spostarli su altri nodi
        """
        source_node_name = self.routing_table.get_nodes(str(hot_cluster_id))[0]
        source_client = self.clients[source_node_name]
        
        print(f"  Fetching vectors from cluster {hot_cluster_id} on {source_node_name}...")
        
        vectors = []
        point_ids = []
        payloads = []
        
        # Scroll per recuperare tutti i punti del cluster
        offset = None
        while True:
            records, next_offset = source_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="cluster_id",
                        match=MatchValue(value=str(hot_cluster_id))
                    )]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True  # Questa volta serve il vettore!
            )
            
            if not records:
                break
            
            for record in records:
                vectors.append(record.vector)
                point_ids.append(record.id)
                payloads.append(record.payload)
            
            offset = next_offset
            if offset is None:
                break
        
        vectors = np.array(vectors, dtype='float32')
        print(f"  Fetched {len(vectors)} vectors from cluster {hot_cluster_id}")
        
        # --- FASE 2: SPLIT CON K-MEANS ---
        """
        Divide i vettori in sub-cluster usando K-means.
        
        PERCHÉ K-MEANS:
        - Mantiene similarità semantica: vettori simili restano insieme
        - Crea sub-cluster bilanciati
        - Preserva la qualità delle ricerche
        """
        print(f"  Running K-means to split into {N_SUB_CLUSTERS} sub-clusters...")
        
        d = vectors.shape[1]
        kmeans = faiss.Kmeans(d=d, k=N_SUB_CLUSTERS, niter=20, verbose=False)
        kmeans.train(vectors)
        
        # Assegna ogni vettore a un sub-cluster
        _, sub_cluster_labels = kmeans.index.search(vectors, 1)
        sub_cluster_labels = sub_cluster_labels.flatten()
        
        # Conta distribuzione
        unique, counts = np.unique(sub_cluster_labels, return_counts=True)
        print(f"  Sub-cluster distribution: {dict(zip(unique, counts))}")
        
        # --- FASE 3: CALCOLA CARICO NODI ---
        """
        NUOVO: Ottieni il carico corrente di ogni nodo per decisioni load-aware.
        """
        node_loads = self.get_node_loads()
        total_vectors = sum(node_loads.values())
        avg_load = total_vectors / len(node_loads) if len(node_loads) > 0 else 0
        
        print(f"\n  Average node load: {avg_load:,.0f} vectors")
        print(f"  Source node ({source_node_name}) will be EXCLUDED from sub-cluster assignment")
        
        # --- FASE 4: ASSEGNA SUB-CLUSTER CON LOAD-AWARE LSH (ESCLUDENDO SORGENTE) ---
        """
        STRATEGIA MODIFICATA:
        - ESCLUDI il nodo sorgente dalla selezione
        - Tra i nodi rimanenti, calcola score ibrido
        - Questo FORZA redistribuzione anche con α alto
        
        ESEMPIO:
        Source: node-2 (50K vettori, sovraccarico)
        Candidati: node-1 (0 vettori), node-3 (0 vettori)
        
        Sub-cluster 5.0:
        - LSH preferisce node-2 → MA node-2 ESCLUSO
        - Tra node-1 e node-3:
          - node-1: semantic=0.3, load=1.0 → score=0.51
          - node-3: semantic=0.3, load=1.0 → score=0.51
        - Tie-breaking: scegli primo (node-1)
        
        Risultato: Redistribuzione garantita!
        """
        from config import ALPHA_SEMANTIC, BETA_LOAD
        
        node_names = list(self.clients.keys())
        
        # MODIFICA CRITICA: Escludi nodo sorgente dai candidati
        candidate_nodes = [n for n in node_names if n != source_node_name]
        
        print(f"  Candidate nodes for migration: {candidate_nodes}")
        
        node_assignments = {}
        sub_cluster_ids = []
        
        # Genera hyperplanes LSH
        n_nodes = len(node_names)
        n_hyperplanes = int(np.ceil(np.log2(n_nodes)))
        
        np.random.seed(43)
        hyperplanes = np.random.randn(n_hyperplanes, d)
        hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        
        print(f"\n  Load-aware assignment (α={ALPHA_SEMANTIC}, β={BETA_LOAD}, excluding source):")
        
        for i in range(N_SUB_CLUSTERS):
            sub_id = f"{hot_cluster_id}.{i}"
            sub_cluster_ids.append(sub_id)
            
            centroid = kmeans.centroids[i]
            
            # --- CALCOLA SCORE SOLO PER NODI CANDIDATI (SENZA SORGENTE) ---
            node_scores = {}
            
            for node_name in candidate_nodes:  # SOLO nodi candidati
                # --- 1. SEMANTIC SCORE ---
                projections = np.dot(hyperplanes, centroid)
                hash_bits = (projections > 0).astype(int)
                hash_value = int(''.join(map(str, hash_bits)), 2)
                preferred_node_idx = hash_value % n_nodes
                
                if node_names[preferred_node_idx] == node_name:
                    semantic_score = 1.0
                else:
                    semantic_score = 0.3
                
                # --- 2. LOAD SCORE ---
                node_load = node_loads[node_name]
                
                if avg_load > 0:
                    load_ratio = node_load / avg_load
                    load_score = 1.0 - min(load_ratio, 2.0) / 2.0
                else:
                    load_score = 1.0
                
                # --- 3. SCORE COMBINATO ---
                combined_score = ALPHA_SEMANTIC * semantic_score + BETA_LOAD * load_score
                
                node_scores[node_name] = {
                    'semantic': semantic_score,
                    'load': load_score,
                    'combined': combined_score
                }
            
            # Scegli nodo con score più alto TRA I CANDIDATI
            best_node = max(node_scores.items(), key=lambda x: x[1]['combined'])[0]
            node_assignments[sub_id] = best_node
            
            # Stampa decisione
            best_scores = node_scores[best_node]
            print(f"    Sub-cluster {sub_id} → {best_node}")
            print(f"      Scores: semantic={best_scores['semantic']:.2f}, load={best_scores['load']:.2f}, combined={best_scores['combined']:.2f}")
            
            # Mostra alternative
            for node_name, scores in sorted(node_scores.items(), key=lambda x: x[1]['combined'], reverse=True)[1:]:
                print(f"      Alternative {node_name}: semantic={scores['semantic']:.2f}, load={scores['load']:.2f}, combined={scores['combined']:.2f}")
            
            print(f"      (Source node {source_node_name} excluded from selection)")
        
        # --- FASE 5: MIGRA DATI IN BATCH ---
        """
        MODIFICA CRITICA: Usa batch di massimo 100-200 punti per evitare payload troppo grande.
        
        PERCHÉ BATCH PICCOLI:
        - Qdrant limite: 32MB per richiesta JSON
        - Con vettori 128-dim: ~100-200 vettori = ~5-10MB
        - Sicuro sotto il limite
        """
        print(f"  Migrating data to target nodes in batches...")
        
        from qdrant_client.http.models import PointStruct
        
        BATCH_SIZE = 100  # Batch sicuro per evitare payload troppo grande
        
        for sub_cluster_idx in range(N_SUB_CLUSTERS):
            sub_id = sub_cluster_ids[sub_cluster_idx]
            target_node = node_assignments[sub_id]
            
            # Filtra vettori che appartengono a questo sub-cluster
            mask = sub_cluster_labels == sub_cluster_idx
            sub_vectors = vectors[mask]
            sub_ids = [point_ids[i] for i in range(len(point_ids)) if mask[i]]
            sub_payloads = [payloads[i] for i in range(len(payloads)) if mask[i]]
            
            total_points = len(sub_vectors)
            print(f"    Migrating {total_points} vectors to {target_node} (sub-cluster {sub_id})...")
            
            # Dividi in batch
            for batch_start in tqdm(range(0, total_points, BATCH_SIZE), 
                                   desc=f"    Uploading to {target_node}",
                                   leave=False):
                batch_end = min(batch_start + BATCH_SIZE, total_points)
                
                # Crea batch di punti
                batch_points = []
                for i in range(batch_start, batch_end):
                    payload = sub_payloads[i].copy()
                    payload["cluster_id"] = sub_id  # Aggiorna metadata
                    
                    batch_points.append(PointStruct(
                        id=sub_ids[i],
                        vector=sub_vectors[i].tolist(),
                        payload=payload
                    ))
                
                # Inserisci batch
                self.clients[target_node].upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch_points,
                    wait=True
                )
        
        # --- FASE 6: CLEANUP ---
        """
        Cancella i vecchi dati dal nodo originale.
        
        PERCHÉ:
        - I vettori sono stati copiati sui nuovi nodi
        - Mantenere duplicati spreca spazio
        - Cluster padre "5" non esiste più, solo "5.0", "5.1", "5.2"
        """
        print(f"  Deleting old cluster {hot_cluster_id} data from {source_node_name}...")
        source_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(
                    key="cluster_id",
                    match=MatchValue(value=str(hot_cluster_id))
                )]
            ),
            wait=True
        )
        
        # --- FASE 7: AGGIORNA ROUTING ---
        """
        Aggiorna la routing table per riflettere la nuova struttura.
        
        PRIMA:  routing_map = {"5": ["node-2"], ...}
        DOPO:   routing_map = {"5.0": ["node-1"], "5.1": ["node-2"], "5.2": ["node-3"], ...}
        """
        self.routing_table.split_cluster_mapping(
            old_cluster_id=hot_cluster_id,
            new_sub_cluster_ids=sub_cluster_ids,
            node_assignments=node_assignments
        )
        
        print(f"✅ Cluster {hot_cluster_id} successfully split and migrated!")