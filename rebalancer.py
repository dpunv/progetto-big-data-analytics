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
    
    def split_and_migrate_cluster(self, hot_cluster_id: int):
        """
        Splitta un cluster hotspot e migra i dati tra nodi.
        
        SCOPO:
        - Ridurre il carico su un nodo sovraccarico
        - Distribuire i dati di un cluster caldo su più nodi
        
        FLUSSO COMPLETO:
        1. Recupera tutti i vettori del cluster caldo dal nodo originale
        2. Usa K-means per dividere in N_SUB_CLUSTERS sub-cluster
        3. Assegna ogni sub-cluster a un nodo (round-robin)
        4. Migra i vettori ai nodi destinazione
        5. Cancella i vettori vecchi dal nodo originale
        6. Aggiorna la routing table
        
        ESEMPIO:
        - Cluster 5 ha 35K vettori su node-2
        - Split in 3 sub-cluster:
          - 5.0 (11.6K vettori) → node-1
          - 5.1 (11.6K vettori) → node-2 (resta)
          - 5.2 (11.8K vettori) → node-3
        
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
        
        # --- FASE 3: ASSEGNA SUB-CLUSTER A NODI CON LSH ---
        """
        Usa LSH anche per assegnare i sub-cluster, mantenendo località semantica.
        I sub-cluster semanticamente vicini vanno su nodi vicini.
        """
        node_names = list(self.clients.keys())
        node_assignments = {}
        sub_cluster_ids = []
        
        # Calcola hash LSH per ogni centroide sub-cluster
        n_nodes = len(node_names)
        n_hyperplanes = int(np.ceil(np.log2(n_nodes)))
        
        np.random.seed(43)  # Seed diverso dal routing iniziale per variazione
        hyperplanes = np.random.randn(n_hyperplanes, d)
        hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        
        for i in range(N_SUB_CLUSTERS):
            sub_id = f"{hot_cluster_id}.{i}"
            sub_cluster_ids.append(sub_id)
            
            # Usa LSH per assegnare basandosi sul centroide del sub-cluster
            centroid = kmeans.centroids[i]
            projections = np.dot(hyperplanes, centroid)
            hash_bits = (projections > 0).astype(int)
            hash_value = int(''.join(map(str, hash_bits)), 2)
            
            node_idx = hash_value % n_nodes
            assigned_node = node_names[node_idx]
            node_assignments[sub_id] = assigned_node
            print(f"  Sub-cluster {sub_id} → {assigned_node} (LSH hash: {hash_value})")
        
        # --- FASE 4: MIGRA DATI IN BATCH ---
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
        
        # --- FASE 5: CLEANUP ---
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
        
        # --- FASE 6: AGGIORNA ROUTING ---
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