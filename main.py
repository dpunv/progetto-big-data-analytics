# main.py
import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import uuid
import random
import os

from config import *
from quantizer import build_quantizer, load_quantizer, predict_cluster
from routing import RoutingTable
from rebalancer import Rebalancer

def setup_qdrant_collections(clients: dict[str, QdrantClient]):
    """
    Inizializza la collezione su tutti i nodi.
    
    SCOPO:
    - Crea una collection Qdrant identica su ogni nodo del cluster
    - La collection è come un "database table" per vettori
    
    COSA FA:
    - Ricrea (cancella e ricrea) la collection su ogni nodo
    - Configura la dimensione dei vettori e la metrica di distanza
    
    PERCHÉ:
    - Ogni nodo deve avere la stessa struttura dati per compatibilità
    - recreate_collection assicura un ambiente pulito (utile per testing)
    - Distance.COSINE è ottima per embeddings semantici (misura similarità)
    
    Args:
        clients: Dizionario {nome_nodo: QdrantClient} per ogni nodo
    """
    for node_name, client in clients.items():
        try:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION,  # Es. 128 dimensioni
                    distance=models.Distance.COSINE  # Similarità coseno
                ),
            )
            print(f"Collection '{COLLECTION_NAME}' created on {node_name}")
        except Exception as e:
            print(f"Collection on {node_name} might already exist. Error: {e}")

def main():
    """
    Funzione principale che orchestra l'intero sistema di sharding semantico.
    
    FLUSSO COMPLETO:
    1. Setup: Connessione ai nodi e creazione collections
    2. Quantizer: Addestra/carica il modello K-means per clustering semantico
    3. Routing: Inizializza la mappa cluster→nodi
    4. Ingestion: Inserisce dati con bias per creare hotspot
    5. Monitor: Rileva cluster sovraccarichi
    6. Rebalance: Splitta e migra cluster caldi
    7. Query: Dimostra che il routing funziona dopo lo split
    """
    
    # --- 1. SETUP ---
    """
    Inizializza l'ambiente connettendosi a tutti i nodi Qdrant.
    
    COSA FA:
    - Crea un client Qdrant per ogni nodo (node-1, node-2, node-3)
    - Ogni client punta a un URL diverso (porta diversa)
    - Crea le collections su tutti i nodi
    
    PERCHÉ:
    - Abbiamo bisogno di comunicare con ogni nodo separatamente
    - Ogni nodo è indipendente (no clustering automatico Qdrant)
    - Il nostro codice gestisce manualmente il routing tra nodi
    """
    print("--- 1. Initializing Environment ---")
    # Crea dizionario {nome: client} es. {"node-1": QdrantClient("http://localhost:6333")}
    clients = {name: QdrantClient(url=url) for name, url in QDRANT_NODES.items()}
    setup_qdrant_collections(clients)

    # --- 2. BUILD QUANTIZER (se non esiste) ---
    """
    Addestra o carica il modello K-means per clustering semantico.
    
    COSA FA:
    - Se il file quantizer non esiste: addestra nuovo K-means
    - Se esiste: carica centroidi salvati
    
    PERCHÉ:
    - Il K-means divide lo spazio vettoriale in regioni semantiche
    - Training è costoso, quindi lo facciamo una sola volta
    - I centroidi determinano quale vettore va in quale cluster
    
    ESEMPIO:
    - Addestriamo su 10K vettori random per trovare 10 cluster
    - Risultato: 10 centroidi che rappresentano le "zone" dello spazio
    """
    print("\n--- 2. Building Quantizer ---")
    if not os.path.exists(QUANTIZER_PATH):
        # Genera dati di training (in produzione useresti veri embeddings)
        sample_embeddings = np.random.rand(SAMPLE_DATA_SIZE_FOR_TRAINING, VECTOR_DIMENSION).astype('float32')
        quantizer_centroids = build_quantizer(sample_embeddings, N_CLUSTERS, QUANTIZER_PATH)
    else:
        # Carica modello esistente
        quantizer_centroids = load_quantizer(QUANTIZER_PATH)

    # --- 3. INITIALIZE ROUTING ---
    """
    Crea la routing table che mappa cluster→nodi usando SIMILARITÀ SEMANTICA.
    
    NUOVA STRATEGIA (no più LSH):
    - Ordina cluster per similarità (centroidi vicini)
    - Divide sequenza ordinata in N parti uguali
    - Ogni nodo riceve cluster contigui semanticamente
    
    BENEFICI:
    - Nodo diventa "esperto" di un'area semantica
    - Query semantiche cercano 1 nodo invece di N
    - Distribuzione naturale basata su contenuto
    
    ESEMPIO:
    - node-1: cluster tech [0,1,2] (AI, ML, data)
    - node-2: cluster sport [3,4,5] (football, basketball, tennis)
    - node-3: cluster food [6,7,8] (italian, asian, desserts)
    
    Query "machine learning" → predice cluster 1 → cerca SOLO node-1
    """
    print("\n--- 3. Initializing Routing Table with Semantic Similarity ---")
    routing_table = RoutingTable(list(QDRANT_NODES.keys()))
    routing_table.assign_clusters_by_semantic_similarity(N_CLUSTERS, quantizer_centroids)

    # --- 4. INITIAL DATA INGESTION (con hotspot) ---
    """
    Inserisce vettori nel sistema, creando deliberatamente un hotspot.
    
    COSA FA:
    - Genera TOTAL_VECTORS_TO_INSERT vettori casuali
    - Per ogni vettore:
      1. Con probabilità HOTSPOT_BIAS_FACTOR → forza cluster HOTSPOT_CLUSTER_ID
      2. Altrimenti → predice cluster normalmente con K-means
    - Raggruppa vettori per nodo in buffer (performance)
    - Esegue upsert quando buffer pieno (batch insert)
    
    PERCHÉ CREIAMO UN HOTSPOT:
    - Vogliamo simulare uno scenario reale di sbilanciamento
    - Es. tanti utenti cercano "COVID-19" → cluster sovraccarico
    - Questo permette di testare il rebalancing
    
    COME FUNZIONA IL BUFFER:
    - Invece di inserire 1 vettore alla volta (lento)
    - Accumuliamo 512 vettori per nodo
    - Quando buffer pieno → upsert batch (molto più veloce)
    
    ESEMPIO:
    - Inseriamo 50K vettori
    - 70% vanno al cluster 5 (hotspot) → node-1 ha 35K vettori
    - 30% distribuiti sugli altri 9 cluster → ~1.6K per cluster
    - node-1 sovraccarico → trigger rebalancing
    """
    print(f"\n--- 4. Ingesting {TOTAL_VECTORS_TO_INSERT} vectors (creating hotspot on cluster {HOTSPOT_CLUSTER_ID}) ---")
    # Buffer per ogni nodo: accumula punti prima di inserirli
    points_buffer = {node: [] for node in QDRANT_NODES.keys()}
    
    for _ in tqdm(range(TOTAL_VECTORS_TO_INSERT), desc="Ingesting data"):
        # Genera vettore casuale
        vector = np.random.rand(VECTOR_DIMENSION).astype('float32')
        
        # Creiamo un bias per generare un hotspot
        if random.random() < HOTSPOT_BIAS_FACTOR:
            # Es. 70% di probabilità → forza cluster 5
            cluster_id = HOTSPOT_CLUSTER_ID
        else:
            # 30% → predice cluster normale basato su K-means
            cluster_id = predict_cluster(quantizer_centroids, vector)
            
        # Trova il nodo target usando la routing table
        target_nodes = routing_table.get_nodes(str(cluster_id))
        target_node_name = target_nodes[0]  # Pre-split, c'è solo un nodo per cluster

        # Crea punto Qdrant con ID unico, vettore e metadata
        point = models.PointStruct(
            id=str(uuid.uuid4()),  # UUID per garantire unicità
            vector=vector.tolist(),  # Vettore come lista Python
            payload={"cluster_id": str(cluster_id)}  # Metadata: a quale cluster appartiene
        )
        points_buffer[target_node_name].append(point)

        # Flush buffer quando pieno (batch insert per performance)
        if len(points_buffer[target_node_name]) >= 512:
            clients[target_node_name].upsert(
                collection_name=COLLECTION_NAME,
                points=points_buffer[target_node_name],
                wait=False  # Asincrono: non aspetta conferma
            )
            points_buffer[target_node_name] = []

    # Flush finale: inserisce punti rimanenti nei buffer
    for node, buffer in points_buffer.items():
        if buffer:
            clients[node].upsert(
                collection_name=COLLECTION_NAME, 
                points=buffer, 
                wait=True  # Sincrono: aspetta che finisca
            )

    print("Initial ingestion complete.")

    # --- 5. MONITOR & REBALANCE (NODE-AWARE) ---
    """
    Monitora e riequilibra il sistema con strategia a due livelli.
    
    NUOVA LOGICA:
    1. FASE 1 - Node-Level Check:
       - Verifica se i nodi sono bilanciati (carico totale)
       - Se bilanciati → SKIP rebalancing (sistema OK)
       - Se sbilanciati → procedi FASE 2
    
    2. FASE 2 - Cluster-Level Rebalancing:
       - Per ogni nodo sovraccarico, decide strategia:
         a) Se ha cluster HOT (> threshold) → SPLIT
         b) Altrimenti → MIGRATE cluster meno simili
    
    PERCHÉ QUESTO APPROCCIO:
    - Evita rebalancing inutile (se sistema già OK)
    - Risparmia risorse (split/migrazione sono costosi)
    - Considera contesto globale (load di tutti i nodi)
    - Preserva località semantica quando possibile
    
    ESEMPIO SCENARIO 1 (Bilanciato):
    - node-1: 16K vettori
    - node-2: 17K vettori
    - node-3: 16K vettori
    - Deviazione: 3% < 30% tolerance
    - Azione: NESSUNA (sistema già bilanciato)
    
    ESEMPIO SCENARIO 2 (Sbilanciato con hotspot):
    - node-1: 0 vettori
    - node-2: 50K vettori (cluster 5 ha 36K)
    - node-3: 0 vettori
    - Deviazione: 200% > 30% tolerance
    - Strategia: SPLIT cluster 5 in 3 parti
    - Risultato: 12K per nodo
    
    ESEMPIO SCENARIO 3 (Sbilanciato senza hotspot):
    - node-1: 0 vettori
    - node-2: 30K vettori (10 cluster da 3K ciascuno)
    - node-3: 0 vettori
    - Deviazione: 100% > 30% tolerance
    - Strategia: MIGRATE 6-7 cluster verso node-1 e node-3
    - Risultato: 10K per nodo
    """
    print("\n--- 5. Starting Node-Aware Monitor & Rebalance ---")
    rebalancer = Rebalancer(clients, routing_table)
    
    is_balanced, node_loads = rebalancer.is_node_balanced(tolerance=0.3)
    
    if is_balanced:
        print("\n✅ System is BALANCED. No rebalancing needed.")
        print("Skipping cluster-level analysis to save resources.")
    else:
        print("\n⚠️ System UNBALANCED. Proceeding with rebalancing...")
        
        total_vectors = sum(node_loads.values())
        avg_load = total_vectors / len(node_loads)
        max_acceptable = avg_load * 1.3
        
        overloaded_nodes = [
            node for node, load in node_loads.items() 
            if load > max_acceptable
        ]
        
        # FASE 2: Per ogni nodo sovraccarico, applica strategia
        for overloaded_node in overloaded_nodes:
            print(f"\n{'='*60}")
            print(f"Processing overloaded node: {overloaded_node}")
            print(f"{'='*60}")
            
            # Decide strategia (NUOVA: con metadata)
            strategy, metadata = rebalancer.identify_rebalancing_strategy(
                overloaded_node, 
                node_loads
            )
            
            if strategy == "MIGRATE_MARGINAL":
                # NUOVA STRATEGIA: Migra cluster marginali
                print(f"\n📦 Executing MIGRATE_MARGINAL strategy for {overloaded_node}")
                
                # Identifica nodi target
                min_acceptable = avg_load * 0.7
                target_nodes = [
                    node for node, load in node_loads.items()
                    if load < min_acceptable and node != overloaded_node
                ]
                
                if not target_nodes:
                    target_nodes = [n for n in node_loads.keys() if n != overloaded_node]
                
                print(f"  Target nodes: {target_nodes}")
                
                # Calcola quanti cluster spostare
                cluster_info = metadata['cluster_info']
                n_clusters_on_node = len(cluster_info)
                
                # Sposta 30-50% dei cluster (i più marginali)
                n_clusters_to_move = max(1, int(n_clusters_on_node * 0.4))
                n_clusters_to_move = min(n_clusters_to_move, n_clusters_on_node - 1)  # Lascia almeno 1
                
                print(f"  Planning to migrate {n_clusters_to_move} most marginal cluster(s)")
                
                # Esegui migrazione marginali
                rebalancer.migrate_marginal_clusters(
                    source_node=overloaded_node,
                    cluster_info=cluster_info,
                    target_nodes=target_nodes,
                    n_clusters_to_move=n_clusters_to_move
                )
            
            elif strategy == "SPLIT":
                # Strategia SPLIT: trova e splitta cluster hot
                print(f"\n📋 Executing SPLIT strategy for {overloaded_node}")
                
                cluster_counts = rebalancer.monitor_clusters()
                
                node_clusters = {}
                for cluster_id, count in cluster_counts.items():
                    try:
                        cluster_node = routing_table.get_nodes(cluster_id)[0]
                        if cluster_node == overloaded_node:
                            node_clusters[cluster_id] = count
                    except KeyError:
                        continue
                
                hot_clusters = [
                    cid for cid, count in node_clusters.items() 
                    if count > REBALANCER_THRESHOLD
                ]
                
                if hot_clusters:
                    print(f"  Found {len(hot_clusters)} hot cluster(s): {hot_clusters}")
                    
                    for hot_cluster_id in hot_clusters:
                        rebalancer.split_and_migrate_cluster(int(hot_cluster_id))
                else:
                    print(f"  No hot clusters found (threshold: {REBALANCER_THRESHOLD:,})")
                    if node_clusters:
                        print(f"  Largest cluster: {max(node_clusters.values()):,} vectors")
            
            elif strategy == "MIGRATE":
                # Strategia MIGRATE generale (fallback)
                print(f"\n📦 Executing MIGRATE strategy for {overloaded_node}")
                
                min_acceptable = avg_load * 0.7
                target_nodes = [
                    node for node, load in node_loads.items()
                    if load < min_acceptable and node != overloaded_node
                ]
                
                if not target_nodes:
                    target_nodes = [n for n in node_loads.keys() if n != overloaded_node]
                
                print(f"  Target nodes: {target_nodes}")
                
                excess_load = node_loads[overloaded_node] - avg_load
                avg_cluster_size = node_loads[overloaded_node] / 10
                n_clusters_to_move = int(excess_load / avg_cluster_size)
                n_clusters_to_move = max(1, min(n_clusters_to_move, 5))
                
                print(f"  Planning to migrate ~{n_clusters_to_move} clusters")
                
                rebalancer.migrate_clusters(
                    source_node=overloaded_node,
                    target_nodes=target_nodes,
                    n_clusters_to_move=n_clusters_to_move
                )
    
    # --- STATO FINALE (DOPO REBALANCING) ---
    """
    Mostra lo stato finale dei nodi e dei cluster dopo il rebalancing.
    
    COSA FA:
    - Stampa il numero di vettori su ogni nodo
    - Per ogni nodo sovraccarico, mostra quali cluster sono stati migrati
    
    PERCHÉ:
    - Verifica visivamente che il rebalancing abbia funzionato
    - Mostra come i cluster sono stati redistribuiti tra i nodi
    - Utile per debugging e monitoraggio
    
    COME INTERPRETARE:
    - Dovresti vedere un bilanciamento dei vettori tra i nodi
    - I nodi che prima erano sovraccarichi ora dovrebbero avere meno vettori
    - I nodi che erano sotto-caricati dovrebbero aver ricevuto nuovi cluster
    """
    print("\n--- Final State (After Rebalancing) ---")
    for node_name, client in clients.items():
        # Conta il numero totale di vettori su ogni nodo
        vector_count = client.count(collection_name=COLLECTION_NAME).count
        print(f"{node_name}: {vector_count} vectors")
        
        # Se il nodo è sovraccarico, mostra i dettagli dei cluster migrati
        if node_name in overloaded_nodes:
            print(f"  Overloaded node, showing migrated clusters details:")
            
            # Monitora i cluster su questo nodo
            cluster_counts = rebalancer.monitor_clusters()
            
            # Filtra solo i cluster che appartengono a questo nodo
            node_clusters = {
                cid: count for cid, count in cluster_counts.items() 
                if routing_table.get_nodes(cid)[0] == node_name
            }
            
            # Ordina i cluster per dimensione decrescente
            sorted_clusters = sorted(node_clusters.items(), key=lambda x: x[1], reverse=True)
            
            for cid, count in sorted_clusters:
                print(f"    - Cluster {cid}: {count} vectors")
    
    # --- 6. QUERY DEMONSTRATION ---
    """
    Dimostra che il sistema funziona correttamente dopo il rebalancing.
    
    MODIFICATO:
    - Adattato per gestire sia SPLIT che MIGRATE_MARGINAL
    - Verifica quale strategia è stata usata
    - Esegue query appropriate per ogni caso
    """
    print("\n--- 6. Query Demonstration ---")
    
    # Crea vettore query vicino al centroide del cluster hotspot
    query_vector_hot = quantizer_centroids[HOTSPOT_CLUSTER_ID] + np.random.normal(0, 0.01, VECTOR_DIMENSION)
    query_vector_hot = query_vector_hot.astype('float32')
    
    print(f"Query vector semantically belongs to cluster: {HOTSPOT_CLUSTER_ID}")
    
    # Determina quale cluster_id cercare (potrebbe essere splittato o meno)
    try:
        # Prova a cercare sub-cluster (se c'è stato split)
        sub_cluster_id = f"{HOTSPOT_CLUSTER_ID}.0"
        nodes_for_subcluster = routing_table.get_nodes(sub_cluster_id)
        
        # Se arriviamo qui, il cluster è stato splittato
        print(f"\n✓ Cluster {HOTSPOT_CLUSTER_ID} was SPLIT into sub-clusters")
        print(f"  Searching in sub-cluster '{sub_cluster_id}' on {nodes_for_subcluster[0]}")
        
        search_results = clients[nodes_for_subcluster[0]].query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector_hot.tolist(),
            query_filter=Filter(must=[
                FieldCondition(
                    key="cluster_id",
                    match=MatchValue(value=sub_cluster_id)
                )
            ]),
            limit=3
        ).points
        
        if search_results:
            print(f"\n  Found {len(search_results)} results in sub-cluster {sub_cluster_id}:")
            for result in search_results:
                print(f"    - Point ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")
        else:
            # Prova altri sub-cluster
            print(f"  No results in {sub_cluster_id}, trying other sub-clusters...")
            
            for i in range(N_SUB_CLUSTERS):
                alt_sub_id = f"{HOTSPOT_CLUSTER_ID}.{i}"
                try:
                    alt_nodes = routing_table.get_nodes(alt_sub_id)
                    alt_results = clients[alt_nodes[0]].query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector_hot.tolist(),
                        query_filter=Filter(must=[
                            FieldCondition(key="cluster_id", match=MatchValue(value=alt_sub_id))
                        ]),
                        limit=3
                    ).points
                    
                    if alt_results:
                        print(f"\n  Found {len(alt_results)} results in sub-cluster {alt_sub_id} on {alt_nodes[0]}:")
                        for result in alt_results:
                            print(f"    - Point ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")
                        break
                except KeyError:
                    continue
    
    except KeyError:
        # Cluster non splittato, cerca come cluster originale
        print(f"\n✓ Cluster {HOTSPOT_CLUSTER_ID} was NOT split (MIGRATE_MARGINAL or MIGRATE strategy)")
        
        # Trova dove si trova ora il cluster
        cluster_id = str(HOTSPOT_CLUSTER_ID)
        try:
            nodes_for_cluster = routing_table.get_nodes(cluster_id)
            print(f"  Cluster '{cluster_id}' is now on {nodes_for_cluster[0]}")
            
            search_results = clients[nodes_for_cluster[0]].query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector_hot.tolist(),
                query_filter=Filter(must=[
                    FieldCondition(
                        key="cluster_id",
                        match=MatchValue(value=cluster_id)
                    )
                ]),
                limit=3
            ).points
            
            if search_results:
                print(f"\n  Found {len(search_results)} results in cluster {cluster_id}:")
                for result in search_results:
                    print(f"    - Point ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")
            else:
                print(f"  ⚠️ No results found in cluster {cluster_id}")
        
        except KeyError:
            print(f"  ⚠️ Cluster {cluster_id} might have been migrated as a whole")
            print(f"  Searching across all nodes for cluster {cluster_id}...")
            
            # Cerca su tutti i nodi
            total_found = 0
            for node_name, client in clients.items():
                try:
                    results = client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector_hot.tolist(),
                        query_filter=Filter(must=[
                            FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))
                        ]),
                        limit=3
                    ).points
                    
                    if results:
                        print(f"\n  Found {len(results)} results on {node_name}:")
                        for result in results:
                            print(f"    - Point ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")
                        total_found += len(results)
                except Exception as e:
                    continue
            
            if total_found == 0:
                print(f"  ⚠️ No results found for cluster {cluster_id} across all nodes")
    
    # --- DIMOSTRAZIONE QUERY CROSS-NODE (opzionale) ---
    """
    Mostra come le query semantiche beneficiano del routing.
    """
    print(f"\n--- Query Routing Efficiency Demonstration ---")
    
    # Conta quanti cluster unici ci sono ora (inclusi sub-cluster)
    all_cluster_ids = set()
    for node_name, client in clients.items():
        offset = None
        while True:
            try:
                records, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not records:
                    break
                
                for record in records:
                    cluster_id = record.payload.get("cluster_id")
                    if cluster_id:
                        all_cluster_ids.add(cluster_id)
                
                offset = next_offset
                if offset is None:
                    break
            except Exception:
                break
    
    print(f"  Total unique clusters in system: {len(all_cluster_ids)}")
    print(f"  Clusters: {sorted(all_cluster_ids)}")
    
    # Mostra distribuzione finale
    print(f"\n  Final cluster distribution per node:")
    for node_name, client in clients.items():
        node_clusters = set()
        offset = None
        while True:
            try:
                records, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not records:
                    break
                
                for record in records:
                    cluster_id = record.payload.get("cluster_id")
                    if cluster_id:
                        node_clusters.add(cluster_id)
                
                offset = next_offset
                if offset is None:
                    break
            except Exception:
                break
        
        cluster_list = ', '.join(sorted(node_clusters)[:5])
        if len(node_clusters) > 5:
            cluster_list += f" ... ({len(node_clusters) - 5} more)"
        
        print(f"    {node_name}: {len(node_clusters)} clusters [{cluster_list}]")

if __name__ == "__main__":
    main()