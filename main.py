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
    Crea la routing table che mappa cluster→nodi usando LSH.
    
    COSA FA:
    - Crea oggetto RoutingTable con i nomi dei nodi
    - Assegna i cluster iniziali usando Locality-Sensitive Hashing
    
    PERCHÉ LSH INVECE DI ROUND-ROBIN:
    - LSH preserva località semantica: cluster simili → stesso nodo
    - Round-robin distribuiva cluster casuali su ogni nodo
    - Con LSH: query semantiche cercano su 1 nodo invece di N nodi
    
    DIFFERENZA:
    ROUND-ROBIN (prima):
    - node-1: cluster 0, 3, 6, 9 (semanticamente scollegati)
    - node-2: cluster 1, 4, 7
    - node-3: cluster 2, 5, 8
    - Query "tech topic" → potrebbe servire tutti e 3 i nodi
    
    LSH (ora):
    - node-1: cluster 0, 1, 2 (regione semantica coerente)
    - node-2: cluster 3, 4, 5 (altra regione coerente)
    - node-3: cluster 6, 7, 8, 9 (terza regione)
    - Query "tech topic" → serve SOLO 1 nodo (quello con cluster tech)
    
    COME FUNZIONA:
    - LSH usa random projections (hyperplanes)
    - Centroidi vicini nello spazio → hash simili → stesso nodo
    - Mantiene distribuzione bilanciata (modulo operation)
    """
    print("\n--- 3. Initializing Routing Table with LSH ---")
    routing_table = RoutingTable(list(QDRANT_NODES.keys()))
    routing_table.assign_initial_clusters_lsh(N_CLUSTERS, quantizer_centroids)

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

    # --- 5. MONITOR & REBALANCE ---
    """
    Monitora i cluster per rilevare hotspot e li riequilibra.
    
    COSA FA:
    1. Monitor: Conta quanti vettori ci sono in ogni cluster
    2. Detect: Identifica cluster che superano la soglia (REBALANCER_THRESHOLD)
    3. Rebalance: Per ogni cluster caldo, esegue split e migrazione
    
    COME FUNZIONA IL REBALANCING:
    - Cluster 5 ha 35K vettori (> threshold 5K)
    - Split: K-means divide cluster 5 in 3 sub-cluster (5.0, 5.1, 5.2)
    - Migrate: Sposta 5.1 e 5.2 su altri nodi (5.0 resta)
    - Update routing: Tabella ora sa che 5.0→node-1, 5.1→node-2, 5.2→node-3
    
    PERCHÉ:
    - Previene overload su un singolo nodo
    - Migliora performance distribuendo il carico
    - Mantiene latency bassa
    
    ESEMPIO OUTPUT:
    - Monitor: {0: 1500, 1: 1600, ..., 5: 35000, ...}
    - Detect: [5] è hot (35000 > 5000)
    - Rebalance: Split cluster 5 → 5.0 (12K), 5.1 (11.5K), 5.2 (11.5K)
    """
    print("\n--- 5. Starting Monitor & Rebalance Cycle ---")
    rebalancer = Rebalancer(clients, routing_table)
    cluster_counts = rebalancer.monitor_clusters()

    # Identifica cluster che superano la soglia
    hot_clusters = [cid for cid, count in cluster_counts.items() if count > REBALANCER_THRESHOLD]
    
    if not hot_clusters:
        print("\n✅ No hotspots detected. System is balanced.")
    else:
        # Splitta e migra ogni cluster caldo
        for hot_cluster_id in hot_clusters:
            rebalancer.split_and_migrate_cluster(int(hot_cluster_id))
            
    # --- 6. QUERY DEMONSTRATION ---
    """
    Dimostra che il sistema funziona correttamente dopo il rebalancing.
    
    COSA FA:
    - Crea un vettore simile al centroide del cluster splittato
    - Mostra come ora le query vengono instradate ai sub-cluster
    - Esegue una ricerca vettoriale su un sub-cluster specifico
    
    PERCHÉ:
    - Verifica che il routing funzioni dopo lo split
    - Mostra che possiamo ancora cercare vettori anche dopo la migrazione
    - Dimostra che i dati sono stati correttamente spostati
    
    COME FUNZIONA:
    - Prima: query cluster 5 → cerca su node-1
    - Dopo split: query cluster 5.0 → cerca su node-1
    -             query cluster 5.1 → cerca su node-2
    -             query cluster 5.2 → cerca su node-3
    
    ESEMPIO:
    - Vettore query simile al centroide originale del cluster 5
    - Ora appartiene semanticamente a sub-cluster 5.0
    - Routing table → 5.0 è su node-1
    - Search su node-1 con filtro cluster_id="5.0"
    - Trova i 3 vettori più simili nel sub-cluster
    """
    print("\n--- 6. Query Demonstration ---")
    # Query per un vettore che apparterrebbe al cluster splittato
    # Creiamo vettore vicino al centroide originale + piccolo rumore
    query_vector_hot = quantizer_centroids[HOTSPOT_CLUSTER_ID] + np.random.normal(0, 0.01, VECTOR_DIMENSION)
    query_vector_hot = query_vector_hot.astype('float32')

    # Prima dello split, il cluster_id sarebbe stato semplice
    original_predicted_cluster = HOTSPOT_CLUSTER_ID
    print(f"Query vector would have belonged to original cluster: {original_predicted_cluster}")
    
    # Ora, dopo lo split, dobbiamo usare la logica di routing aggiornata
    # (In questo esempio non ricalcoliamo il sub-cluster, ma mostriamo dove cercheremmo)
    nodes_after_split = routing_table.get_nodes(f"{HOTSPOT_CLUSTER_ID}.0")  # Esempio per il sub-cluster 0
    print(f"After split, queries for this semantic region are now routed to nodes for sub-clusters like '{HOTSPOT_CLUSTER_ID}.0', '{HOTSPOT_CLUSTER_ID}.1'...")
    print(f"For sub-cluster '{HOTSPOT_CLUSTER_ID}.0', the target node is: {nodes_after_split[0]}")
    
    # Eseguiamo la ricerca sul nodo corretto del sub-cluster
    search_results = clients[nodes_after_split[0]].search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector_hot.tolist(),  # Vettore query
        query_filter=Filter(must=[  # Filtro: cerca solo nel sub-cluster specifico
            FieldCondition(
                key="cluster_id", 
                match=MatchValue(value=f"{HOTSPOT_CLUSTER_ID}.0")
            )
        ]),
        limit=3  # Top 3 risultati più simili
    )
    print(f"Found {len(search_results)} results in sub-cluster {HOTSPOT_CLUSTER_ID}.0:")
    for result in search_results:
        print(f"  - Point ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")

if __name__ == "__main__":
    main()