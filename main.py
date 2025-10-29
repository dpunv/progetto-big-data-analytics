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
    """Inizializza la collezione su tutti i nodi."""
    for node_name, client in clients.items():
        try:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
            )
            print(f"Collection '{COLLECTION_NAME}' created on {node_name}")
        except Exception as e:
            print(f"Collection on {node_name} might already exist. Error: {e}")

def main():
    # --- 1. SETUP ---
    print("--- 1. Initializing Environment ---")
    clients = {name: QdrantClient(url=url) for name, url in QDRANT_NODES.items()}
    setup_qdrant_collections(clients)

    # --- 2. BUILD QUANTIZER (se non esiste) ---
    print("\n--- 2. Building Quantizer ---")
    if not os.path.exists(QUANTIZER_PATH):
        sample_embeddings = np.random.rand(SAMPLE_DATA_SIZE_FOR_TRAINING, VECTOR_DIMENSION).astype('float32')
        quantizer_centroids = build_quantizer(sample_embeddings, N_CLUSTERS, QUANTIZER_PATH)
    else:
        quantizer_centroids = load_quantizer(QUANTIZER_PATH)

    # --- 3. INITIALIZE ROUTING ---
    print("\n--- 3. Initializing Routing Table ---")
    routing_table = RoutingTable(list(QDRANT_NODES.keys()))
    routing_table.assign_initial_clusters(N_CLUSTERS)

    # --- 4. INITIAL DATA INGESTION (con hotspot) ---
    print(f"\n--- 4. Ingesting {TOTAL_VECTORS_TO_INSERT} vectors (creating hotspot on cluster {HOTSPOT_CLUSTER_ID}) ---")
    points_buffer = {node: [] for node in QDRANT_NODES.keys()}
    
    for _ in tqdm(range(TOTAL_VECTORS_TO_INSERT), desc="Ingesting data"):
        vector = np.random.rand(VECTOR_DIMENSION).astype('float32')
        
        # Creiamo un bias per generare un hotspot
        if random.random() < HOTSPOT_BIAS_FACTOR:
            cluster_id = HOTSPOT_CLUSTER_ID
        else:
            cluster_id = predict_cluster(quantizer_centroids, vector)
            
        target_nodes = routing_table.get_nodes(str(cluster_id))
        target_node_name = target_nodes[0] # Pre-split, c'è solo un nodo

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={"cluster_id": str(cluster_id)}
        )
        points_buffer[target_node_name].append(point)

        # Flush buffer quando pieno
        if len(points_buffer[target_node_name]) >= 512:
            clients[target_node_name].upsert(
                collection_name=COLLECTION_NAME,
                points=points_buffer[target_node_name],
                wait=False
            )
            points_buffer[target_node_name] = []

    # Flush finale
    for node, buffer in points_buffer.items():
        if buffer:
            clients[node].upsert(collection_name=COLLECTION_NAME, points=buffer, wait=True)

    print("Initial ingestion complete.")

    # --- 5. MONITOR & REBALANCE ---
    print("\n--- 5. Starting Monitor & Rebalance Cycle ---")
    rebalancer = Rebalancer(clients, routing_table)
    cluster_counts = rebalancer.monitor_clusters()

    hot_clusters = [cid for cid, count in cluster_counts.items() if count > REBALANCER_THRESHOLD]
    
    if not hot_clusters:
        print("\n✅ No hotspots detected. System is balanced.")
    else:
        for hot_cluster_id in hot_clusters:
            rebalancer.split_and_migrate_cluster(int(hot_cluster_id))
            
    # --- 6. QUERY DEMONSTRATION ---
    print("\n--- 6. Query Demonstration ---")
    # Query per un vettore che apparterrebbe al cluster splittato
    query_vector_hot = quantizer_centroids[HOTSPOT_CLUSTER_ID] + np.random.normal(0, 0.01, VECTOR_DIMENSION)
    query_vector_hot = query_vector_hot.astype('float32')

    # Prima dello split, il cluster_id sarebbe stato semplice
    original_predicted_cluster = HOTSPOT_CLUSTER_ID
    print(f"Query vector would have belonged to original cluster: {original_predicted_cluster}")
    
    # Ora, dopo lo split, dobbiamo usare la logica di routing aggiornata
    # (In questo esempio non ricalcoliamo il sub-cluster, ma mostriamo dove cercheremmo)
    nodes_after_split = routing_table.get_nodes(f"{HOTSPOT_CLUSTER_ID}.0") # Esempio per il sub-cluster 0
    print(f"After split, queries for this semantic region are now routed to nodes for sub-clusters like '{HOTSPOT_CLUSTER_ID}.0', '{HOTSPOT_CLUSTER_ID}.1'...")
    print(f"For sub-cluster '{HOTSPOT_CLUSTER_ID}.0', the target node is: {nodes_after_split[0]}")
    
    # Eseguiamo la ricerca sul nodo corretto del sub-cluster
    search_results = clients[nodes_after_split[0]].search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector_hot.tolist(),
        query_filter=Filter(must=[FieldCondition(key="cluster_id", match=MatchValue(value=f"{HOTSPOT_CLUSTER_ID}.0"))]),
        limit=3
    )
    print(f"Found {len(search_results)} results in sub-cluster {HOTSPOT_CLUSTER_ID}.0:")
    for result in search_results:
        print(f"  - Point ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")

if __name__ == "__main__":
    main()