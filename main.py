# main.py
import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import uuid
import os

from wikipedia_loader import load_wikipedia_embeddings
from config import *
from quantizer import build_quantizer, load_quantizer, predict_cluster
from routing import RoutingTable

def setup_qdrant_collections(clients: dict[str, QdrantClient]):
    """Crea collections su tutti i nodi."""
    for node_name, client in clients.items():
        try:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=models.Distance.COSINE
                ),
            )
            print(f"✓ Collection '{COLLECTION_NAME}' created on {node_name}")
        except Exception as e:
            print(f"⚠️  Error on {node_name}: {e}")

def main():
    print("\n" + "="*70)
    print("SEMANTIC CLUSTERING & ROUTING SYSTEM")
    print("="*70 + "\n")
    
    # --- 1. SETUP NODI ---
    print("--- Step 1: Connecting to Qdrant nodes ---")
    clients = {name: QdrantClient(url=url) for name, url in QDRANT_NODES.items()}
    setup_qdrant_collections(clients)
    
    # --- 2. BUILD/LOAD QUANTIZER ---
    print("\n--- Step 2: Building K-means Quantizer ---")
    
    need_retrain = False
    
    if os.path.exists(QUANTIZER_PATH):
        try:
            quantizer_centroids = load_quantizer(QUANTIZER_PATH)
            expected_shape = (N_CLUSTERS, VECTOR_DIMENSION)
            actual_shape = quantizer_centroids.shape
            
            if actual_shape != expected_shape:
                print(f"⚠️  Dimension mismatch: expected {expected_shape}, found {actual_shape}")
                os.remove(QUANTIZER_PATH)
                need_retrain = True
            else:
                print(f"✓ Loaded quantizer: {actual_shape}")
        except Exception as e:
            print(f"⚠️  Error loading: {e}")
            need_retrain = True
    else:
        need_retrain = True
    
    if need_retrain:
        # SOLO WIKIPEDIA: nessuna opzione random
        print(f"Downloading Wikipedia embeddings ({WIKIPEDIA_LANGUAGE})...")
        sample_embeddings = load_wikipedia_embeddings(
            n_samples=SAMPLE_DATA_SIZE_FOR_TRAINING,
            language=WIKIPEDIA_LANGUAGE,
            cache_path=WIKIPEDIA_CACHE_PATH
        )
        
        quantizer_centroids = build_quantizer(sample_embeddings, N_CLUSTERS, QUANTIZER_PATH)
    
    # --- 3. SEMANTIC ROUTING ---
    print("\n--- Step 3: Creating Semantic Routing Table ---")
    routing_table = RoutingTable(list(QDRANT_NODES.keys()))
    routing_table.assign_clusters_by_semantic_similarity(N_CLUSTERS, quantizer_centroids)
    
    # --- 4. DATA INGESTION ---
    print(f"\n--- Step 4: Ingesting {TOTAL_VECTORS_TO_INSERT} vectors ---")
    points_buffer = {node: [] for node in QDRANT_NODES.keys()}
    
    for _ in tqdm(range(TOTAL_VECTORS_TO_INSERT), desc="Inserting vectors"):
        # Genera vettore random
        vector = np.random.rand(VECTOR_DIMENSION).astype('float32')
        
        # Predici cluster
        cluster_id = predict_cluster(quantizer_centroids, vector)
        
        # Trova nodo target
        target_node = routing_table.get_nodes(str(cluster_id))[0]
        
        # Crea punto
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={"cluster_id": str(cluster_id)}
        )
        points_buffer[target_node].append(point)
        
        # Batch insert
        if len(points_buffer[target_node]) >= 512:
            clients[target_node].upsert(
                collection_name=COLLECTION_NAME,
                points=points_buffer[target_node],
                wait=False
            )
            points_buffer[target_node] = []
    
    # Flush finale
    for node, buffer in points_buffer.items():
        if buffer:
            clients[node].upsert(
                collection_name=COLLECTION_NAME,
                points=buffer,
                wait=True
            )
    
    print("✓ Ingestion complete")
    
    # --- 5. STATISTICHE FINALI ---
    print("\n--- Step 5: Final Statistics ---")
    
    print("\nVectors per node:")
    for node_name, client in clients.items():
        count = client.count(collection_name=COLLECTION_NAME).count
        print(f"  {node_name}: {count:,} vectors")
    
    print("\nCluster distribution:")
    cluster_counts = {}
    for node_name, client in clients.items():
        offset = None
        while True:
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
                cid = record.payload.get("cluster_id")
                if cid:
                    cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            
            offset = next_offset
            if offset is None:
                break
    
    for cid in sorted(cluster_counts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        count = cluster_counts[cid]
        node = routing_table.get_nodes(cid)[0]
        print(f"  Cluster {cid}: {count:,} vectors on {node}")
    
    # --- 6. QUERY DEMO ---
    print("\n--- Step 6: Query Demonstration ---")
    
    # Query su cluster 0
    test_cluster = 0
    query_vector = quantizer_centroids[test_cluster] + np.random.normal(0, 0.01, VECTOR_DIMENSION)
    query_vector = query_vector.astype('float32')
    
    target_node = routing_table.get_nodes(str(test_cluster))[0]
    
    print(f"\nQuery semantically belongs to cluster {test_cluster}")
    print(f"Routing to: {target_node}")
    
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    
    results = clients[target_node].query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        query_filter=Filter(must=[
            FieldCondition(key="cluster_id", match=MatchValue(value=str(test_cluster)))
        ]),
        limit=5
    ).points
    
    if results:
        print(f"\n✓ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.4f}, Cluster: {result.payload['cluster_id']}")
    else:
        print("⚠️  No results found")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()