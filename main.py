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
    
    # --- 2. LOAD WIKIPEDIA EMBEDDINGS ---
    print("\n--- Step 2: Loading Wikipedia Embeddings ---")
    print(f"Downloading {TOTAL_VECTORS_TO_INSERT} Wikipedia sentences ({WIKIPEDIA_LANGUAGE})...")
    
    # CARICA EMBEDDINGS REALI (non random!)
    from wikipedia_loader import WikipediaEmbeddingGenerator
    
    generator = WikipediaEmbeddingGenerator(language=WIKIPEDIA_LANGUAGE)
    wikipedia_embeddings, wikipedia_sentences = generator.download_and_embed(
        n_samples=TOTAL_VECTORS_TO_INSERT,
        cache_path=WIKIPEDIA_CACHE_PATH
    )
    
    print(f"✓ Loaded {len(wikipedia_embeddings)} real Wikipedia embeddings")
    print(f"\nSample sentences:")
    for i in range(min(5, len(wikipedia_sentences))):
        print(f"  {i+1}. {wikipedia_sentences[i][:80]}...")
    
    # --- 3. BUILD/LOAD QUANTIZER ---
    print("\n--- Step 3: Building K-means Quantizer ---")
    
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
        # Usa subset Wikipedia per training K-means
        training_size = min(SAMPLE_DATA_SIZE_FOR_TRAINING, len(wikipedia_embeddings))
        sample_embeddings = wikipedia_embeddings[:training_size]
        
        quantizer_centroids = build_quantizer(sample_embeddings, N_CLUSTERS, QUANTIZER_PATH)
    
    # --- 4. SEMANTIC ROUTING ---
    print("\n--- Step 4: Creating Semantic Routing Table ---")
    routing_table = RoutingTable(list(QDRANT_NODES.keys()))
    routing_table.assign_clusters_by_semantic_similarity(N_CLUSTERS, quantizer_centroids)
    
    # --- 5. INGESTION WIKIPEDIA REALE ---
    print(f"\n--- Step 5: Ingesting {len(wikipedia_embeddings)} REAL Wikipedia vectors ---")
    points_buffer = {node: [] for node in QDRANT_NODES.keys()}
    
    for idx, (vector, sentence) in enumerate(tqdm(
        zip(wikipedia_embeddings, wikipedia_sentences), 
        total=len(wikipedia_embeddings),
        desc="Inserting Wikipedia data"
    )):
        # Predici cluster per questo embedding REALE
        cluster_id = predict_cluster(quantizer_centroids, vector)
        
        # Trova nodo target
        target_node = routing_table.get_nodes(str(cluster_id))[0]
        
        # Crea punto con METADATA RICCO
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "cluster_id": str(cluster_id),
                "sentence": sentence,  # Testo originale
                "source": "wikipedia",
                "index": idx
            }
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
    
    print("✓ Wikipedia ingestion complete")
    
    # --- 6. STATISTICHE FINALI ---
    print("\n--- Step 6: Final Statistics ---")
    
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
    
    # --- 7. SAMPLE DATA PER CLUSTER ---
    print("\n--- Step 7: Sample Sentences per Cluster ---")
    
    for cluster_id in range(min(3, N_CLUSTERS)):  # Mostra primi 3 cluster
        print(f"\n📂 Cluster {cluster_id} samples:")
        
        target_node = routing_table.get_nodes(str(cluster_id))[0]
        
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        
        samples = clients[target_node].scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="cluster_id", match=MatchValue(value=str(cluster_id)))
            ]),
            limit=5,
            with_payload=True,
            with_vectors=False
        )[0]
        
        for i, sample in enumerate(samples, 1):
            sentence = sample.payload.get("sentence", "N/A")
            print(f"  {i}. {sentence[:100]}...")
    
    # --- 8. QUERY DEMO ---
    print("\n--- Step 8: Query Demonstration ---")
    
    # Query con frase Wikipedia reale
    test_idx = 0
    test_vector = wikipedia_embeddings[test_idx]
    test_sentence = wikipedia_sentences[test_idx]
    
    cluster_id = predict_cluster(quantizer_centroids, test_vector)
    target_node = routing_table.get_nodes(str(cluster_id))[0]
    
    print(f"\nTest query:")
    print(f"  Sentence: {test_sentence[:100]}...")
    print(f"  Predicted cluster: {cluster_id}")
    print(f"  Routing to: {target_node}")
    
    results = clients[target_node].query_points(
        collection_name=COLLECTION_NAME,
        query=test_vector.tolist(),
        limit=5
    ).points
    
    if results:
        print(f"\n✓ Found {len(results)} similar results:")
        for i, result in enumerate(results, 1):
            result_sentence = result.payload.get("sentence", "N/A")
            print(f"  {i}. Score: {result.score:.4f}")
            print(f"     {result_sentence[:80]}...")
    else:
        print("⚠️  No results found")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()