# main.py
import numpy as np
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import uuid
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from wikipedia_loader import load_wikipedia_embeddings
from config import *
from quantizer import build_quantizer, load_quantizer, predict_cluster
from routing import RoutingTable
from meta_hnsw import MetaHNSW
from rebalancer import CapacityConstrainedRebalancer, simulate_capacity_aware_assignment
from replication import ReplicationManager, LoadAwareReplicaSelector

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

def parallel_upsert_batch(client: QdrantClient, 
                         collection_name: str, 
                         points: list, 
                         node_name: str,
                         wait: bool = False) -> tuple[str, int, float]:
    """
    Esegue upsert batch su Qdrant (funzione per thread pool).
    
    SCOPO:
    - Funzione wrapper per eseguire upsert in thread separato
    - Traccia timing per statistiche performance
    
    Args:
        client: Qdrant client per il nodo
        collection_name: Nome collection
        points: Lista di punti da inserire
        node_name: Nome nodo (per logging)
        wait: Se True, attende conferma scrittura
        
    Returns:
        (node_name, num_points, elapsed_time)
    """
    start_time = time.time()
    
    try:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait
        )
        elapsed = time.time() - start_time
        return (node_name, len(points), elapsed)
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⚠️  Error upserting to {node_name}: {e}")
        return (node_name, 0, elapsed)


def flush_buffers_parallel(clients: dict[str, QdrantClient], 
                          points_buffer: dict[str, list],
                          collection_name: str,
                          max_workers: int = 8,
                          wait: bool = False) -> dict[str, float]:
    """
    Flush tutti i buffer in parallelo usando ThreadPoolExecutor.
    
    VANTAGGI:
    - Upsert paralleli su nodi diversi
    - Riduce latenza totale da O(N×latency) a O(latency)
    - Sfrutta I/O concorrente
    
    ESEMPIO:
    Sequenziale (vecchio):
    - node-1: 50ms
    - node-2: 50ms  (attende node-1)
    - node-3: 50ms  (attende node-2)
    Total: 150ms
    
    Parallelo (nuovo):
    - node-1, node-2, node-3: tutti partono insieme
    Total: 50ms (tempo del più lento)
    
    Args:
        clients: Dictionary {node_name: QdrantClient}
        points_buffer: Dictionary {node_name: [points]}
        collection_name: Nome collection
        max_workers: Numero thread paralleli
        wait: Se True, attende conferma scrittura
        
    Returns:
        Dictionary {node_name: elapsed_time}
    """
    timings = {}
    
    # Filtra solo buffer non vuoti
    non_empty_buffers = {node: points for node, points in points_buffer.items() if points}
    
    if not non_empty_buffers:
        return timings
    
    # Esegui upsert in parallelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Sottometti tutti i task
        futures = {
            executor.submit(
                parallel_upsert_batch,
                clients[node],
                collection_name,
                points,
                node,
                wait
            ): node
            for node, points in non_empty_buffers.items()
        }
        
        # Raccogli risultati
        for future in as_completed(futures):
            node_name, num_points, elapsed = future.result()
            timings[node_name] = elapsed
    
    return timings


def main(num_nodes: int = None):
    """
    Main execution function.
    
    Args:
        num_nodes: Number of Qdrant nodes to connect to. If None, uses all nodes from QDRANT_NODES.
    """
    print("\n" + "="*70)
    print("SEMANTIC CLUSTERING & ROUTING SYSTEM")
    print("="*70 + "\n")
    
    # --- 1. SETUP NODI ---
    print("--- Step 1: Connecting to Qdrant nodes ---")
    
    # Filtra nodi se num_nodes è specificato
    if num_nodes is not None:
        all_nodes = dict(list(QDRANT_NODES.items())[:num_nodes])
        print(f"Using {num_nodes} nodes (specified as argument)")
    else:
        all_nodes = QDRANT_NODES
        print(f"Using all {len(QDRANT_NODES)} nodes from config")
    
    clients = {name: QdrantClient(url=url) for name, url in all_nodes.items()}
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
    routing_table = RoutingTable(list(all_nodes.keys()))
    routing_table.assign_clusters_by_semantic_similarity(N_CLUSTERS, quantizer_centroids)
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Capacity-Aware Cluster Assignment (PRE-INGESTION)
    # ═══════════════════════════════════════════════════════════════
    
    if ENABLE_AUTO_REBALANCING:
        print("\n--- Step 4b: Capacity-Constrained Cluster Validation (Pre-Ingestion) ---")
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Calcolo DINAMICO capacità basato su dataset e nodi
        # ═══════════════════════════════════════════════════════════════
        
        total_vectors = len(wikipedia_embeddings)
        num_nodes = len(all_nodes)
        avg_load_expected = total_vectors / num_nodes
        
        # Capacità dinamica = media × safety margin
        dynamic_capacity = int(avg_load_expected * CAPACITY_SAFETY_MARGIN)
        
        # Threshold dinamico = capacità × threshold
        dynamic_threshold = int(dynamic_capacity * REBALANCING_THRESHOLD)
        
        print(f"\n📊 Dynamic Capacity Calculation:")
        print(f"  Total vectors: {total_vectors:,}")
        print(f"  Number of nodes: {num_nodes}")
        print(f"  Average load (expected): {avg_load_expected:,.0f} vectors/node")
        print(f"  Safety margin: {CAPACITY_SAFETY_MARGIN}× (allows {(CAPACITY_SAFETY_MARGIN-1)*100:.0f}% variance)")
        print(f"  Capacity (dynamic): {dynamic_capacity:,} vectors/node")
        print(f"  Rebalancing threshold ({REBALANCING_THRESHOLD*100:.0f}%): {dynamic_threshold:,} vectors/node")
        
        # Predici cluster per tutti i vettori
        print("\nPredicting cluster assignments for all vectors...")
        all_cluster_assignments = np.array([
            predict_cluster(quantizer_centroids, vec) 
            for vec in tqdm(wikipedia_embeddings, desc="Predicting clusters")
        ])
        
        # Simula assignment con capacity constraints DINAMICA
        print(f"\nSimulating capacity-constrained assignment...")
        
        new_assignments, rebalancing_stats = simulate_capacity_aware_assignment(
            vectors=wikipedia_embeddings,
            cluster_assignments=all_cluster_assignments,
            cluster_centroids=quantizer_centroids,
            max_capacity_per_cluster=dynamic_capacity  # USA CAPACITÀ DINAMICA
        )
        
        # Statistiche rebalancing
        print(f"\n📊 Capacity Rebalancing Statistics:")
        print(f"  Initial max load: {rebalancing_stats['initial_max_load']:,} vectors")
        print(f"  Initial min load: {rebalancing_stats['initial_min_load']:,} vectors")
        print(f"  Vectors reassigned: {rebalancing_stats['vectors_reassigned']:,}")
        print(f"  Final max load: {rebalancing_stats['final_max_load']:,} vectors")
        print(f"  Final min load: {rebalancing_stats['final_min_load']:,} vectors")
        print(f"  Final imbalance: {rebalancing_stats['final_imbalance']:,} vectors")
        
        if rebalancing_stats['overflow_clusters']:
            print(f"\n  Overflow clusters fixed:")
            for overflow in rebalancing_stats['overflow_clusters']:
                print(f"    Cluster {overflow['cluster_id']}: reduced by {overflow['overflow']:,} vectors")
        
        # Aggiorna assignments globale
        print(f"\n✓ Using capacity-constrained assignments for ingestion")
        
        # Crea mapping per usare new_assignments durante ingestion
        vector_cluster_map = {idx: new_assignments[idx] for idx in range(len(new_assignments))}
    else:
        print("\n⚠️  Auto-rebalancing disabled, using standard K-means assignment")
        vector_cluster_map = None
        dynamic_capacity = None
        dynamic_threshold = None
    
    # --- 5. INGESTION WIKIPEDIA REALE ---
    print(f"\n--- Step 5: Ingesting {len(wikipedia_embeddings)} REAL Wikipedia vectors ---")
    print(f"Centroid update strategy: {CENTROID_UPDATE_STRATEGY}")
    if CENTROID_UPDATE_STRATEGY == 'batch':
        print(f"Update every {CENTROID_UPDATE_BATCH_SIZE} insertions per node")
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Configurazione parallel ingestion
    # ═══════════════════════════════════════════════════════════════
    print(f"Parallel ingestion: {INGESTION_PARALLEL_WORKERS} workers")
    print(f"Batch size: {QDRANT_UPSERT_BATCH_SIZE}")
    print(f"Upsert mode: {'sync' if QDRANT_UPSERT_WAIT else 'async'}")
    
    points_buffer = {node: [] for node in all_nodes.keys()}
    centroid_update_buffer = {node: [] for node in all_nodes.keys()}
    insert_counters = {node: 0 for node in all_nodes.keys()}
    
    # Statistiche performance
    total_upserts = 0
    total_upsert_time = 0.0
    upsert_timings = []
    
    # --- 5a. BUILD INITIAL META-HNSW ---
    print("\n--- Step 5a: Building Initial Meta-HNSW ---")
    meta_hnsw = MetaHNSW(
        dimension=VECTOR_DIMENSION,
        max_nodes=len(all_nodes),
        ef_construction=META_HNSW_EF_CONSTRUCTION,
        M=META_HNSW_M
    )
    
    # Configura rebuild policy
    meta_hnsw.rebuild_drift_threshold = HNSW_REBUILD_DRIFT_THRESHOLD
    meta_hnsw.rebuild_every_n_updates = HNSW_REBUILD_EVERY_N_UPDATES
    
    print(f"✓ HNSW rebuild policy:")
    print(f"  Drift threshold: {HNSW_REBUILD_DRIFT_THRESHOLD}")
    print(f"  Update interval: every {HNSW_REBUILD_EVERY_N_UPDATES} batches")
    
    # ═══════════════════════════════════════════════════════════════
    # MODIFICATO: Bootstrap centroidi con primi vettori reali
    # ═══════════════════════════════════════════════════════════════
    
    # Inizializza strutture per bootstrap
    for node_name in all_nodes.keys():
        meta_hnsw.node_names.append(node_name)
        meta_hnsw.node_vector_counts[node_name] = 0
        meta_hnsw.node_vector_sums[node_name] = np.zeros(VECTOR_DIMENSION)
        # NON inizializziamo node_centroids qui — aspettiamo primi vettori
    
    # Buffer per bootstrap (primo batch per ogni nodo)
    bootstrap_buffer = {node: [] for node in all_nodes.keys()}
    bootstrap_size = 100  # Usa primi 100 vettori per bootstrap
    is_bootstrapped = {node: False for node in all_nodes.keys()}
    
    print(f"✓ Meta-HNSW initialized with {len(all_nodes)} nodes")
    print(f"✓ Bootstrap mode: will use first {bootstrap_size} vectors per node for initial centroids")
    
    # --- 5b. INGESTIONE WIKIPEDIA REALE CON BOOTSTRAP ---
    print(f"\n--- Step 5b: Ingesting Wikipedia Vectors with Bootstrap ---")
    print(f"Centroid update strategy: {CENTROID_UPDATE_STRATEGY}")
    if CENTROID_UPDATE_STRATEGY == 'batch':
        print(f"Update every {CENTROID_UPDATE_BATCH_SIZE} insertions per node")
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Ingestion loop con bootstrap iniziale
    # ═══════════════════════════════════════════════════════════════
    
    ingestion_start = time.time()
    
    # Inizializza con centroidi vuoti
    for node_name in all_nodes.keys():
        meta_hnsw.node_centroids[node_name] = np.zeros(VECTOR_DIMENSION)
    
    for idx, (vector, sentence) in enumerate(tqdm(
        zip(wikipedia_embeddings, wikipedia_sentences), 
        total=len(wikipedia_embeddings),
        desc="Inserting Wikipedia data"
    )):
        # Predici cluster
        if vector_cluster_map is not None and idx in vector_cluster_map:
            # Usa assignment rebalanced
            cluster_id = vector_cluster_map[idx]
        else:
            # Fallback: predici normalmente
            cluster_id = predict_cluster(quantizer_centroids, vector)
        
        target_node = routing_table.get_nodes(str(cluster_id))[0]
        
        # Crea punto
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "cluster_id": str(cluster_id),
                "sentence": sentence,
                "source": "wikipedia",
                "index": idx
            }
        )
        points_buffer[target_node].append(point)
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Bootstrap phase - accumula primi vettori
        # ═══════════════════════════════════════════════════════════════
        
        if not is_bootstrapped[target_node]:
            bootstrap_buffer[target_node].append(vector)
            
            # Quando raggiungiamo bootstrap_size, calcola centroide iniziale
            if len(bootstrap_buffer[target_node]) >= bootstrap_size:
                vectors_array = np.array(bootstrap_buffer[target_node])
                
                # Calcola centroide iniziale (media normalizzata)
                initial_centroid = np.mean(vectors_array, axis=0)
                initial_centroid = initial_centroid / (np.linalg.norm(initial_centroid) + 1e-8)
                
                # Inizializza meta-HNSW per questo nodo
                meta_hnsw.node_centroids[target_node] = initial_centroid
                meta_hnsw.node_vector_counts[target_node] = len(vectors_array)
                meta_hnsw.node_vector_sums[target_node] = np.sum(vectors_array, axis=0)
                
                # Marca come bootstrapped
                is_bootstrapped[target_node] = True
                
                print(f"\n  ✓ Bootstrap {target_node}: centroid initialized from {len(vectors_array)} vectors")
                print(f"    Centroid magnitude: {np.linalg.norm(initial_centroid):.6f}")
                
                # Libera memoria
                bootstrap_buffer[target_node] = []
        
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Update centroidi solo DOPO bootstrap
        # ═══════════════════════════════════════════════════════════════
        
        # Skip update se nodo non ancora bootstrapped
        if not is_bootstrapped[target_node]:
            # Durante bootstrap, non fare update (aspettiamo primo centroide reale)
            pass
        
        elif CENTROID_UPDATE_STRATEGY == 'incremental':
            # Update ad ogni inserimento
            meta_hnsw.update_centroid_incremental(target_node, vector)
        
        elif CENTROID_UPDATE_STRATEGY == 'batch':
            # Accumula in buffer, update ogni N inserimenti
            centroid_update_buffer[target_node].append(vector)
            insert_counters[target_node] += 1
            
            # Trigger batch update
            if insert_counters[target_node] >= CENTROID_UPDATE_BATCH_SIZE:
                vectors_batch = np.array(centroid_update_buffer[target_node])
                meta_hnsw.update_centroid_batch(target_node, vectors_batch)
                
                # Reset buffer
                centroid_update_buffer[target_node] = []
                insert_counters[target_node] = 0
        
        # ═══════════════════════════════════════════════════════════════
        # MODIFICATO: Batch insert PARALLELO
        # ═══════════════════════════════════════════════════════════════
        
        # Controlla se qualche buffer ha raggiunto la soglia
        nodes_to_flush = [node for node, buffer in points_buffer.items() 
                         if len(buffer) >= QDRANT_UPSERT_BATCH_SIZE]
        
        if nodes_to_flush:
            # Prepara buffer da flushare
            buffers_to_flush = {node: points_buffer[node] for node in nodes_to_flush}
            
            # Flush parallelo
            flush_start = time.time()
            timings = flush_buffers_parallel(
                clients=clients,
                points_buffer=buffers_to_flush,
                collection_name=COLLECTION_NAME,
                max_workers=INGESTION_PARALLEL_WORKERS,
                wait=QDRANT_UPSERT_WAIT
            )
            flush_elapsed = time.time() - flush_start
            
            # Statistiche
            total_upserts += len(nodes_to_flush)
            total_upsert_time += flush_elapsed
            upsert_timings.append(flush_elapsed)
            
            # Reset buffer flushati
            for node in nodes_to_flush:
                points_buffer[node] = []
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Post-bootstrap check
    # ═══════════════════════════════════════════════════════════════
    
    print("\n🔧 Post-ingestion bootstrap check...")
    
    # Controlla se qualche nodo non ha ricevuto abbastanza vettori per bootstrap
    non_bootstrapped_nodes = [node for node, bootstrapped in is_bootstrapped.items() 
                              if not bootstrapped]
    
    if non_bootstrapped_nodes:
        print(f"⚠️  {len(non_bootstrapped_nodes)} nodes did not receive enough vectors for bootstrap:")
        
        for node_name in non_bootstrapped_nodes:
            vectors_received = len(bootstrap_buffer[node_name])
            print(f"  {node_name}: only {vectors_received} vectors (needed {bootstrap_size})")
            
            if vectors_received > 0:
                # Bootstrap con vettori disponibili (anche se < threshold)
                vectors_array = np.array(bootstrap_buffer[node_name])
                initial_centroid = np.mean(vectors_array, axis=0)
                initial_centroid = initial_centroid / (np.linalg.norm(initial_centroid) + 1e-8)
                
                meta_hnsw.node_centroids[node_name] = initial_centroid
                meta_hnsw.node_vector_counts[node_name] = len(vectors_array)
                meta_hnsw.node_vector_sums[node_name] = np.sum(vectors_array, axis=0)
                
                print(f"    ✓ Bootstrapped with {vectors_received} vectors (partial)")
            else:
                # Fallback: usa media di tutti i centroidi esistenti
                if meta_hnsw.node_centroids:
                    avg_centroid = np.mean(
                        list(meta_hnsw.node_centroids.values()), 
                        axis=0
                    )
                    avg_centroid = avg_centroid / (np.linalg.norm(avg_centroid) + 1e-8)
                    
                    meta_hnsw.node_centroids[node_name] = avg_centroid
                    meta_hnsw.node_vector_counts[node_name] = 0
                    meta_hnsw.node_vector_sums[node_name] = np.zeros(VECTOR_DIMENSION)
                    
                    print(f"    ⚠️  Fallback: using average of other centroids")
                else:
                    # Ultimo fallback: vettore casuale normalizzato
                    random_centroid = np.random.randn(VECTOR_DIMENSION).astype('float32')
                    random_centroid = random_centroid / (np.linalg.norm(random_centroid) + 1e-8)
                    
                    meta_hnsw.node_centroids[node_name] = random_centroid
                    meta_hnsw.node_vector_counts[node_name] = 0
                    meta_hnsw.node_vector_sums[node_name] = np.zeros(VECTOR_DIMENSION)
                    
                    print(f"    ⚠️  Fallback: using random normalized vector")
    else:
        print(f"✓ All {len(all_nodes)} nodes successfully bootstrapped")
    
    # Controlla centroidi finali
    print(f"\n📊 Bootstrap Statistics:")
    for node_name in sorted(all_nodes.keys()):
        if node_name in meta_hnsw.node_centroids:
            centroid = meta_hnsw.node_centroids[node_name]
            magnitude = np.linalg.norm(centroid)
            count = meta_hnsw.node_vector_counts[node_name]
            
            status = "✓" if magnitude > 0.9 else "⚠️"
            print(f"  {status} {node_name}: {count} vectors, magnitude {magnitude:.6f}")
    
    # ═══════════════════════════════════════════════════════════════
    # MODIFICATO: Flush finale PARALLELO
    # ═══════════════════════════════════════════════════════════════
    
    print("\n🔄 Final flush of remaining buffers...")
    flush_start = time.time()
    
    final_timings = flush_buffers_parallel(
        clients=clients,
        points_buffer=points_buffer,
        collection_name=COLLECTION_NAME,
        max_workers=INGESTION_PARALLEL_WORKERS,
        wait=True  # Finale: aspetta conferma
    )
    
    final_flush_elapsed = time.time() - flush_start
    total_upsert_time += final_flush_elapsed
    
    print(f"✓ Final flush completed in {final_flush_elapsed:.2f}s")
    for node, elapsed in final_timings.items():
        num_points = len(points_buffer[node]) if node in points_buffer else 0
        if num_points > 0:
            print(f"  {node}: {num_points} points in {elapsed:.3f}s")
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Statistiche performance ingestion
    # ═══════════════════════════════════════════════════════════════
    
    ingestion_elapsed = time.time() - ingestion_start
    
    print("\n📊 Ingestion Performance Statistics:")
    print(f"  Total time: {ingestion_elapsed:.2f}s")
    print(f"  Total vectors: {len(wikipedia_embeddings):,}")
    print(f"  Throughput: {len(wikipedia_embeddings) / ingestion_elapsed:.0f} vectors/sec")
    print(f"  Total upsert operations: {total_upserts}")
    print(f"  Total upsert time: {total_upsert_time:.2f}s")
    
    if upsert_timings:
        print(f"  Average upsert time: {np.mean(upsert_timings):.3f}s")
        print(f"  Min upsert time: {np.min(upsert_timings):.3f}s")
        print(f"  Max upsert time: {np.max(upsert_timings):.3f}s")
        print(f"  Upsert overhead: {total_upsert_time / ingestion_elapsed * 100:.1f}%")
    
    print(f"\n  💡 Parallel ingestion with {INGESTION_PARALLEL_WORKERS} workers:")
    print(f"     Estimated sequential time: ~{total_upsert_time * INGESTION_PARALLEL_WORKERS / 2:.1f}s")
    print(f"     Actual time: {ingestion_elapsed:.1f}s")
    print(f"     Speedup: ~{(total_upsert_time * INGESTION_PARALLEL_WORKERS / 2) / ingestion_elapsed:.1f}x")
    
    print("✓ Wikipedia ingestion complete")
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Se strategy='none', costruisci centroidi DOPO ingestion
    # ═══════════════════════════════════════════════════════════════
    if CENTROID_UPDATE_STRATEGY == 'none':
        print("\n--- Step 5b: Building Meta-HNSW (post-ingestion) ---")
        
        for node_name, client in clients.items():
            records, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=False,
                with_vectors=True
            )
            
            if not records:
                continue
            
            node_vectors = np.array([record.vector for record in records])
            meta_hnsw.add_node_centroid(
                node_name=node_name,
                vectors=node_vectors,
                method=CENTROID_CALCULATION_METHOD
            )
    
    # Salva meta-HNSW (ora con centroidi aggiornati)
    meta_hnsw.save(META_HNSW_PATH)
    
    # Statistiche meta-HNSW
    stats = meta_hnsw.get_statistics()
    print(f"\n📊 Meta-HNSW Statistics (after ingestion):")
    print(f"  Nodi indicizzati: {stats['num_nodes']}")
    print(f"  Dimensione: {stats['dimension']}")
    print(f"  Total vectors tracked: {stats.get('total_vectors_tracked', 0):,}")
    print(f"  Distanza minima tra centroidi: {stats.get('min_distance', 0):.4f}")
    print(f"  Distanza massima tra centroidi: {stats.get('max_distance', 0):.4f}")
    print(f"  Distanza media: {stats.get('mean_distance', 0):.4f}")
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Statistiche rebuild
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  🔧 Rebuild Statistics:")
    print(f"    Max centroid drift: {stats.get('max_centroid_drift', 0):.6f}")
    print(f"    Rebuild threshold: {stats.get('rebuild_threshold', 0):.6f}")
    print(f"    Needs rebuild: {stats.get('needs_rebuild', False)}")
    print(f"    Updates since rebuild: {stats.get('updates_since_rebuild', 0)}")
    
    print(f"\n  Vectors per node (tracked in meta-HNSW):")
    for node_name, count in stats.get('vectors_per_node', {}).items():
        print(f"    {node_name}: {count:,}")
    
    # --- 6. STATISTICHE FINALI ---
    print("\n--- Step 6: Final Statistics ---")
    
    print("\nVectors per node:")
    for node_name, client in clients.items():
        count = client.count(collection_name=COLLECTION_NAME).count
        print(f"  {node_name}: {count:,} vectors")
    
    print("\nCluster distribution:")
    cluster_counts = {}
    for node_name, client in clients.items():
        try:
            # Controlla se collection ha vettori prima di scrollare
            count = client.count(collection_name=COLLECTION_NAME).count
            if count == 0:
                print(f"  ⚠️  {node_name}: empty collection, skipping scroll")
                continue
            
            offset = None
            scroll_attempts = 0
            max_scroll_attempts = 100  # Limita numero scroll per evitare loop infiniti
            
            while scroll_attempts < max_scroll_attempts:
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
                        cid = record.payload.get("cluster_id")
                        if cid:
                            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
                    
                    offset = next_offset
                    scroll_attempts += 1
                    
                    if offset is None:
                        break
                        
                except Exception as scroll_error:
                    print(f"  ⚠️  Error scrolling {node_name}: {scroll_error}")
                    break
                    
        except Exception as e:
            print(f"  ⚠️  Error processing {node_name}: {e}")
            continue
    
    for cid in sorted(cluster_counts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        count = cluster_counts[cid]
        node = routing_table.get_nodes(cid)[0]
        print(f"  Cluster {cid}: {count:,} vectors on {node}")
    
    # --- 6b. NODE CENTROID STATISTICS ---
    print("\n--- Step 6b: Node Centroid Statistics & Load Balancing ---")
    
    node_stats = {}
    
    for node_name in all_nodes.keys():
        # Ottieni cluster assegnati a questo nodo
        assigned_clusters = [str(cid) for cid in range(N_CLUSTERS) 
                           if routing_table.get_nodes(str(cid))[0] == node_name]
        
        # Calcola media centroidi per questo nodo
        if assigned_clusters:
            centroids_for_node = quantizer_centroids[
                [int(cid) for cid in assigned_clusters]
            ]
            node_centroid_mean = np.mean(centroids_for_node, axis=0)
        else:
            node_centroid_mean = np.zeros(VECTOR_DIMENSION)
        
        # Calcola carico (numero vettori)
        load = sum(cluster_counts.get(cid, 0) for cid in assigned_clusters)
        
        node_stats[node_name] = {
            'centroid': node_centroid_mean,
            'load': load,
            'assigned_clusters': assigned_clusters,
            'n_clusters': len(assigned_clusters)
        }
    
    # Stampa statistiche per nodo
    print("\nNode Statistics:")
    total_vectors = sum(stat['load'] for stat in node_stats.values())
    avg_load = total_vectors / len(node_stats) if node_stats else 0
    
    for node_name, stats in sorted(node_stats.items()):
        load_pct = (stats['load'] / total_vectors * 100) if total_vectors > 0 else 0
        balance_indicator = "✓" if abs(stats['load'] - avg_load) < avg_load * 0.2 else "⚠️"
        
        print(f"\n  {balance_indicator} {node_name}:")
        print(f"     Load: {stats['load']:,} vectors ({load_pct:.1f}%)")
        print(f"     Assigned clusters: {len(stats['assigned_clusters'])}")
        print(f"     Cluster IDs: {stats['assigned_clusters'][:5]}{'...' if len(stats['assigned_clusters']) > 5 else ''}")
        print(f"     Mean centroid (first 5 dims): {stats['centroid'][:5]}")
    
    # Statistiche di bilanciamento globale
    print("\n  Load Balancing Summary:")
    loads = [stat['load'] for stat in node_stats.values()]
    max_load = max(loads) if loads else 0
    min_load = min(loads) if loads else 0
    std_dev = np.std(loads) if loads else 0
    imbalance_ratio = (max_load - min_load) / avg_load if avg_load > 0 else 0
    
    print(f"    Total vectors: {total_vectors:,}")
    print(f"    Average load per node: {avg_load:.1f}")
    print(f"    Max load: {max_load:,}")
    print(f"    Min load: {min_load:,}")
    print(f"    Std deviation: {std_dev:.1f}")
    print(f"    Imbalance ratio: {imbalance_ratio:.2f} ({'Balanced' if imbalance_ratio < 0.3 else 'Unbalanced'})")
    
    # ═══════════════════════════════════════════════════════════════
    # NUOVO: Step 6d - REPLICATION SYSTEM
    # ═══════════════════════════════════════════════════════════════
    
    if ENABLE_REPLICATION:
        print("\n--- Step 6d: Hot Cluster Replication ---")
        
        replication_manager = ReplicationManager(replication_factor=REPLICATION_FACTOR)
        
        simulated_cluster_accesses = {
            str(cid): cluster_counts.get(str(cid), 0) * 10
            for cid in range(N_CLUSTERS)
        }
        
        replication_plan = replication_manager.calculate_replication_plan(
            node_loads={name: node_stats[name]['load'] for name in all_nodes.keys()},
            cluster_loads=simulated_cluster_accesses,
            routing_table=routing_table,
            threshold_pct=REBALANCING_THRESHOLD,
            force_replication=False,
            min_replicas_per_cluster=MIN_REPLICAS_PER_CLUSTER,
            max_replicas_per_node=MAX_REPLICAS_PER_NODE
        )
        
        if replication_plan:
            print(f"\n  📋 Replication Plan ({len(replication_plan)} operations):")
            for i, op in enumerate(replication_plan, 1):
                reason_emoji = "🔥" if "hot" in op['reason'] else "⚖️"
                print(f"     {i}. {reason_emoji} Cluster {op['cluster_id']}: {op['source_node']} → {op['target_node']}")
                print(f"        Query count: {op['query_count']:,}, Est. vectors: {op['estimated_vectors']:,}")
                print(f"        Reason: {op['reason']}")
            
            total_replicated = replication_manager.execute_replication(
                clients=clients,
                plan=replication_plan,
                collection_name=COLLECTION_NAME,
                routing_table=routing_table
            )
            
            repl_stats = routing_table.get_replication_stats()
            print(f"\n  📊 Replication Statistics:")
            print(f"     Clusters with replicas: {repl_stats['total_clusters_with_replicas']}")
            print(f"     Total replicas created: {repl_stats['total_replicas']}")
            print(f"     Avg replicas per cluster: {repl_stats['avg_replicas_per_cluster']:.2f}")
            
            storage_overhead = (total_replicated / total_vectors * 100) if total_vectors > 0 else 0
            print(f"     Storage overhead: +{storage_overhead:.1f}%")
            print(f"     Read capacity gain: ~{REPLICATION_FACTOR}x")
            
            print(f"\n  📂 Replication Details:")
            for cluster_id, nodes in sorted(repl_stats['clusters'].items(), key=lambda x: int(x[0])):
                if nodes > 1:
                    all_nodes_for_cluster = routing_table.get_all_nodes_for_cluster(cluster_id)
                    print(f"     Cluster {cluster_id}: {nodes} copies → {all_nodes_for_cluster}")
            
            # ═══════════════════════════════════════════════════════════════
            # NUOVO: Statistiche peso nodi dopo replicazione
            # ═══════════════════════════════════════════════════════════════
            
            print(f"\n  ⚖️  Node Weight After Replication:")
            print(f"      (Primary vectors + Replica vectors = Total weight)")
            
            # Calcola peso per ogni nodo
            node_weights = {}
            for node_name in all_nodes.keys():
                # Carico primary (cluster assegnati originalmente)
                primary_load = node_stats[node_name]['load']
                
                # Carico repliche (cluster replicati su questo nodo)
                replica_load = 0
                for cluster_id_str, replica_nodes in routing_table.replicas.items():
                    # Se questo nodo ha una replica (non è il primary)
                    if node_name in replica_nodes[1:]:  # Skip primary (indice 0)
                        replica_load += cluster_counts.get(cluster_id_str, 0)
                
                total_weight = primary_load + replica_load
                node_weights[node_name] = {
                    'primary': primary_load,
                    'replicas': replica_load,
                    'total': total_weight
                }
            
            # Statistiche globali
            total_weight = sum(w['total'] for w in node_weights.values())
            max_weight = max(w['total'] for w in node_weights.values())
            min_weight = min(w['total'] for w in node_weights.values())
            avg_weight = total_weight / len(node_weights)
            
            # Stampa per nodo
            for node_name in sorted(all_nodes.keys()):
                weights = node_weights[node_name]
                weight_pct = (weights['total'] / total_weight * 100) if total_weight > 0 else 0
                deviation = weights['total'] - avg_weight
                status = "✓" if abs(deviation) < avg_weight * 0.2 else "⚠️"
                
                print(f"\n      {status} {node_name}:")
                print(f"         Primary: {weights['primary']:,} vectors")
                print(f"         Replicas: {weights['replicas']:,} vectors")
                print(f"         Total: {weights['total']:,} vectors ({weight_pct:.1f}%)")
                print(f"         Deviation from avg: {deviation:+,.0f} vectors")
            
            # Summary
            print(f"\n      📊 Weight Distribution Summary:")
            print(f"         Total weight: {total_weight:,} vectors")
            print(f"         Average weight: {avg_weight:,.0f} vectors/node")
            print(f"         Max weight: {max_weight:,} vectors")
            print(f"         Min weight: {min_weight:,} vectors")
            print(f"         Weight std dev: {np.std([w['total'] for w in node_weights.values()]):.1f}")
            print(f"         Weight imbalance: {(max_weight - min_weight) / avg_weight:.3f}")
            
        else:
            print(f"\n  ⚠️  No replication plan generated")
    
    # --- 7. SAMPLE DATA PER CLUSTER ---
    print("\n--- Step 7: Sample Sentences per Cluster ---")
    
    for cluster_id in range(min(3, N_CLUSTERS)):  # Mostra primi 3 cluster
        print(f"\n📂 Cluster {cluster_id} samples:")
        
        target_node = routing_table.get_nodes(str(cluster_id))[0]
        
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            
            # Controlla se il nodo ha vettori
            count = clients[target_node].count(collection_name=COLLECTION_NAME).count
            if count == 0:
                print(f"  ⚠️  No vectors in {target_node}, skipping samples")
                continue
            
            # Tenta scroll con retry e fallback
            try:
                samples = clients[target_node].scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="cluster_id", match=MatchValue(value=str(cluster_id)))
                    ]),
                    limit=5,  # RIDOTTO da 5 per sicurezza
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                if not samples:
                    print(f"  ⚠️  No vectors found for cluster {cluster_id} in {target_node}")
                    continue
                
                for i, sample in enumerate(samples, 1):
                    sentence = sample.payload.get("sentence", "N/A")
                    print(f"  {i}. {sentence[:100]}...")
                    
            except Exception as scroll_error:
                error_msg = str(scroll_error)
                if "OutputTooSmall" in error_msg or "500" in error_msg:
                    print(f"  ⚠️  Buffer overflow for cluster {cluster_id}, trying alternative method...")
                    
                    # Fallback: usa query invece di scroll (più sicuro)
                    try:
                        # Prendi sample random invece di scroll
                        all_points = clients[target_node].query_points(
                            collection_name=COLLECTION_NAME,
                            query=[0.1] * VECTOR_DIMENSION,  # Query dummy
                            query_filter=Filter(must=[
                                FieldCondition(key="cluster_id", match=MatchValue(value=str(cluster_id)))
                            ]),
                            limit=5,
                            with_payload=True
                        ).points
                        
                        if all_points:
                            for i, point in enumerate(all_points, 1):
                                sentence = point.payload.get("sentence", "N/A")
                                print(f"  {i}. {sentence[:100]}...")
                        else:
                            print(f"  ⚠️  No samples available via fallback method")
                            
                    except Exception as fallback_error:
                        print(f"  ⚠️  Fallback also failed: {str(fallback_error)[:100]}")
                else:
                    raise
                
        except Exception as e:
            print(f"  ⚠️  Error fetching samples for cluster {cluster_id}: {str(e)[:100]}")
            continue
    
    # --- 8. QUERY DEMO ---
    print("\n--- Step 8: Query Demonstration with Replica-Aware Routing ---")
    
    # Query con frase Wikipedia reale
    test_idx = 0
    test_vector = wikipedia_embeddings[test_idx]
    test_sentence = wikipedia_sentences[test_idx]
    
    print(f"\nTest query:")
    print(f"  Sentence: {test_sentence[:100]}...")
    
    # METODO 1: Routing tradizionale (K-means → primary node)
    cluster_id = predict_cluster(quantizer_centroids, test_vector)
    kmeans_node = routing_table.get_primary_node(str(cluster_id))
    print(f"\n  📍 K-means routing (primary only):")
    print(f"     Cluster: {cluster_id}")
    print(f"     Target node: {kmeans_node}")
    
    # METODO 2: Meta-HNSW routing (top-k cluster simili)
    nearest_nodes = meta_hnsw.find_nearest_nodes(test_vector, k=TOP_K_NODES_FOR_ROUTING)
    print(f"\n  🎯 Meta-HNSW routing (top-{TOP_K_NODES_FOR_ROUTING} clusters):")
    for i, (node, dist) in enumerate(nearest_nodes, 1):
        print(f"     {i}. {node} (distance: {dist:.4f})")
    
    # METODO 3 (NUOVO): Load-aware replica selection
    if ENABLE_REPLICATION:
        print(f"\n  🔄 Load-Aware Replica Selection:")
        
        replica_selector = LoadAwareReplicaSelector()
        
        # Simula carico nodi (in produzione: metriche real-time)
        simulated_node_loads = {name: node_stats[name]['load'] // 100 for name in all_nodes.keys()}
        
        # Per ogni cluster top-k, trova replica ottimale
        selected_nodes_with_replicas = []
        for node_name, dist in nearest_nodes:
            # Trova cluster di questo nodo (semplificazione)
            # In realtà: usa cluster_id da Meta-HNSW
            assigned_clusters = routing_table.get_clusters_for_node(node_name)
            
            if assigned_clusters:
                cluster_for_query = str(assigned_clusters[0])
                
                # Ottieni tutte le repliche
                all_replicas = routing_table.get_all_nodes_for_cluster(cluster_for_query)
                
                if len(all_replicas) > 1:
                    # Ha repliche: scegli migliore
                    best_node = replica_selector.select_best_replica(
                        cluster_for_query,
                        routing_table,
                        simulated_node_loads
                    )
                    
                    selected_nodes_with_replicas.append((
                        best_node,
                        cluster_for_query,
                        len(all_replicas),
                        simulated_node_loads.get(best_node, 0)
                    ))
        
        if selected_nodes_with_replicas:
            print(f"     Selected replicas:")
            for node, cluster, num_replicas, load in selected_nodes_with_replicas[:3]:
                print(f"       • {node} (cluster {cluster}, {num_replicas} replicas, load: {load})")
    
    # Query sui top-k nodi (usa repliche se disponibili)
    print(f"\n  🔍 Querying top-{len(nearest_nodes)} nodes...")
    all_results = []
    
    for node_name, _ in nearest_nodes:
        # ═══════════════════════════════════════════════════════════════
        # FIX: Aggiungi with_vectors=True per ottenere vettori nei risultati
        # ═══════════════════════════════════════════════════════════════
        results = clients[node_name].query_points(
            collection_name=COLLECTION_NAME,
            query=test_vector.tolist(),
            limit=5,
            with_vectors=True  # NUOVO: necessario per calcolo similarità manuale
        ).points
        
        for result in results:
            all_results.append((node_name, result))
    
    # Ordina per score (cosine similarity in Qdrant)
    all_results.sort(key=lambda x: x[1].score, reverse=True)
    
    if all_results:
        print(f"\n✓ Found {len(all_results)} total results from {len(nearest_nodes)} nodes:")
        
        print(f"\n  📊 Similarity Analysis:")
        print(f"      Query vector: {test_sentence[:60]}...")
        print(f"\n      Top-5 Results by Cosine Similarity:")
        
        for i, (node, result) in enumerate(all_results[:5], 1):
            result_sentence = result.payload.get("sentence", "N/A")
            result_cluster = result.payload.get("cluster_id", "N/A")
            
            # Qdrant score è già cosine similarity [0, 1]
            similarity_score = result.score
            
            # ═══════════════════════════════════════════════════════════════
            # FIX: Verifica che result.vector non sia None
            # ═══════════════════════════════════════════════════════════════
            
            if result.vector is not None:
                # Calcola cosine similarity manualmente per verifica
                result_vector = np.array(result.vector)
                query_norm = test_vector / (np.linalg.norm(test_vector) + 1e-8)
                result_norm = result_vector / (np.linalg.norm(result_vector) + 1e-8)
                manual_similarity = float(np.dot(query_norm, result_norm))
            else:
                manual_similarity = similarity_score  # Fallback su score Qdrant
            
            # Status indicator
            if similarity_score >= 0.9:
                status = "🔥"
            elif similarity_score >= 0.8:
                status = "✅"
            elif similarity_score >= 0.7:
                status = "📝"
            else:
                status = "⚠️"
            
            print(f"\n      {status} Result #{i} [from {node}, cluster {result_cluster}]:")
            print(f"         Qdrant score: {similarity_score:.6f}")
            print(f"         Manual cosine: {manual_similarity:.6f}")
            print(f"         Distance: {1 - similarity_score:.6f}")
            print(f"         Text: {result_sentence[:80]}...")
            
            # Interpretazione similarità
            if similarity_score >= 0.95:
                interpretation = "Nearly identical"
            elif similarity_score >= 0.85:
                interpretation = "Highly similar"
            elif similarity_score >= 0.75:
                interpretation = "Moderately similar"
            elif similarity_score >= 0.60:
                interpretation = "Somewhat related"
            else:
                interpretation = "Weakly related"
            
            print(f"         Interpretation: {interpretation}")
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Statistiche similarità per nodo
        # ═══════════════════════════════════════════════════════════════
        
        print(f"\n  📈 Similarity Statistics by Node:")
        
        results_by_node = {}
        for node, result in all_results:
            if node not in results_by_node:
                results_by_node[node] = []
            results_by_node[node].append(result.score)
        
        for node_name in sorted(results_by_node.keys()):
            scores = results_by_node[node_name]
            avg_sim = np.mean(scores)
            max_sim = np.max(scores)
            min_sim = np.min(scores)
            
            print(f"\n      {node_name}:")
            print(f"        Results returned: {len(scores)}")
            print(f"        Avg similarity: {avg_sim:.4f}")
            print(f"        Max similarity: {max_sim:.4f}")
            print(f"        Min similarity: {min_sim:.4f}")
            print(f"        Std deviation: {np.std(scores):.4f}")
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Confronto similarità tra nodi
        # ═══════════════════════════════════════════════════════════════
        
        print(f"\n  🎯 Node Comparison:")
        
        if len(results_by_node) > 1:
            best_node = max(results_by_node.items(), key=lambda x: np.mean(x[1]))
            worst_node = min(results_by_node.items(), key=lambda x: np.mean(x[1]))
            
            print(f"      Best node: {best_node[0]} (avg similarity: {np.mean(best_node[1]):.4f})")
            print(f"      Worst node: {worst_node[0]} (avg similarity: {np.mean(worst_node[1]):.4f})")
            print(f"      Similarity gap: {np.mean(best_node[1]) - np.mean(worst_node[1]):.4f}")
            
            # Verifica se Meta-HNSW ha scelto bene
            meta_hnsw_nodes = [node for node, _ in nearest_nodes]
            if best_node[0] in meta_hnsw_nodes:
                print(f"      ✅ Meta-HNSW correctly identified best node!")
            else:
                print(f"      ⚠️  Meta-HNSW missed best node (would need larger k)")
    
    else:
        print("⚠️  No results found")
    
    print(f"\n  ⚡ Performance Comparison:")
    print(f"     K-means: query 1 node ({kmeans_node})")
    print(f"     Meta-HNSW: query {len(nearest_nodes)} nodes (potentially better recall)")
    
    # --- 9. DETAILED NODE STATISTICS ---
    print("\n--- Step 9: Detailed Node Statistics ---")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE NODE REPORT")
    print("="*70)
    '''
    for node_name, stats in sorted(node_stats.items()):
        load_pct = (stats['load'] / total_vectors * 100) if total_vectors > 0 else 0
        balance_status = "✓ BALANCED" if abs(stats['load'] - avg_load) < avg_load * 0.2 else "⚠️  UNBALANCED"
        
        print(f"\n{'─'*70}")
        print(f"📊 NODE: {node_name}")
        print(f"{'─'*70}")
        
        print(f"\n  📈 Load Information:")
        print(f"    • Total vectors: {stats['load']:,}")
        print(f"    • Percentage of total: {load_pct:.2f}%")
        print(f"    • Deviation from average: {stats['load'] - avg_load:+.1f} vectors")
        print(f"    • Status: {balance_status}")
        
        print(f"\n  🎯 Cluster Assignment:")
        print(f"    • Number of assigned clusters: {stats['n_clusters']}")
        print(f"    • Cluster IDs: {', '.join(stats['assigned_clusters'])}")
        
        # Dettagli carico per cluster
        print(f"\n  📑 Per-Cluster Breakdown:")
        cluster_details = []
        for cid in stats['assigned_clusters']:
            cid_load = cluster_counts.get(cid, 0)
            cluster_pct = (cid_load / stats['load'] * 100) if stats['load'] > 0 else 0
            cluster_details.append((cid, cid_load, cluster_pct))
        
        for cid, cid_load, cluster_pct in sorted(cluster_details, key=lambda x: int(x[0])):
            print(f"    • Cluster {cid}: {cid_load:,} vectors ({cluster_pct:.1f}% of node)")
        
        print(f"\n  🔢 Centroid Statistics:")
        print(f"    • Mean centroid magnitude: {np.linalg.norm(stats['centroid']):.6f}")
        print(f"    • Centroid dimensions (first 10): {stats['centroid'][:10]}")
        print(f"    • Min value: {np.min(stats['centroid']):.6f}")
        print(f"    • Max value: {np.max(stats['centroid']):.6f}")
        print(f"    • Mean value: {np.mean(stats['centroid']):.6f}")
    '''
    print(f"\n{'─'*70}")
    print("📊 GLOBAL STATISTICS")
    print(f"{'─'*70}")
    
    print(f"\n  🌐 Overall System State:")
    print(f"    • Total nodes: {len(node_stats)}")
    print(f"    • Total vectors across all nodes: {total_vectors:,}")
    print(f"    • Total clusters: {N_CLUSTERS}")
    print(f"    • Average vectors per node: {avg_load:.2f}")
    print(f"    • Average clusters per node: {np.mean([s['n_clusters'] for s in node_stats.values()]):.2f}")
    
    print(f"\n  ⚖️  Load Balance Metrics:")
    print(f"    • Maximum load: {max_load:,} ({max_load/total_vectors*100:.2f}%)")
    print(f"    • Minimum load: {min_load:,} ({min_load/total_vectors*100:.2f}%)")
    print(f"    • Standard deviation: {std_dev:.2f}")
    print(f"    • Imbalance ratio: {imbalance_ratio:.4f}")
    
    # Calcola metriche aggiuntive
    variance = np.var([stat['load'] for stat in node_stats.values()])
    cv = std_dev / avg_load if avg_load > 0 else 0
    
    print(f"    • Variance: {variance:.2f}")
    print(f"    • Coefficient of variation: {cv:.4f}")
    
    # Raccomandazioni
    print(f"\n  💡 Recommendations:")
    if imbalance_ratio < 0.2:
        print(f"    ✓ System is well-balanced - no rebalancing needed")
    elif imbalance_ratio < 0.5:
        print(f"    ⚠️  Minor imbalance detected - consider monitoring")
    else:
        print(f"    ⚠️  Significant imbalance detected - rebalancing recommended")
    
    if cv > 0.3:
        print(f"    ⚠️  High coefficient of variation - uneven cluster distribution")
    else:
        print(f"    ✓ Cluster distribution is relatively uniform")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Leggi numero di nodi da argomento da riga di comando
    num_nodes = None
    if len(sys.argv) > 1:
        try:
            num_nodes = int(sys.argv[1])
            print(f"🔧 Running with {num_nodes} nodes")
        except ValueError:
            print(f"⚠️  Invalid argument: {sys.argv[1]}. Expected integer.")
            sys.exit(1)
    
    main(num_nodes=num_nodes)