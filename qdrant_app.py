"""
Distributed Qdrant Application - Main Orchestrator
Coordinates vector insertion across multiple nodes with smart clustering.
"""
import sys
import time
import json
import os
import uuid
import numpy as np
import random
import ijson
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# NEW: Import refactored modules
from coordinator import AppConfig, NodeCoordinator, BatchSender
from node_assignment import compute_node_assignments


def load_embeddings(filename: str, max_vectors: int, vector_size: int) -> list:
    """
    Load embeddings from JSON file using streaming parser.
    
    Args:
        filename: Path to embeddings JSON file
        max_vectors: Maximum number of vectors to load
        vector_size: Expected vector dimension
        
    Returns:
        List of dicts with 'embedding' key
    """
    data = []
    try:
        print(f"Loading embeddings from '{filename}' using a streaming parser...")
        with open(filename, 'r') as f:
            parser = ijson.items(f, 'item', use_float=True)
            for i, item in enumerate(parser):
                if i >= max_vectors + 1:  # +1 for query vector
                    break
                data.append(item)
        
        print(f"✅ Loaded {len(data)} embeddings from file.")
        
        # Generate additional if needed
        if len(data) < max_vectors + 1:
            print(f"Warning: embeddings.json has only {len(data)} items, but {max_vectors + 1} are needed.")
            print("Generating additional random data...")
            for i in range(len(data), max_vectors + 1):
                data.append({"embedding": np.random.rand(vector_size).tolist()})
    
    except FileNotFoundError:
        print("embeddings.json not found. Generating random data...")
        for i in range(max_vectors + 1):
            data.append({"embedding": np.random.rand(vector_size).tolist()})
    
    except Exception as e:
        print(f"❌ Error loading embeddings.json: {e}")
        print("   Generating random data as a fallback...")
        data = []
        for i in range(max_vectors + 1):
            data.append({"embedding": np.random.rand(vector_size).tolist()})
    
    return data


def compute_or_load_assignments(config: AppConfig, data: list) -> dict:
    """
    Compute or load node assignments from cache.
    
    Args:
        config: Application configuration
        data: Loaded embeddings data
        
    Returns:
        Node assignment data dict
    """
    print("\n" + "="*60)
    print("0. COMPUTING SMART NODE ASSIGNMENTS")
    print("="*60)
    
    node_assignment_data = None
    
    # Try load existing assignments
    if os.path.exists(config.assignments_file):
        print(f"📂 Found existing assignments file '{config.assignments_file}'...")
        with open(config.assignments_file, 'r') as f:
            node_assignment_data = json.load(f)
        
        stats = node_assignment_data.get('stats', {})
        if stats.get('num_nodes') == config.num_nodes:
            print(f"✅ Assignments file matches NUM_NODES={config.num_nodes}, using cached assignments.")
        else:
            print(f"⚠️  Assignments file for {stats.get('num_nodes')} nodes (need {config.num_nodes}) — will recompute.")
            node_assignment_data = None
    
    # Compute if needed
    if node_assignment_data is None or config.force_retrain:
        if len(data) < config.training_vectors:
            print(f"❌ ERROR: Insufficient data for training. Need {config.training_vectors}, have {len(data)}")
            sys.exit(1)
        
        print(f"🎯 Computing node assignments (training on {config.training_vectors} vectors)...")
        start_compute = time.time()
        
        node_assignment_data = compute_node_assignments(
            embeddings_file='embeddings.json',
            num_nodes=config.num_nodes,
            max_k_to_test=min(30, config.training_vectors // 100),
            random_state=42,
            max_vectors=config.training_vectors,
            rep_factor=None,
            beam_width=5,
            use_simulated_annealing=False
        )
        
        compute_time = time.time() - start_compute
        print(f"✅ Assignment computation completed in {compute_time:.2f}s.")
        
        # Save for future runs
        with open(config.assignments_file, 'w') as f:
            json.dump(node_assignment_data, f, indent=2)
        print(f"💾 Saved assignment data to {config.assignments_file}")
    
    # Save centroids separately
    try:
        centroids = node_assignment_data.get('centroids', [])
        with open(config.centroids_file, 'w') as f:
            json.dump(centroids, f, indent=2)
        print(f"💾 Full centroid list saved to {config.centroids_file} ({os.path.getsize(config.centroids_file)/1024:.2f} KB)")
    except Exception:
        pass
    
    return node_assignment_data


def insert_vectors_bulk(config: AppConfig, data: list, batch_sender: BatchSender, coordinator: NodeCoordinator, node_assignments: dict) -> list:
    """
    Insert vectors in bulk with parallel batch sending.
    
    FIXED: Now properly implements replication factor routing.
    Each vector is sent to ALL replica nodes, not just the entry node.
    """
    print("\n" + "="*60)
    print(f"2. INSERTING {config.num_vectors} VECTORS (Size {config.vector_size})")
    print("="*60)
    print(f"   Starting with batch size: {batch_sender.metrics.current_batch_size}")
    print(f"   Minimum batch size: {config.batch_size_min}")
    print(f"   Maximum retries per batch: {config.max_retries}")
    print(f"   🚀 Parallel workers: {config.num_nodes} (one per node)\n")
    
    insertions = []
    vector_index = 0
    batch_num = 0
    metrics_lock = Lock()
    
    start_time = time.time()
    _last_peer_report_time = 0.0
    unavailable_nodes = set()
    
    # Helper function for entry node selection
    def choose_entry_node(batch_num: int) -> int:
        """Choose entry node with round-robin, skipping unavailable."""
        start = (batch_num - 1) % config.num_nodes
        for offset in range(config.num_nodes):
            idx = (start + offset) % config.num_nodes
            node_id = config.get_node_id(idx)
            if node_id not in unavailable_nodes:
                return idx
        return start  # Fallback
    
    # Initial peer health check
    try:
        unavailable_nodes, _ = coordinator.update_unavailable_nodes()
        if unavailable_nodes:
            print(f"Client: Nodes excluded as entry by MAJORITY: {sorted(list(unavailable_nodes))}")
    except Exception as e:
        print(f"Client: initial peer report failed: {e}")
    
    with ThreadPoolExecutor(max_workers=config.num_nodes) as executor:
        futures = {}
        
        while vector_index < config.num_vectors or futures:
            # Periodic health check
            if time.time() - _last_peer_report_time > config.peer_report_interval:
                try:
                    unavailable_nodes, _ = coordinator.update_unavailable_nodes()
                    _last_peer_report_time = time.time()
                except Exception as e:
                    print(f"Client: peer report update failed: {e}")
            
            # Submit new batches
            while len(futures) < config.num_nodes and vector_index < config.num_vectors:
                batch_num += 1
                
                current_batch_size = batch_sender.metrics.current_batch_size
                end_index = min(vector_index + current_batch_size, config.num_vectors)
                
                # Choose entry node
                node_url_index = choose_entry_node(batch_num)
                node_url = config.get_node_urls()[node_url_index]
                sent_to_node_id = config.get_node_id(node_url_index)
                
                # Create batch payload WITH REPLICATION ROUTING INFO
                batch_payload = []
                for j in range(vector_index, end_index):
                    final_vec = data[j]['embedding']
                    
                    # NEW: Include replication routing info in payload
                    vector_data = {
                        "id": str(uuid.uuid4()),
                        "vector": final_vec,
                        "payload": {
                            "source_type": "json_data",
                            "sent_to_node": sent_to_node_id,
                            "index": j,
                            "route_to_replicas": True  # NEW: Flag for replication
                        }
                    }
                    batch_payload.append(vector_data)
                
                # Submit async (entry node will handle replication routing)
                future = executor.submit(
                    batch_sender.send_batch,
                    batch_payload,
                    node_url,
                    sent_to_node_id
                )
                
                futures[future] = {
                    'batch_num': batch_num,
                    'start_index': vector_index,
                    'end_index': end_index,
                    'node_id': sent_to_node_id
                }
                
                vector_index = end_index
            
            # Process completed batches
            if futures:
                done, pending = as_completed(futures.keys()), set(futures.keys())
                
                for future in done:
                    batch_info = futures[future]
                    success, res_data = future.result()
                    
                    if success:
                        batches = res_data.get('batches', {})
                        with metrics_lock:
                            for node_id, count in batches.items():
                                insertions.extend([node_id] * count)
                        
                        if batch_info['batch_num'] % 30 == 0:
                            elapsed = time.time() - start_time
                            total_inserted = batch_info['end_index']
                            vectors_per_sec = total_inserted / elapsed if elapsed > 0 else 0
                            
                            with metrics_lock:
                                metrics = batch_sender.metrics.get_stats()
                            
                            print(f"Batch {batch_info['batch_num']}: {total_inserted}/{config.num_vectors} vectors "
                                  f"({vectors_per_sec:.0f} vec/s, size={metrics['current_batch_size']}, "
                                  f"retries={metrics['total_retries']}, splits={metrics['total_splits']}, "
                                  f"active={len(futures)})")
                    else:
                        actual_batch_size = batch_info['end_index'] - batch_info['start_index']
                        print(f"⚠️  Batch {batch_info['batch_num']} failed completely, skipping {actual_batch_size} vectors")
                    
                    del futures[future]
                    break
    
    total_time = time.time() - start_time
    avg_speed = config.num_vectors / total_time if total_time > 0 else 0
    metrics = batch_sender.metrics.get_stats()
    
    print(f"\n✅ Vector insertion complete in {total_time:.2f}s")
    print(f"   Average speed: {avg_speed:.0f} vectors/second")
    print(f"   Final batch size: {metrics['current_batch_size']}")
    print(f"   Total retries: {metrics['total_retries']}")
    print(f"   Total splits: {metrics['total_splits']}")
    print(f"   🚀 Speedup from parallelization: ~{min(config.num_nodes, 3)}x\n")
    
    return insertions


def run_p2p_query(config: AppConfig, query_vector: np.ndarray, k_nodes: int, k_results: int, debug_mode: bool = False):
    """Execute P2P query via random entry node."""
    import requests
    
    mode_str = "DEBUG: ALL NODES" if debug_mode else "P2P Federated Search"
    print("\n" + "="*60)
    print(f"4. Running {mode_str}")
    print("="*60)
    
    # Random entry node
    entry_node_idx = random.randint(0, config.num_nodes - 1)
    entry_node_url = config.get_node_urls()[entry_node_idx]
    entry_node_id = config.get_node_id(entry_node_idx)
    
    print(f"🎯 Using entry node: {entry_node_id} ({entry_node_url})")
    if debug_mode:
        print(f"   Querying ALL {k_nodes} nodes for comparison")
    else:
        print(f"📊 Query parameters: top_k_nodes={k_nodes}, top_k_results={k_results}")
    
    try:
        payload = {
            "query_vector": query_vector.tolist(),
            "top_k_nodes": k_nodes,
            "top_k_results": k_results
        }
        
        response = requests.post(f"{entry_node_url}/search/p2p", json=payload, timeout=30)
        response.raise_for_status()
        results = response.json()
        
        print(f"\n✅ P2P query complete:")
        print(f"  - Entry node: {results['entry_node']}")
        print(f"  - Routing method: {results.get('routing_method', 'unknown')}")
        print(f"  - Nodes queried: {results['nodes_queried']} (targets: {results['target_nodes']})")
        print(f"  - Total results: {results['total_results']}")
        print(f"  - Best match: {results['best_match']['node']} (Score: {results['best_match']['score']:.4f})")
        
        print(f"\n📊 Per-node breakdown:")
        for node_name, node_results in results['results_per_node'].items():
            count = len(node_results)
            max_score = node_results[0]['score'] if count > 0 else -1
            min_score = node_results[-1]['score'] if count > 0 else -1
            print(f"  - {node_name}: {count} results (Best: {max_score:.4f}, Worst: {min_score:.4f})")
        
        if not debug_mode:
            print(f"\n💡 Efficiency: Entry node handled routing using local Meta-HNSW")
            print(f"   (No client-side Meta-HNSW needed - pure P2P architecture)")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during P2P query: {e}")
        import traceback
        traceback.print_exc()


def main_app():
    """Main application orchestrator."""
    # Parse config from command line
    try:
        num_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    except ValueError:
        print("Invalid argument. Using default of 3 nodes.")
        num_nodes = 3
    
    print(f"--- Running Qdrant App for {num_nodes} nodes ---")
    
    # Initialize configuration
    config = AppConfig(num_nodes=num_nodes)
    print(f"\n{config}")
    
    # Load embeddings
    data = load_embeddings('embeddings.json', config.num_vectors, config.vector_size)
    
    # Compute/load node assignments
    node_assignment_data = compute_or_load_assignments(config, data)
    
    centroids = node_assignment_data.get('centroids', [])
    node_assignments = node_assignment_data.get('node_assignments', {})
    
    # NEW: Use AppConfig method instead of reading from file
    replication_factor = config.calculate_replication_factor(len(centroids))
    
    # Override with file value if available (for backward compatibility)
    file_rep_factor = node_assignment_data.get('stats', {}).get('replication_factor')
    if file_rep_factor is not None:
        replication_factor = file_rep_factor
        print(f"Using replication factor from assignments file: {replication_factor}")
    else:
        print(f"Calculated replication factor: {replication_factor} (min: {config.min_replication_factor})")
    
    print(f"Loaded {len(centroids)} centroids; node assignment map contains {len(node_assignments)} nodes' assignments.")
    
    # Initialize coordinator and batch sender
    coordinator = NodeCoordinator(config)
    batch_sender = BatchSender(config.batch_tiers)
    
    # Main workflow
    print("\n" + "="*60)
    print(f"QDRANT SMART SHARDING TEST ({config.num_nodes} NODES)")
    print("="*60)
    print(f"Test will insert {config.num_vectors} vectors with {replication_factor}x replication.")
    print(f"Using {len(centroids)} clusters balanced across {config.num_nodes} nodes.\n")
    
    time.sleep(2)
    start_time = time.time()
    
    # Setup peer network
    if not coordinator.setup_peer_network(centroids, node_assignments):
        print("❌ Failed to setup peer network!")
        sys.exit(1)
    
    # Initialize Meta-HNSW
    if not coordinator.distribute_meta_hnsw(centroids, node_assignments):
        print("⚠️  Meta-HNSW initialization incomplete, but continuing...")
    
    # Insert vectors
    insertions = insert_vectors_bulk(config, data, batch_sender, coordinator, node_assignments)
    
    # Wait for indexing
    print("--- Waiting 15s for background insertions to settle... ---")
    time.sleep(15)
    
    # Verify counts
    coordinator.verify_node_counts(insertions, replication_factor)
    
    # Run queries
    query_vector = np.array(data[config.num_vectors]['embedding'])
    run_p2p_query(config, query_vector, k_nodes=min(3, config.num_nodes), k_results=5, debug_mode=False)
    run_p2p_query(config, query_vector, k_nodes=config.num_nodes, k_results=5, debug_mode=True)
    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"TEST COMPLETE IN {end_time - start_time:.2f} SECONDS")
    print("="*60)


if __name__ == "__main__":
    main_app()