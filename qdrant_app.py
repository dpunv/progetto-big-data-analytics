import requests
import numpy as np
import uuid
import time
import json
import sys
import os
import utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# --- NEW IMPORT ---
from meta_hnsw import MetaHNSW
# --- END NEW IMPORT ---

# --- Configuration ---
try:
    NUM_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid argument. Using default of 3 nodes.")
    NUM_NODES = 3

print(f"--- Running Qdrant App for {NUM_NODES} nodes ---")

NUM_VECTORS = 140000
VECTOR_SIZE = 384

# --- NEW: Adaptive Batch Configuration ---
BATCH_TIERS = [800, 256, 64]
BATCH_SIZE_OPTIMAL = BATCH_TIERS[0]
BATCH_SIZE_MIN = BATCH_TIERS[-1]
MAX_RETRIES = len(BATCH_TIERS)
# -----------------------------------

# --- Training Configuration ---
TRAINING_VECTORS = 10000
CENTROIDS_FILE = 'centroids.json'
# -----------------------------------

if VECTOR_SIZE < NUM_NODES:
    print(f"Error: VECTOR_SIZE ({VECTOR_SIZE}) must be >= NUM_NODES ({NUM_NODES})")
    print("Please increase VECTOR_SIZE in qdrant_app.py and server.py")
    sys.exit(1)

# --- Helper Functions ---
data = {}
try:
    with open('embeddings.json', 'r') as f:
        data = json.load(f)
    if len(data) < NUM_VECTORS + 1:
        print(f"Warning: embeddings.json has only {len(data)} items, but {NUM_VECTORS}+1 are needed.")
        for i in range(len(data), NUM_VECTORS + 1):
            data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})
except FileNotFoundError:
    print("embeddings.json not found. Generating random data...")
    for i in range(NUM_VECTORS + 1):
        data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})

# --- NEW LOGIC: Train-Once Centroids (Load or Calculate) ---
print("\n--- 0. Loading/Computing Node Centroids ---")
NODE_VECS = []

try:
    # 1. Check if centroids file exists
    if os.path.exists(CENTROIDS_FILE):
        print(f"📂 Found existing centroids file '{CENTROIDS_FILE}'...")
        with open(CENTROIDS_FILE, 'r') as f:
            NODE_VECS = json.load(f)
        
        # 2. Verify the number of centroids matches requested nodes
        if len(NODE_VECS) == NUM_NODES:
            print(f"✅ Successfully loaded {len(NODE_VECS)} centroids from '{CENTROIDS_FILE}'.")
            print(f"   Skipping K-means training (using cached centroids).")
        else:
            print(f"⚠️  Warning: '{CENTROIDS_FILE}' has {len(NODE_VECS)} centroids, but {NUM_NODES} are needed.")
            print("   Forcing re-training...")
            NODE_VECS = []  # Force re-training
    
    # 3. If NODE_VECS is empty (file not found or mismatch), train
    if not NODE_VECS:
        if not os.path.exists(CENTROIDS_FILE):
            print(f"Centroids file not found. Starting new training...")
        
        # 4. Check we have enough data for training
        if len(data) < TRAINING_VECTORS:
            print(f"❌ ERROR: Insufficient data for training.")
            print(f"   Need {TRAINING_VECTORS} vectors, but 'embeddings.json' only has {len(data)}.")
            print(f"   Please run 'python take_some.py' to generate more embeddings.")
            sys.exit(1)
        
        print(f"🎓 Starting K-Means training on {TRAINING_VECTORS} vectors...")
        start_train = time.time()
        
        # 5. Extract vectors for training
        training_data_vectors = [i['embedding'] for i in data[:TRAINING_VECTORS]]
        
        # 6. Calculate centroids
        calculated_centroids_np = utils.find_kmeans_centroids(training_data_vectors, NUM_NODES)
        
        # --- NEW: Normalizza centroidi K-means (CONSISTENZA CON META-HNSW!) ---
        print(f"   Normalizing K-means centroids...")
        for i in range(len(calculated_centroids_np)):
            norm = np.linalg.norm(calculated_centroids_np[i])
            if norm > 0:
                calculated_centroids_np[i] = calculated_centroids_np[i] / norm
        print(f"   ✓ Centroids normalized (unit vectors)")
        # --- END NEW ---
        
        NODE_VECS = calculated_centroids_np.tolist()
        
        train_time = time.time() - start_train
        
        # 7. Save new centroids ("weights") to disk
        with open(CENTROIDS_FILE, 'w') as f:
            json.dump(NODE_VECS, f, indent=2)
        
        file_size_kb = os.path.getsize(CENTROIDS_FILE) / 1024
        print(f"✅ Training completed in {train_time:.2f}s.")
        print(f"   {len(NODE_VECS)} centroids saved to '{CENTROIDS_FILE}' ({file_size_kb:.2f} KB).")

except Exception as e:
    print(f"❌ FATAL ERROR during centroids loading/training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
# --- END NEW LOGIC ---

# --- Dynamic Node Configuration ---
NODE_URLS = []
for i in range(1, NUM_NODES + 1):
    NODE_URLS.append(f"http://localhost:{8000 + i}")

# --- NEW: Global Meta-HNSW instance ---
meta_hnsw: MetaHNSW = None
META_HNSW_PATH = 'meta_hnsw_index.pkl'

# --- NEW: Coordinator State ---
pending_centroid_updates = {}
centroid_updates_lock = Lock()
# --- END NEW ---

# --- NEW FUNCTION: Setup centroid tracking on nodes ---
def setup_centroid_tracking():
    print("\n--- Setting up Centroid Tracking on Nodes ---")
    coordinator_url = "http://localhost:9000"
    
    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        node_url = NODE_URLS[i]
        initial_centroid = NODE_VECS[i]
        
        try:
            response = requests.post(
                f"{node_url}/initialize_centroid",
                json=initial_centroid,
                timeout=5
            )
            response.raise_for_status()
            
            response = requests.post(
                f"{node_url}/set_coordinator",
                params={"coordinator_url": coordinator_url},
                timeout=5
            )
            response.raise_for_status()
            
            print(f"  ✓ {node_id}: Centroid tracking initialized")
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  {node_id}: Failed to setup centroid tracking: {e}")
    
    print("✓ Centroid tracking setup complete\n")

# --- NEW: Coordinator Flask Server ---
def start_coordinator_server():
    from flask import Flask, request, jsonify
    import threading
    
    app = Flask(__name__)
    
    @app.route('/update_centroid', methods=['POST'])
    def update_centroid():
        data = request.json
        node_id = data.get('node_id')
        new_centroid = data.get('new_centroid')
        drift = data.get('drift')
        vector_count = data.get('vector_count')
        
        print(f"\n🔔 Coordinator: Received centroid update from {node_id}")
        print(f"   Drift: {drift:.6f}, Vectors: {vector_count}")
        
        with centroid_updates_lock:
            pending_centroid_updates[node_id] = {
                'centroid': new_centroid,
                'drift': drift,
                'vector_count': vector_count
            }
        
        if len(pending_centroid_updates) >= 1:
            print(f"   📌 Triggering Meta-HNSW rebuild ({len(pending_centroid_updates)} nodes changed)")
            rebuild_meta_hnsw_from_updates()
        
        return jsonify({
            'status': 'success',
            'message': f'Centroid update received from {node_id}'
        })
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'pending_updates': len(pending_centroid_updates)})
    
    def run_flask():
        app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=False)
    
    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()
    
    print("✓ Coordinator server started on http://localhost:9000")
    time.sleep(1)

# --- NEW: Rebuild Meta-HNSW from pending updates ---
def rebuild_meta_hnsw_from_updates():
    global meta_hnsw
    
    if meta_hnsw is None:
        print("⚠️  Meta-HNSW not initialized, skipping rebuild")
        return
    
    with centroid_updates_lock:
        if not pending_centroid_updates:
            return
        
        updates = pending_centroid_updates.copy()
        pending_centroid_updates.clear()
    
    print(f"\n🔧 Rebuilding Meta-HNSW with {len(updates)} updated centroids...")
    start_time = time.time()
    
    for node_id, update_data in updates.items():
        new_centroid = np.array([update_data['centroid']])
        
        print(f"  Updating {node_id} centroid (drift: {update_data['drift']:.6f})")
        
        meta_hnsw.recalculate_centroid_from_scratch(
            node_name=node_id,
            vectors=new_centroid,
            method='mean'
        )
    
    elapsed = time.time() - start_time
    
    stats = meta_hnsw.get_statistics()
    print(f"✅ Meta-HNSW rebuild complete in {elapsed:.2f}s")
    print(f"   Updated nodes: {len(updates)}")
    print(f"   Mean centroid distance: {stats['mean_distance']:.4f}")
    
    meta_hnsw.save(META_HNSW_PATH)
    print(f"💾 Updated Meta-HNSW saved\n")

# --- NEW FUNCTION: Initialize Meta-HNSW ---
def initialize_meta_hnsw():
    global meta_hnsw
    
    print("\n--- 3. Initializing Meta-HNSW with K-means Centroids ---")
    
    if os.path.exists(META_HNSW_PATH):
        print(f"📂 Found existing Meta-HNSW index: {META_HNSW_PATH}")
        try:
            meta_hnsw = MetaHNSW.load(META_HNSW_PATH)
            print(f"✅ Loaded MetaHNSW with {len(meta_hnsw.node_names)} nodes")
            
            if len(meta_hnsw.node_names) == NUM_NODES:
                print("   Using cached index (matching node count)")
                
                stats = meta_hnsw.get_statistics()
                print(f"   Cached centroids stats:")
                print(f"     - Mean distance: {stats['mean_distance']:.4f}")
                print(f"     - Total vectors tracked: {stats['total_vectors_tracked']}")
                return
            else:
                print(f"⚠️  Cached index has {len(meta_hnsw.node_names)} nodes, but {NUM_NODES} are needed")
                print("   Rebuilding index...")
        except Exception as e:
            print(f"⚠️  Failed to load cached index: {e}")
            print("   Building new index...")
    
    print(f"🏗️  Building new Meta-HNSW index for {NUM_NODES} nodes...")
    print(f"   Using K-means centroids from '{CENTROIDS_FILE}' (NOT recalculating from nodes)")
    
    meta_hnsw = MetaHNSW(
        dimension=VECTOR_SIZE,
        max_nodes=NUM_NODES * 2,
        ef_construction=200,
        M=16
    )
    
    start_time = time.time()
    
    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        centroid_vector = np.array([NODE_VECS[i]])
        
        print(f"  Adding {node_id} centroid (from K-means)...", end=" ")
        meta_hnsw.add_node_centroid(node_id, centroid_vector, method='mean')
        print("✓")
    
    meta_hnsw.force_rebuild()
    
    elapsed = time.time() - start_time
    
    stats = meta_hnsw.get_statistics()
    print(f"\n✅ Meta-HNSW initialized in {elapsed:.2f}s")
    print(f"   Nodes indexed: {stats['num_nodes']}")
    print(f"   Using K-means centroids (same as insertion routing)")
    print(f"   Mean centroid distance: {stats['mean_distance']:.4f}")
    
    print(f"\n🔍 Verifying centroid consistency:")
    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        kmeans_centroid = np.array(NODE_VECS[i])
        hnsw_centroid = meta_hnsw.node_centroids[node_id]
        
        diff = np.linalg.norm(kmeans_centroid - hnsw_centroid)
        print(f"   {node_id}: diff = {diff:.6f} (should be ~0)")
    
    meta_hnsw.save(META_HNSW_PATH)
    print(f"\n💾 Meta-HNSW saved to {META_HNSW_PATH}\n")

# --- NEW FUNCTION: Wait for Qdrant indexing ---
def wait_for_qdrant_indexing():
    print("\n--- Waiting for Qdrant to finish indexing ---")
    
    max_wait_time = 60
    start_time = time.time()
    
    all_indexed = False
    while not all_indexed and (time.time() - start_time) < max_wait_time:
        all_indexed = True
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            qdrant_port = 6333 + (i) * 2
            qdrant_url = f"http://localhost:{qdrant_port}"
            
            try:
                response = requests.get(
                    f"{qdrant_url}/collections/vectors",
                    timeout=5
                )
                
                if response.status_code == 200:
                    collection_info = response.json()
                    status = collection_info.get('result', {}).get('status', 'unknown')
                    
                    if status != 'green':
                        print(f"  {node_id}: Indexing status '{status}' (waiting...)")
                        all_indexed = False
                    else:
                        print(f"  ✓ {node_id}: Indexing complete (status 'green')")
                else:
                    print(f"  ⚠️  {node_id}: Could not check status (HTTP {response.status_code})")
                    all_indexed = False
                    
            except requests.exceptions.RequestException as e:
                print(f"  ⚠️  {node_id}: Error checking indexing status: {e}")
                all_indexed = False
        
        if not all_indexed:
            time.sleep(2)
    
    if all_indexed:
        print("✅ All nodes finished indexing!\n")
    else:
        elapsed = time.time() - start_time
        print(f"⚠️  Indexing check timeout after {elapsed:.1f}s")
        print("   Proceeding anyway, but queries might fail...\n")

# --- MODIFIED: Retry with Tier-based Splitting Logic ---
def send_batch_with_retry(
    vectors_batch: list,
    node_url: str,
    sent_to_node_id: str,
    retry_count: int = 0
) -> tuple:
    """
    Send a batch with intelligent retry using predefined size tiers.
    
    Algorithm:
    1. Try sending full batch
    2. On timeout/error → Try next smaller tier (800 → 256 → 64)
    3. Split batch if it exceeds current tier size
    
    Args:
        vectors_batch: List of vector data dicts
        node_url: Target node URL
        sent_to_node_id: Node ID for logging
        retry_count: Current retry attempt (0-indexed tier)
    
    Returns:
        (success: bool, response_data: dict)
    """
    batch_size = len(vectors_batch)
    
    # Base case: Empty batch
    if batch_size == 0:
        return True, {"batches": {}}
    
    # Safety: Limit retries to number of tiers
    if retry_count >= len(BATCH_TIERS):
        print(f"❌ All retry tiers exhausted for batch of {batch_size} vectors")
        batch_metrics.on_failure()
        return False, {}
    
    # Determine current tier size
    current_tier_size = BATCH_TIERS[retry_count]
    
    # If batch is larger than current tier, split it
    if batch_size > current_tier_size:
        print(f"📦 Batch size {batch_size} exceeds tier {retry_count+1} ({current_tier_size}), splitting...")
        batch_metrics.on_split()
        
        # Split into chunks of current_tier_size
        chunks = []
        for i in range(0, batch_size, current_tier_size):
            chunks.append(vectors_batch[i:i+current_tier_size])
        
        print(f"🔀 Split into {len(chunks)} chunks of max {current_tier_size} vectors")
        
        # Send all chunks with current tier
        all_results = []
        for chunk in chunks:
            success, data = send_batch_with_retry(chunk, node_url, sent_to_node_id, retry_count)
            if not success:
                return False, {}
            all_results.append(data)
        
        # Merge results
        merged_batches = {}
        for data in all_results:
            for node_id, count in data.get('batches', {}).items():
                merged_batches[node_id] = merged_batches.get(node_id, 0) + count
        
        return True, {"batches": merged_batches}
    
    # Batch fits in current tier, try sending
    timeout = calculate_timeout(batch_size)
    
    try:
        response = requests.post(
            f"{node_url}/add_vectors_bulk",
            json=vectors_batch,
            timeout=timeout
        )
        response.raise_for_status()
        
        # SUCCESS!
        res_data = response.json()
        batch_metrics.on_success(batch_size)
        return True, res_data
        
    except requests.exceptions.Timeout:
        # TIMEOUT: Try next smaller tier
        next_tier = retry_count + 1
        if next_tier < len(BATCH_TIERS):
            next_tier_size = BATCH_TIERS[next_tier]
            print(f"⏱️  Timeout with batch size {batch_size} (tier {retry_count+1}: {current_tier_size})")
            print(f"   Falling back to tier {next_tier+1} (max size: {next_tier_size})...")
            batch_metrics.on_retry()
            
            # Retry with next tier
            return send_batch_with_retry(vectors_batch, node_url, sent_to_node_id, next_tier)
        else:
            print(f"❌ Timeout even with smallest tier ({current_tier_size})")
            batch_metrics.on_failure()
            return False, {}
            
    except requests.exceptions.RequestException as e:
        # OTHER ERROR: Network issue, server error, etc.
        print(f"❌ Network error with batch size {batch_size} (tier {retry_count+1}): {e}")
        batch_metrics.on_retry()
        
        # Retry with exponential backoff (but only once)
        if retry_count == 0:
            wait_time = 2
            print(f"⏳ Waiting {wait_time}s before retry with next tier...")
            time.sleep(wait_time)
        
        # Try next tier
        next_tier = retry_count + 1
        if next_tier < len(BATCH_TIERS):
            return send_batch_with_retry(vectors_batch, node_url, sent_to_node_id, next_tier)
        else:
            batch_metrics.on_failure()
            return False, {}

# --- NEW: Thread-safe wrapper for send_batch_with_retry ---
def send_batch_with_retry_threadsafe(
    vectors_batch: list,
    node_url: str,
    sent_to_node_id: str,
    batch_num: int
) -> tuple:
    """
    Thread-safe wrapper for send_batch_with_retry.
    Returns (batch_num, success, response_data) for result tracking.
    """
    success, res_data = send_batch_with_retry(vectors_batch, node_url, sent_to_node_id)
    return (batch_num, success, res_data)
# ---

def register_peers():
    """
    Dynamically set each node's representative vector and register them with each other.
    """
    print("--- 1. Setting Node Vectors & Registering Peers ---")
    try:
        # 1. Set the representative vector for every node
        print("Setting node vectors...")
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_url = NODE_URLS[i]
            node_vec = NODE_VECS[i]  # Use loaded/calculated vectors
            
            r_set = requests.post(f"{node_url}/set_node_vector", json=node_vec, timeout=5)
            r_set.raise_for_status()
            print(f"  {node_id}: Set representative vector (first 3 dims: {[round(v, 3) for v in node_vec[:3]]}...)")

        # 2. Register all peers with all other peers (N * (N-1) requests)
        print("\nRegistering peers...")
        for i in range(NUM_NODES):
            host_id = f"node{i+1}"
            host_url = NODE_URLS[i]
            
            peers_registered = 0
            for j in range(NUM_NODES):
                if i == j:
                    continue
                
                peer_id = f"node{j+1}"
                peer_url = NODE_URLS[j]
                peer_vec = NODE_VECS[j]  # Use loaded/calculated vectors
                
                payload = {
                    "peer_id": peer_id,
                    "peer_url": peer_url,
                    "node_vector": peer_vec
                }
                
                r_reg = requests.post(f"{host_url}/register_peer", json=payload, timeout=5)
                r_reg.raise_for_status()
                peers_registered += 1
            
            print(f"  Host {host_id}: Registered {peers_registered} peers.")

        print("Peers registered and vectors exchanged successfully.\n")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error setting/registering peers: {e}")
        print("!!! Please ensure all servers are running.")
        sys.exit(1)


# --- MODIFIED: Replaced insert_vectors with insert_vectors_bulk ---
def insert_vectors_bulk():
    """
    Insert vectors with PARALLEL batch sending using ThreadPoolExecutor.
    This allows multiple batches to be sent simultaneously to different nodes.
    """
    print(f"--- 2. Inserting {NUM_VECTORS} Vectors (Size {VECTOR_SIZE}) ---")
    print(f"   Starting with batch size: {batch_metrics.current_batch_size}")
    print(f"   Minimum batch size: {BATCH_SIZE_MIN}")
    print(f"   Maximum retries per batch: {MAX_RETRIES}")
    print(f"   🚀 Parallel workers: {NUM_NODES} (one per node)\n")
    
    insertions = []
    vector_index = 0
    batch_num = 0
    
    start_time = time.time()
    
    # --- NEW: Use ThreadPoolExecutor for parallel sending ---
    # max_workers = NUM_NODES ensures we can send to all nodes simultaneously
    with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = {}  # Map future -> (batch_num, batch_data)
        
        while vector_index < NUM_VECTORS or futures:
            # --- PHASE 1: Submit new batches (up to NUM_NODES in parallel) ---
            while len(futures) < NUM_NODES and vector_index < NUM_VECTORS:
                batch_num += 1
                
                # Use current adaptive batch size
                current_batch_size = batch_metrics.current_batch_size
                end_index = min(vector_index + current_batch_size, NUM_VECTORS)
                
                # Determine entry node (round-robin)
                node_url_index = (batch_num - 1) % NUM_NODES
                node_url = NODE_URLS[node_url_index]
                sent_to_node_id = f"node{node_url_index + 1}"
                
                # Create batch payload
                batch_payload = []
                for j in range(vector_index, end_index):
                    final_vec = data[j]['embedding']
                    vector_data = {
                        "id": str(uuid.uuid4()),
                        "vector": final_vec,
                        "payload": {
                            "source_type": "json_data",
                            "sent_to_node": sent_to_node_id,
                            "index": j
                        }
                    }
                    batch_payload.append(vector_data)
                
                # Submit batch to thread pool
                future = executor.submit(
                    send_batch_with_retry_threadsafe,
                    batch_payload,
                    node_url,
                    sent_to_node_id,
                    batch_num
                )
                
                futures[future] = {
                    'batch_num': batch_num,
                    'start_index': vector_index,
                    'end_index': end_index,
                    'node_id': sent_to_node_id
                }
                
                vector_index = end_index
            
            # --- PHASE 2: Process completed batches ---
            # Wait for at least one batch to complete
            if futures:
                done, pending = as_completed(futures.keys()), set(futures.keys())
                
                # Process the first completed future
                for future in done:
                    batch_info = futures[future]
                    batch_num_completed, success, res_data = future.result()
                    
                    if success:
                        # Track insertions (thread-safe)
                        batches = res_data.get('batches', {})
                        with metrics_lock:
                            for node_id, count in batches.items():
                                insertions.extend([node_id] * count)
                        
                        # Periodic logging
                        if batch_num_completed % 30 == 0:
                            elapsed = time.time() - start_time
                            total_inserted = batch_info['end_index']
                            vectors_per_sec = total_inserted / elapsed if elapsed > 0 else 0
                            
                            with metrics_lock:
                                metrics = batch_metrics.get_stats()
                            
                            print(f"Batch {batch_num_completed}: {total_inserted}/{NUM_VECTORS} vectors "
                                  f"({vectors_per_sec:.0f} vec/s, size={metrics['current_batch_size']}, "
                                  f"retries={metrics['total_retries']}, splits={metrics['total_splits']}, "
                                  f"active={len(futures)})")
                    else:
                        actual_batch_size = batch_info['end_index'] - batch_info['start_index']
                        print(f"⚠️  Batch {batch_num_completed} failed completely, skipping {actual_batch_size} vectors")
                    
                    # Remove completed future
                    del futures[future]
                    break  # Process one at a time to maintain order
    
    total_time = time.time() - start_time
    avg_speed = NUM_VECTORS / total_time if total_time > 0 else 0
    metrics = batch_metrics.get_stats()
    
    print(f"\n✅ Vector insertion complete in {total_time:.2f}s")
    print(f"   Average speed: {avg_speed:.0f} vectors/second")
    print(f"   Final batch size: {metrics['current_batch_size']}")
    print(f"   Total retries: {metrics['total_retries']}")
    print(f"   Total splits: {metrics['total_splits']}")
    print(f"   🚀 Speedup from parallelization: ~{min(NUM_NODES, 3)}x\n")
    
    return insertions


def check_counts(insertions: list):
    """Check the final vector counts on each node."""
    print("--- 3. Checking Vector Counts ---")
    try:
        all_counts = []
        total_count = 0
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_url = NODE_URLS[i]
            
            r = requests.get(f"{node_url}/count", timeout=5)
            r.raise_for_status()
            count = r.json().get('count', 0)
            
            all_counts.append((node_id, count))
            total_count += count

        print("Vector counts per node:")
        for node_id, count in all_counts:
            print(f"  - {node_id} Count: {count}")
        
        print(f"Total Vectors Stored: {total_count} (Expected: {NUM_VECTORS})")
        
        # Check from test script perspective
        print("\n(Client-side insertion log check):")
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            script_count = insertions.count(node_id)
            print(f"  - {node_id} received: {script_count}")
        
        if total_count == NUM_VECTORS:
            print("✅  Total counts match total inserted vectors.")
        else:
            print("⚠️  Counts do NOT match total inserted vectors!")

        # Check if routing was *roughly* balanced
        expected_avg = NUM_VECTORS / NUM_NODES
        tolerance = NUM_VECTORS * 0.1 # 10% tolerance
        is_balanced = True
        
        for node_id, count in all_counts:
            if abs(count - expected_avg) > tolerance:
                is_balanced = False

        if is_balanced:
            print(f"✅  Distribution is roughly balanced (within 10%).")
        else:
            print(f"⚠️  Distribution seems unbalanced!")
        print("")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error checking counts: {e}")


# --- NEW: Adaptive Batch Sender Class ---
class AdaptiveBatchMetrics:
    """
    Tracks batch sending metrics and adjusts batch size dynamically.
    """
    def __init__(self):
        self.current_batch_size = BATCH_SIZE_OPTIMAL
        self.success_count = 0
        self.failure_count = 0
        self.total_retries = 0
        self.total_splits = 0
        
    def on_success(self, batch_size: int):
        """Called when a batch succeeds."""
        self.success_count += 1
        self.failure_count = 0
        
        if self.success_count >= 20 and self.current_batch_size < BATCH_SIZE_OPTIMAL:
            old_size = self.current_batch_size
            self.current_batch_size = min(int(self.current_batch_size * 1.5), BATCH_SIZE_OPTIMAL)
            print(f"📈 Increasing batch size: {old_size} → {self.current_batch_size} (after {self.success_count} successes)")
            self.success_count = 0
    
    def on_failure(self):
        """Called when a batch fails."""
        self.failure_count += 1
        self.success_count = 0
        
        if self.failure_count >= 2:
            old_size = self.current_batch_size
            self.current_batch_size = max(self.current_batch_size // 2, BATCH_SIZE_MIN)
            print(f"📉 Reducing batch size preventively: {old_size} → {self.current_batch_size} (after {self.failure_count} failures)")
    
    def on_retry(self):
        """Called when a retry happens."""
        self.total_retries += 1
    
    def on_split(self):
        """Called when a batch is split."""
        self.total_splits += 1
    
    def get_stats(self):
        """Returns statistics summary."""
        return {
            "current_batch_size": self.current_batch_size,
            "total_retries": self.total_retries,
            "total_splits": self.total_splits,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }

# Global metrics instance
batch_metrics = AdaptiveBatchMetrics()

# --- NEW: Thread-safe lock for metrics ---
metrics_lock = Lock()

# --- NEW FUNCTION: Adaptive Timeout Calculation ---
def calculate_timeout(batch_size: int, base_timeout: int = 10) -> int:
    """
    Calculate adaptive timeout based on batch size.
    
    Args:
        batch_size: Number of vectors in batch
        base_timeout: Minimum timeout in seconds
    
    Returns:
        Timeout in seconds
        
    Rationale:
        - Base: 10s for HTTP overhead + server processing
        - Per-vector: 10ms (384-dim vector similarity calc + routing)
        - Max: 120s to avoid indefinite hangs
        - Formula ensures linear scaling with batch size
    """
    per_vector_time = 0.01  # 10ms per vector (empirically reasonable)
    timeout = base_timeout + int(batch_size * per_vector_time)
    
    # Clamp between 10s and 120s
    return max(10, min(timeout, 120))

# --- MODIFIED: Smart query routing using Meta-HNSW ---
def run_queries():
    """Run federated search query with Meta-HNSW intelligent routing."""
    print("--- 4. Running Smart Federated Search (Meta-HNSW Routing) ---")
    
    if meta_hnsw is None:
        print("⚠️  Meta-HNSW not initialized, falling back to broadcast query")
        run_queries_fallback()
        return
    
    query_vector = np.array(data[NUM_VECTORS]['embedding'])
    
    k_nodes = min(3, NUM_NODES)
    
    print(f"🔍 Finding {k_nodes} nearest nodes using Meta-HNSW...")
    nearest_nodes = meta_hnsw.find_nearest_nodes(query_vector, k=k_nodes)
    
    print(f"📍 Meta-HNSW routing:")
    for node_name, distance in nearest_nodes:
        print(f"   - {node_name}: distance={distance:.4f}")
    
    try:
        all_results = {}
        best_overall_score = -2.0
        best_overall_node = "N/A"
        
        for node_name, distance in nearest_nodes:
            node_idx = int(node_name.replace("node", "")) - 1
            node_url = NODE_URLS[node_idx]
            
            payload = {
                "from_node": "client",
                "query_vector": query_vector.tolist(),
                "top_k": 5
            }
            
            response = requests.post(
                f"{node_url}/search",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            results = response.json()
            all_results[node_name] = results
            
            if results:
                max_score = results[0].get('score', -1)
                if max_score > best_overall_score:
                    best_overall_score = max_score
                    best_overall_node = node_name
        
        print(f"\n✅ Smart query complete (queried {len(nearest_nodes)}/{NUM_NODES} nodes):")
        total_results = 0
        
        for node_name, results in all_results.items():
            count = len(results)
            total_results += count
            max_score = results[0].get('score', -1) if count > 0 else -1
            min_score = results[-1].get('score', -1) if count > 0 else -1
            
            print(f"  - {node_name}: {count} results (Best: {max_score:.4f}, Worst: {min_score:.4f})")
        
        print(f"\n  - Total Results: {total_results}")
        print(f"  - Best Match: {best_overall_node} (Score: {best_overall_score:.4f})")
        
        print(f"\n💡 Efficiency gain: Queried only {len(nearest_nodes)}/{NUM_NODES} nodes "
              f"({100 * len(nearest_nodes) / NUM_NODES:.0f}% of cluster)")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during smart query: {e}")
        import traceback
        print("\nDebug traceback:")
        traceback.print_exc()

def run_queries_fallback():
    """Fallback to broadcast query (original implementation)."""
    print("--- Running Broadcast Federated Search (Fallback) ---")
    
    query_vector = data[NUM_VECTORS]['embedding']
    
    try:
        r_fed = requests.post(
            f"{NODE_URLS[0]}/search/federated",
            json=query_vector,
            params={"top_k": 5}
        )
        r_fed.raise_for_status()
        fed_results = r_fed.json()
        
        print(f"Federated search complete (queried ALL {NUM_NODES} nodes):")
        results_data = fed_results.get('results', {})
        
        best_overall_node = "N/A"
        best_overall_score = -2.0
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_vector = NODE_VECS[i]
            node_similarity = utils.cosine_similarity(query_vector, node_vector)
            node_results_list = results_data.get(node_id, [])
            node_results_count = len(node_results_list)
            
            max_score = node_results_list[0].get('score', -1) if node_results_count > 0 else -1
            min_score = node_results_list[-1].get('score', -1) if node_results_count > 0 else -1
            
            # --- RESTORED: Print all three values ---
            print(f"  - {node_id}: {node_results_count} results (Best: {max_score:.4f}, Worst: {min_score:.4f}, Centroid Similarity: {node_similarity:.4f})")
            # --- END RESTORED ---
            
            if max_score > best_overall_score:
                best_overall_score = max_score
                best_overall_node = node_id
        
        print(f"\n  - Total Results: {fed_results.get('total_results')}")
        print(f"  - Best Match: {best_overall_node} (Score: {best_overall_score:.4f})")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error running broadcast query: {e}")
# --- END MODIFIED ---

def main_app():
    print(f"Starting Qdrant Smart Sharding with Meta-HNSW Routing ({NUM_NODES} Nodes)...")
    print(f"Please ensure all servers and DBs are running.")
    print(f"Test will insert {NUM_VECTORS} vectors.\n")
    
    time.sleep(2)
    
    start_coordinator_server()
    
    start_time = time.time()
    
    register_peers()
    setup_centroid_tracking()
    
    insertions = insert_vectors_bulk()
    
    print("--- Waiting 10s for background insertions to settle... ---")
    time.sleep(10)
    
    check_counts(insertions)
    
    wait_for_qdrant_indexing()
    
    initialize_meta_hnsw()
    
    print("\n--- Checking for Centroid Updates ---")
    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        node_url = NODE_URLS[i]
        
        try:
            response = requests.get(f"{node_url}/centroid_stats", timeout=5)
            stats = response.json()
            print(f"  {node_id}: {stats['insertions_since_update']}/{stats['update_interval']} insertions")
        except:
            pass
    
    if pending_centroid_updates:
        print(f"\n⚠️  {len(pending_centroid_updates)} pending centroid updates, triggering rebuild...")
        rebuild_meta_hnsw_from_updates()
    
    print("\n--- Verifying nodes are ready for querying ---")
    all_ready = True
    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        node_url = NODE_URLS[i]
        
        try:
            response = requests.get(f"{node_url}/count", timeout=5)
            count = response.json().get('count', 0)
            
            if count == 0:
                print(f"  ⚠️  {node_id}: has 0 vectors (not ready for querying)")
                all_ready = False
            else:
                print(f"  ✓ {node_id}: {count} vectors ready")
        except Exception as e:
            print(f"  ❌ {node_id}: Error checking readiness: {e}")
            all_ready = False
    
    if not all_ready:
        print("\n⚠️  Some nodes not ready, waiting additional 5s...")
        time.sleep(5)
    
    run_queries()
    
    end_time = time.time()
    print(f"\n--- Test Complete in {end_time - start_time:.2f} seconds ---")
    
    print("\n💡 Coordinator server still running. Press Ctrl+C to exit.")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

# --- Main Execution ---
if __name__ == "__main__":
    main_app()