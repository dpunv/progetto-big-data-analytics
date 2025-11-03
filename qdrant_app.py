import requests
import numpy as np
import uuid
import time
import json
import sys
import os
import utils
from concurrent.futures import ThreadPoolExecutor, as_completed  # NEW IMPORT
from threading import Lock  # NEW IMPORT

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
BATCH_TIERS = [800, 256, 64]  # Predefined fallback tiers
                              # provo prima con batch size 800, poi 256, poi 64
BATCH_SIZE_OPTIMAL = BATCH_TIERS[0]  # Start with largest
BATCH_SIZE_MIN = BATCH_TIERS[-1]     # Never go below smallest
MAX_RETRIES = len(BATCH_TIERS)       # One retry per tier
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
NODE_VECS = []  # This will contain our centroids

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
        self.failure_count = 0  # Reset failure counter
        
        # After 20 consecutive successes, try increasing batch size
        if self.success_count >= 20 and self.current_batch_size < BATCH_SIZE_OPTIMAL:
            old_size = self.current_batch_size
            self.current_batch_size = min(int(self.current_batch_size * 1.5), BATCH_SIZE_OPTIMAL)
            print(f"📈 Increasing batch size: {old_size} → {self.current_batch_size} (after {self.success_count} successes)")
            self.success_count = 0
    
    def on_failure(self):
        """Called when a batch fails."""
        self.failure_count += 1
        self.success_count = 0  # Reset success counter
        
        # After 2 consecutive failures, reduce batch size preventively
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
# ---

# --- NEW: Dynamic Timeout Calculator ---
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

            # set_node_vectors expects a list of vectors; send list with single centroid
            r_set = requests.post(f"{node_url}/set_node_vectors", json=[node_vec], timeout=5)
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
                    "node_vectors": [peer_vec]
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


def run_queries():
    """Run a single federated search query."""
    print("--- 4. Running Federated Search Query ---")
    
    # Use the last embedding as the query vector
    query_vector = data[NUM_VECTORS]['embedding']
    # print(f"Query Vector (first 5 dims): {[round(x, 2) for x in query_vector[:5]]}...\n")

    try:
        # Run Federated Search (always from Node 1, doesn't matter which)
        r_fed = requests.post(
            f"{NODE_URLS[0]}/search/federated",
            json=query_vector,
            params={"top_k": 5} # Request top 5 from EACH node
        )
        r_fed.raise_for_status()
        fed_results = r_fed.json()
        
        print(f"Federated search complete:")
        results_data = fed_results.get('results', {})
        
        all_results = []
        best_overall_node = "N/A"
        best_overall_score = -2.0 # Cosine similarity can be -1
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_vector = NODE_VECS[i]
            node_similarity = utils.cosine_similarity(query_vector, node_vector)
            node_results_list = results_data.get(node_id, [])
            node_results_count = len(node_results_list)
            
            max_score = node_results_list[0].get('score', -1) if node_results_count > 0 else -1
            min_score = node_results_list[-1].get('score', -1) if node_results_count > 0 else -1
            
            print(f"  - Results from {node_id}: {node_results_count} (Best: {max_score:.4f}, Worst: {min_score:.4f}, Node: {node_similarity})")
            
            if max_score > best_overall_score:
                best_overall_score = max_score
                best_overall_node = node_id
        
        print(f"\n  - Total Results Found: {fed_results.get('total_results')}")
        print(f"ℹ️  {best_overall_node} had the best matching vector (Score: {best_overall_score:.4f}).")

    except requests.exceptions.RequestException as e:
        print(f"!!! Error running federated query: {e}")


def main_app():
    print(f"Starting Qdrant Smart Sharding Test ({NUM_NODES} Nodes)...")
    print(f"Please ensure all servers and DBs are running.")
    print(f"Test will insert {NUM_VECTORS} vectors.\n")
    
    # Give servers a moment to start
    time.sleep(2)
    
    start_time = time.time()
    
    register_peers()
    
    # --- MODIFIED: Call the bulk function ---
    insertions = insert_vectors_bulk()
    # --- END MODIFICATION ---
    
    # --- NEW: Add a delay to allow background tasks to finish ---
    print("--- Waiting 5s for background insertions to settle... ---")
    time.sleep(5) 
    # In a real system, you might need a more robust check
    # --- END NEW ---
    
    check_counts(insertions)
    run_queries()
    
    end_time = time.time()
    print(f"\n--- Test Complete in {end_time - start_time:.2f} seconds ---")

# --- Main Execution ---
if __name__ == "__main__":
    main_app()