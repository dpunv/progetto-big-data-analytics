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

# NEW: Import the node assignment computation
try:
    from node_assignment import compute_node_assignments
except ImportError:
    print("Error: node_assignment.py not found in the same directory.")
    sys.exit(1)

# --- Configuration ---
try:
    NUM_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid argument. Using default of 3 nodes.")
    NUM_NODES = 3

print(f"--- Running Qdrant App for {NUM_NODES} nodes ---")

NUM_VECTORS = 140000
VECTOR_SIZE = 384

# --- Adaptive Batch Configuration ---
BATCH_TIERS = [800, 256, 64]
BATCH_SIZE_OPTIMAL = BATCH_TIERS[0]
BATCH_SIZE_MIN = BATCH_TIERS[-1]
MAX_RETRIES = len(BATCH_TIERS)

# --- Training Configuration ---
TRAINING_VECTORS = 10000
ASSIGNMENTS_FILE = 'node_assignments.json'
FORCE_RETRAIN = False  # Set to True to force recomputation

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

# --- NEW: Smart Node Assignment with Multiple Centroids per Node ---
print("\n" + "="*60)
print("0. COMPUTING SMART NODE ASSIGNMENTS")
print("="*60)

node_assignment_data = None

try:
    # 1. Check if assignment file exists and is valid
    if os.path.exists(ASSIGNMENTS_FILE) and not FORCE_RETRAIN:
        print(f"📂 Found existing assignments file '{ASSIGNMENTS_FILE}'...")
        with open(ASSIGNMENTS_FILE, 'r') as f:
            node_assignment_data = json.load(f)
        
        # Verify it matches current configuration
        stats = node_assignment_data.get('stats', {})
        if stats.get('num_nodes') == NUM_NODES:
            print(f"✅ Successfully loaded assignments for {NUM_NODES} nodes.")
            print(f"   - Number of clusters: {stats.get('num_clusters')}")
            print(f"   - Replication factor: {stats.get('replication_factor')}")
            print(f"   Skipping computation (using cached assignments).")
        else:
            print(f"⚠️  Warning: '{ASSIGNMENTS_FILE}' is for {stats.get('num_nodes')} nodes, but {NUM_NODES} are needed.")
            print("   Forcing recomputation...")
            node_assignment_data = None
    
    # 2. If no valid data, compute assignments
    if node_assignment_data is None:
        if not os.path.exists(ASSIGNMENTS_FILE):
            print(f"Assignments file not found. Starting computation...")
        
        # Check we have enough data
        if len(data) < TRAINING_VECTORS:
            print(f"❌ ERROR: Insufficient data for training.")
            print(f"   Need {TRAINING_VECTORS} vectors, but 'embeddings.json' only has {len(data)}.")
            sys.exit(1)
        
        print(f"🎯 Starting smart clustering and load balancing...")
        print(f"   Training on {TRAINING_VECTORS} vectors")
        print(f"   Target: {NUM_NODES} nodes")
        
        start_compute = time.time()
        
        # 3. Run the complete pipeline
        node_assignment_data = compute_node_assignments(
            embeddings_file='embeddings.json',
            num_nodes=NUM_NODES,
            max_k_to_test=min(30, TRAINING_VECTORS // 100),  # Sensible default
            random_state=42,
            max_vectors=TRAINING_VECTORS,
            rep_factor=None,  # Auto-calculate
            beam_width=5,
            use_simulated_annealing=False
        )
        
        compute_time = time.time() - start_compute
        print(f"\n✅ Assignment computation completed in {compute_time:.2f}s.")

except Exception as e:
    print(f"❌ FATAL ERROR during node assignment computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Extract the computed data
CENTROIDS = node_assignment_data['centroids']
NODE_ASSIGNMENTS = node_assignment_data['node_assignments']
ASSIGNMENT_DETAILS = node_assignment_data['assignment_details']
REPLICATION_FACTOR = node_assignment_data['stats']['replication_factor']

print("\n" + "="*60)
print("ASSIGNMENT SUMMARY")
print("="*60)
print(f"Total Clusters: {len(CENTROIDS)}")
print(f"Replication Factor: {REPLICATION_FACTOR}")
print(f"Nodes: {NUM_NODES}")
print("\nClusters per node:")
for node_idx in range(NUM_NODES):
    cluster_indices = NODE_ASSIGNMENTS.get(str(node_idx), NODE_ASSIGNMENTS.get(node_idx, []))
    print(f"  Node {node_idx+1}: {len(cluster_indices)} clusters -> {cluster_indices}")

# --- END NEW LOGIC ---

# --- Dynamic Node Configuration ---
NODE_URLS = []
for i in range(1, NUM_NODES + 1):
    NODE_URLS.append(f"http://localhost:{8000 + i}")

# --- Adaptive Batch Metrics Class ---
class AdaptiveBatchMetrics:
    """Tracks batch sending metrics and adjusts batch size dynamically."""
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

batch_metrics = AdaptiveBatchMetrics()
metrics_lock = Lock()

# --- Dynamic Timeout Calculator ---
def calculate_timeout(batch_size: int, base_timeout: int = 10) -> int:
    """Calculate adaptive timeout based on batch size."""
    per_vector_time = 0.01
    timeout = base_timeout + int(batch_size * per_vector_time)
    return max(10, min(timeout, 120))

# --- Retry with Tier-based Splitting Logic ---
def send_batch_with_retry(
    vectors_batch: list,
    node_url: str,
    sent_to_node_id: str,
    retry_count: int = 0
) -> tuple:
    """Send a batch with intelligent retry using predefined size tiers."""
    batch_size = len(vectors_batch)
    
    if batch_size == 0:
        return True, {"batches": {}}
    
    if retry_count >= len(BATCH_TIERS):
        print(f"❌ All retry tiers exhausted for batch of {batch_size} vectors")
        batch_metrics.on_failure()
        return False, {}
    
    current_tier_size = BATCH_TIERS[retry_count]
    
    if batch_size > current_tier_size:
        print(f"📦 Batch size {batch_size} exceeds tier {retry_count+1} ({current_tier_size}), splitting...")
        batch_metrics.on_split()
        
        chunks = []
        for i in range(0, batch_size, current_tier_size):
            chunks.append(vectors_batch[i:i+current_tier_size])
        
        print(f"🔀 Split into {len(chunks)} chunks of max {current_tier_size} vectors")
        
        all_results = []
        for chunk in chunks:
            success, data = send_batch_with_retry(chunk, node_url, sent_to_node_id, retry_count)
            if not success:
                return False, {}
            all_results.append(data)
        
        merged_batches = {}
        for data in all_results:
            for node_id, count in data.get('batches', {}).items():
                merged_batches[node_id] = merged_batches.get(node_id, 0) + count
        
        return True, {"batches": merged_batches}
    
    timeout = calculate_timeout(batch_size)
    
    try:
        response = requests.post(
            f"{node_url}/add_vectors_bulk",
            json=vectors_batch,
            timeout=timeout
        )
        response.raise_for_status()
        
        res_data = response.json()
        batch_metrics.on_success(batch_size)
        return True, res_data
        
    except requests.exceptions.Timeout:
        next_tier = retry_count + 1
        if next_tier < len(BATCH_TIERS):
            next_tier_size = BATCH_TIERS[next_tier]
            print(f"⏱️  Timeout with batch size {batch_size} (tier {retry_count+1}: {current_tier_size})")
            print(f"   Falling back to tier {next_tier+1} (max size: {next_tier_size})...")
            batch_metrics.on_retry()
            return send_batch_with_retry(vectors_batch, node_url, sent_to_node_id, next_tier)
        else:
            print(f"❌ Timeout even with smallest tier ({current_tier_size})")
            batch_metrics.on_failure()
            return False, {}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error with batch size {batch_size} (tier {retry_count+1}): {e}")
        batch_metrics.on_retry()
        
        if retry_count == 0:
            wait_time = 2
            print(f"⏳ Waiting {wait_time}s before retry with next tier...")
            time.sleep(wait_time)
        
        next_tier = retry_count + 1
        if next_tier < len(BATCH_TIERS):
            return send_batch_with_retry(vectors_batch, node_url, sent_to_node_id, next_tier)
        else:
            batch_metrics.on_failure()
            return False, {}

def send_batch_with_retry_threadsafe(
    vectors_batch: list,
    node_url: str,
    sent_to_node_id: str,
    batch_num: int
) -> tuple:
    """Thread-safe wrapper for send_batch_with_retry."""
    success, res_data = send_batch_with_retry(vectors_batch, node_url, sent_to_node_id)
    return (batch_num, success, res_data)

def register_peers():
    """
    Register nodes with their assigned cluster centroids and peer information.
    Each node now receives MULTIPLE vectors (one per assigned cluster).
    """
    print("\n" + "="*60)
    print("1. SETTING NODE VECTORS & REGISTERING PEERS")
    print("="*60)
    
    try:
        # 1. Set the representative vectors for every node
        print("Setting node vectors (multiple per node)...")
        for node_idx in range(NUM_NODES):
            node_id = f"node{node_idx+1}"
            node_url = NODE_URLS[node_idx]
            
            # Get cluster indices assigned to this node
            cluster_indices = NODE_ASSIGNMENTS.get(str(node_idx), NODE_ASSIGNMENTS.get(node_idx, []))
            
            # Get the actual centroid vectors for those clusters
            node_vectors = [CENTROIDS[cluster_idx] for cluster_idx in cluster_indices]
            
            if not node_vectors:
                print(f"  ⚠️  {node_id}: No clusters assigned! Using random vector.")
                node_vectors = [np.random.rand(VECTOR_SIZE).tolist()]
            
            # Send all vectors for this node
            r_set = requests.post(f"{node_url}/set_node_vectors", json=node_vectors, timeout=5)
            r_set.raise_for_status()
            print(f"  {node_id}: Set {len(node_vectors)} representative vector(s) for clusters {cluster_indices}")

        # 2. Register all peers with all other peers
        print("\nRegistering peers...")
        for node_idx in range(NUM_NODES):
            host_id = f"node{node_idx+1}"
            host_url = NODE_URLS[node_idx]
            
            peers_registered = 0
            for peer_idx in range(NUM_NODES):
                if node_idx == peer_idx:
                    continue
                
                peer_id = f"node{peer_idx+1}"
                peer_url = NODE_URLS[peer_idx]
                
                # Get peer's cluster indices and vectors
                peer_cluster_indices = NODE_ASSIGNMENTS.get(str(peer_idx), NODE_ASSIGNMENTS.get(peer_idx, []))
                peer_vectors = [CENTROIDS[cluster_idx] for cluster_idx in peer_cluster_indices]
                
                if not peer_vectors:
                    peer_vectors = [np.random.rand(VECTOR_SIZE).tolist()]
                
                payload = {
                    "peer_id": peer_id,
                    "peer_url": peer_url,
                    "node_vectors": peer_vectors
                }
                
                r_reg = requests.post(f"{host_url}/register_peer", json=payload, timeout=5)
                r_reg.raise_for_status()
                peers_registered += 1
            
            print(f"  Host {host_id}: Registered {peers_registered} peers.")

        print("✅ Peers registered and vectors exchanged successfully.\n")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error setting/registering peers: {e}")
        print("!!! Please ensure all servers are running.")
        sys.exit(1)


def insert_vectors_bulk():
    """Insert vectors with PARALLEL batch sending using ThreadPoolExecutor."""
    print("\n" + "="*60)
    print(f"2. INSERTING {NUM_VECTORS} VECTORS (Size {VECTOR_SIZE})")
    print("="*60)
    print(f"   Starting with batch size: {batch_metrics.current_batch_size}")
    print(f"   Minimum batch size: {BATCH_SIZE_MIN}")
    print(f"   Maximum retries per batch: {MAX_RETRIES}")
    print(f"   🚀 Parallel workers: {NUM_NODES} (one per node)\n")
    
    insertions = []
    vector_index = 0
    batch_num = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = {}
        
        while vector_index < NUM_VECTORS or futures:
            # Submit new batches
            while len(futures) < NUM_NODES and vector_index < NUM_VECTORS:
                batch_num += 1
                
                current_batch_size = batch_metrics.current_batch_size
                end_index = min(vector_index + current_batch_size, NUM_VECTORS)
                
                # Round-robin entry node selection
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
            
            # Process completed batches
            if futures:
                done, pending = as_completed(futures.keys()), set(futures.keys())
                
                for future in done:
                    batch_info = futures[future]
                    batch_num_completed, success, res_data = future.result()
                    
                    if success:
                        batches = res_data.get('batches', {})
                        with metrics_lock:
                            for node_id, count in batches.items():
                                insertions.extend([node_id] * count)
                        
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
                    
                    del futures[future]
                    break
    
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
    print("\n" + "="*60)
    print("3. CHECKING VECTOR COUNTS")
    print("="*60)
    
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
            # Show expected count based on cluster assignment
            cluster_indices = NODE_ASSIGNMENTS.get(str(int(node_id[4:])-1), [])
            expected_note = f"({len(cluster_indices)} clusters assigned)" if cluster_indices else ""
            print(f"  - {node_id} Count: {count} {expected_note}")
        
        print(f"\nTotal Vectors Stored: {total_count}")
        print(f"Expected (with replication): {NUM_VECTORS * REPLICATION_FACTOR}")
        print(f"Expected (unique): {NUM_VECTORS}")
        
        # Check from test script perspective
        print("\n(Client-side insertion log check):")
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            script_count = insertions.count(node_id)
            print(f"  - {node_id} received: {script_count}")
        
        # Validate total count accounting for replication
        expected_total = NUM_VECTORS * REPLICATION_FACTOR
        if abs(total_count - expected_total) < NUM_VECTORS * 0.05:  # 5% tolerance
            print(f"✅  Total counts match expected (within 5% tolerance).")
        else:
            print(f"⚠️  Counts deviate from expected!")

        # Check load balancing quality
        expected_avg = expected_total / NUM_NODES
        max_count = max(c for _, c in all_counts)
        min_count = min(c for _, c in all_counts)
        imbalance = (max_count - min_count) / expected_avg * 100 if expected_avg > 0 else 0
        
        print(f"\nLoad Balance Statistics:")
        print(f"  - Expected avg per node: {expected_avg:,.1f}")
        print(f"  - Actual range: {min_count:,} to {max_count:,}")
        print(f"  - Imbalance: {imbalance:.1f}%")
        
        if imbalance < 20:
            print(f"✅  Distribution is well balanced (<20% imbalance).")
        else:
            print(f"⚠️  Distribution could be more balanced!")
        print("")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error checking counts: {e}")


def run_queries():
    """Run a single federated search query."""
    print("\n" + "="*60)
    print("4. RUNNING FEDERATED SEARCH QUERY")
    print("="*60)
    
    query_vector = data[NUM_VECTORS]['embedding']

    try:
        r_fed = requests.post(
            f"{NODE_URLS[0]}/search/federated",
            json=query_vector,
            params={"top_k": 5}
        )
        r_fed.raise_for_status()
        fed_results = r_fed.json()
        
        print(f"Federated search complete:")
        results_data = fed_results.get('results', {})
        
        all_results = []
        best_overall_node = "N/A"
        best_overall_score = -2.0
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            cluster_indices = NODE_ASSIGNMENTS.get(str(i), NODE_ASSIGNMENTS.get(i, []))
            node_vectors = [CENTROIDS[idx] for idx in cluster_indices] if cluster_indices else []
            
            # Calculate best similarity to any of this node's centroids
            node_similarity = -1.0
            if node_vectors:
                similarities = [utils.cosine_similarity(query_vector, nv) for nv in node_vectors]
                node_similarity = max(similarities)
            
            node_results_list = results_data.get(node_id, [])
            node_results_count = len(node_results_list)
            
            max_score = node_results_list[0].get('score', -1) if node_results_count > 0 else -1
            min_score = node_results_list[-1].get('score', -1) if node_results_count > 0 else -1
            
            print(f"  - Results from {node_id}: {node_results_count} "
                  f"(Best: {max_score:.4f}, Worst: {min_score:.4f}, "
                  f"Node Best: {node_similarity:.4f}, Clusters: {len(cluster_indices)})")
            
            if max_score > best_overall_score:
                best_overall_score = max_score
                best_overall_node = node_id
        
        print(f"\n  - Total Results Found: {fed_results.get('total_results')}")
        print(f"ℹ️  {best_overall_node} had the best matching vector (Score: {best_overall_score:.4f}).")

    except requests.exceptions.RequestException as e:
        print(f"!!! Error running federated query: {e}")


def main_app():
    print("\n" + "="*60)
    print(f"QDRANT SMART SHARDING TEST ({NUM_NODES} NODES)")
    print("="*60)
    print(f"Test will insert {NUM_VECTORS} vectors with {REPLICATION_FACTOR}x replication.")
    print(f"Using {len(CENTROIDS)} clusters balanced across {NUM_NODES} nodes.\n")
    
    time.sleep(2)
    
    start_time = time.time()
    
    register_peers()
    insertions = insert_vectors_bulk()
    
    print("--- Waiting 5s for background insertions to settle... ---")
    time.sleep(15)
    
    check_counts(insertions)
    run_queries()
    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"TEST COMPLETE IN {end_time - start_time:.2f} SECONDS")
    print("="*60)

if __name__ == "__main__":
    main_app()