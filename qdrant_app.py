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
from meta_hnsw import MetaHNSW

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

NUM_VECTORS = 15000
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

try:
    node_assignment_data = None

    # Try load existing assignments file
    if os.path.exists(ASSIGNMENTS_FILE):
        print(f"📂 Found existing assignments file '{ASSIGNMENTS_FILE}'...")
        with open(ASSIGNMENTS_FILE, 'r') as f:
            node_assignment_data = json.load(f)
        stats = node_assignment_data.get('stats', {})
        if stats.get('num_nodes') == NUM_NODES:
            print(f"✅ Assignments file matches NUM_NODES={NUM_NODES}, using cached assignments.")
        else:
            print(f"⚠️  Assignments file for {stats.get('num_nodes')} nodes (need {NUM_NODES}) — will recompute.")
            node_assignment_data = None

    # Compute assignments if needed
    if node_assignment_data is None:
        if len(data) < TRAINING_VECTORS:
            print(f"❌ ERROR: Insufficient data for training. Need {TRAINING_VECTORS}, have {len(data)}")
            print("   Please generate more embeddings.")
            sys.exit(1)

        print(f"🎯 Computing node assignments (training on {TRAINING_VECTORS} vectors)...")
        start_compute = time.time()

        node_assignment_data = compute_node_assignments(
            embeddings_file='embeddings.json',
            num_nodes=NUM_NODES,
            max_k_to_test=min(30, TRAINING_VECTORS // 100),
            random_state=42,
            max_vectors=TRAINING_VECTORS,
            rep_factor=None,
            beam_width=5,
            use_simulated_annealing=False
        )

        compute_time = time.time() - start_compute
        print(f"✅ Assignment computation completed in {compute_time:.2f}s.")

        # persist full assignment data for future runs
        with open(ASSIGNMENTS_FILE, 'w') as f:
            json.dump(node_assignment_data, f, indent=2)
        print(f"💾 Saved assignment data to {ASSIGNMENTS_FILE}")

    # Extract CENTROIDS (tutti i centroidi) e la mappa NODE_ASSIGNMENTS (lista indici per nodo)
    CENTROIDS = node_assignment_data.get('centroids', [])
    NODE_ASSIGNMENTS = node_assignment_data.get('node_assignments', {})
    ASSIGNMENT_DETAILS = node_assignment_data.get('assignment_details', {})
    REPLICATION_FACTOR = node_assignment_data.get('stats', {}).get('replication_factor', 1)

    # Optional: save the full centroid list for compatibility with other tools
    try:
        with open(CENTROIDS_FILE, 'w') as f:
            json.dump(CENTROIDS, f, indent=2)
        print(f"💾 Full centroid list saved to {CENTROIDS_FILE} ({os.path.getsize(CENTROIDS_FILE)/1024:.2f} KB)")
    except Exception:
        pass

    # Print brief summary
    print(f"Loaded {len(CENTROIDS)} centroids; node assignment map contains {len(NODE_ASSIGNMENTS)} nodes' assignments.")

except Exception as e:
    print(f"❌ FATAL ERROR during assignment/centroid preparation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Dynamic Node Configuration ---
NODE_URLS = []
for i in range(1, NUM_NODES + 1):
    NODE_URLS.append(f"http://localhost:{8000 + i}")

# --- NEW: Availability / peer-aggregated health logic ---
MAJORITY = (NUM_NODES // 2) + 1  # need >N/2 reporters to mark a node DOWN to exclude it
PEER_REPORT_REFRESH_INTERVAL = 30  # seconds between aggregate checks
_last_peer_report_time = 0.0
UNAVAILABLE_NODES = set()  # node ids like "node1", "node2"
meta_hnsw: MetaHNSW = None
META_HNSW_PATH = 'meta_hnsw_index.pkl'

def initialize_meta_hnsw():
    global meta_hnsw

    print("\n--- 3. Initializing Meta-HNSW with K-means Centroids ---")

    if os.path.exists(META_HNSW_PATH):
        print(f"📂 Found existing Meta-HNSW index: {META_HNSW_PATH}")
        try:
            meta_hnsw = MetaHNSW.load(META_HNSW_PATH)
            print(f"✅ Loaded MetaHNSW with {len(meta_hnsw.node_to_clusters)} nodes")
            if len(meta_hnsw.node_to_clusters) == NUM_NODES:
                print("   Using cached index (matching node count)")
                stats = meta_hnsw.get_statistics()
                print(f"   Cached stats:")
                print(f"     - Total clusters: {stats['num_clusters_total']}")
                print(f"     - Avg clusters/node: {stats['avg_clusters_per_node']:.1f}")
                print(f"     - Mean node distance: {stats['mean_node_distance']:.4f}")
                return
            else:
                print(f"⚠️  Cached index has {len(meta_hnsw.node_to_clusters)} nodes, but {NUM_NODES} are needed")
                print("   Rebuilding index...")
        except Exception as e:
            print(f"⚠️  Failed to load cached index: {e}")
            print("   Building new index...")

    print(f"🏗️  Building new Meta-HNSW index for {NUM_NODES} nodes...")
    meta_hnsw = MetaHNSW(
        dimension=VECTOR_SIZE,
        max_clusters=max(len(CENTROIDS), NUM_NODES * 10),
        ef_construction=200,
        M=16
    )

    start_time = time.time()

    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        cluster_indices = NODE_ASSIGNMENTS.get(str(i), NODE_ASSIGNMENTS.get(i, []))
        cluster_list = []
        for idx in cluster_indices:
            cluster_list.append(np.array(CENTROIDS[idx]))
        if not cluster_list:
            # fallback: use a random unit vector if node has no clusters
            v = np.random.rand(VECTOR_SIZE)
            v = v / np.linalg.norm(v)
            cluster_list = [v]
            print(f"  {node_id}: WARNING - no clusters assigned, adding random fallback cluster")

        print(f"  Adding {node_id} with {len(cluster_list)} cluster(s)...", end=" ")
        meta_hnsw.add_node_clusters(node_id, cluster_list)
        print("✓")

    meta_hnsw.force_rebuild()
    elapsed = time.time() - start_time

    stats = meta_hnsw.get_statistics()
    print(f"\n✅ Meta-HNSW initialized in {elapsed:.2f}s")
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Total clusters: {stats['num_clusters_total']}")
    print(f"   Avg clusters/node: {stats['avg_clusters_per_node']:.1f}")
    print(f"   Mean node distance: {stats['mean_node_distance']:.4f}")

    print(f"\n🔍 Verifying cluster consistency (per-node first cluster sample):")
    for i in range(NUM_NODES):
        node_id = f"node{i+1}"
        cluster_indices = NODE_ASSIGNMENTS.get(str(i), NODE_ASSIGNMENTS.get(i, []))
        if cluster_indices:
            sample_idx = cluster_indices[0]
            kmeans_centroid = np.array(CENTROIDS[sample_idx])
            cluster_ids = meta_hnsw.node_to_clusters[node_id]
            hnsw_cluster = meta_hnsw.cluster_centroids[cluster_ids[0]]
            diff = np.linalg.norm(kmeans_centroid - hnsw_cluster)
            print(f"   {node_id}: sample_cluster_idx={sample_idx}, hnsw_cluster_id={cluster_ids[0]}, diff={diff:.6f}")
        else:
            print(f"   {node_id}: no assigned clusters to verify")

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

        
def update_unavailable_nodes():
    """
    Query /peers on all nodes and aggregate how many reporters mark each node DOWN.
    Mark nodes unavailable if they are considered DOWN by >= MAJORITY reporters.
    """
    global _last_peer_report_time, UNAVAILABLE_NODES
    counts_down = {f"node{i+1}": 0 for i in range(NUM_NODES)}
    reporters = 0

    for i, reporter_url in enumerate(NODE_URLS):
        try:
            resp = requests.get(f"{reporter_url}/peers", timeout=2)
            if resp.status_code != 200:
                continue
            reporters += 1
            info = resp.json()
            peers = info.get("peers", {})
            # peers keys are like "node1", ...
            for peer_id, pdata in peers.items():
                status = pdata.get("status", "UNKNOWN")
                if status == "DOWN":
                    counts_down[peer_id] = counts_down.get(peer_id, 0) + 1
        except requests.exceptions.RequestException:
            # reporter unreachable -> skip (do not count as a DOWN report)
            continue

    new_unavailable = set()
    for node_id, down_count in counts_down.items():
        if down_count >= MAJORITY:
            new_unavailable.add(node_id)

    UNAVAILABLE_NODES = new_unavailable
    _last_peer_report_time = time.time()
    if UNAVAILABLE_NODES:
        print(f"Client: Nodes excluded as entry by MAJORITY: {sorted(list(UNAVAILABLE_NODES))}")
    else:
        print("Client: No entry nodes excluded by MAJORITY (all eligible)")

def choose_entry_node(batch_num: int) -> int:
    """
    Choose an entry node index (0-based) using round-robin but skipping UNAVAILABLE_NODES.
    If all nodes are excluded, fall back to plain round-robin.
    """
    start = (batch_num - 1) % NUM_NODES
    for offset in range(NUM_NODES):
        idx = (start + offset) % NUM_NODES
        node_id = f"node{idx + 1}"
        if node_id not in UNAVAILABLE_NODES:
            return idx
    # fallback: all excluded -> return round-robin index and log warning
    print("Client WARNING: all nodes reported excluded by peers; falling back to round-robin entry selection.")
    return start

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
    
    # initial update of peer reports
    try:
        update_unavailable_nodes()
    except Exception as e:
        print(f"Client: initial peer report failed: {e}")
    
    with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = {}
        
        while vector_index < NUM_VECTORS or futures:
            # refresh peer reports periodically
            if time.time() - _last_peer_report_time > PEER_REPORT_REFRESH_INTERVAL:
                try:
                    update_unavailable_nodes()
                except Exception as e:
                    print(f"Client: peer report update failed: {e}")

            # Submit new batches
            while len(futures) < NUM_NODES and vector_index < NUM_VECTORS:
                batch_num += 1
                
                current_batch_size = batch_metrics.current_batch_size
                end_index = min(vector_index + current_batch_size, NUM_VECTORS)
                
                # Determine entry node (round-robin but skip unavailable)
                node_url_index = choose_entry_node(batch_num)
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

def run_queries_debug():
    """Run federated search query with Meta-HNSW intelligent routing."""
    print("--- 4. Running Smart Federated Search (Meta-HNSW Routing) ---")

    if meta_hnsw is None:
        print("⚠️  Meta-HNSW not initialized, falling back to broadcast query")
        run_queries_fallback()
        return

    query_vector = np.array(data[NUM_VECTORS]['embedding'])

    k_nodes = NUM_NODES

    print(k_nodes)
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
            cluster_indices = NODE_ASSIGNMENTS.get(str(i), NODE_ASSIGNMENTS.get(i, []))
            node_vectors = [CENTROIDS[idx] for idx in cluster_indices] if cluster_indices else []

            # Compute best similarity between query and any centroid assigned to the node
            node_similarity = -1.0
            if node_vectors:
                sims = [utils.cosine_similarity(query_vector, nv) for nv in node_vectors]
                node_similarity = max(sims)

            node_results_list = results_data.get(node_id, [])
            node_results_count = len(node_results_list)

            max_score = node_results_list[0].get('score', -1) if node_results_count > 0 else -1
            min_score = node_results_list[-1].get('score', -1) if node_results_count > 0 else -1

            print(f"  - {node_id}: {node_results_count} results (Best: {max_score:.4f}, Worst: {min_score:.4f}, Centroid Best: {node_similarity:.4f}, Clusters: {len(cluster_indices)})")

            if max_score > best_overall_score:
                best_overall_score = max_score
                best_overall_node = node_id

        print(f"\n  - Total Results: {fed_results.get('total_results')}")
        print(f"  - Best Match: {best_overall_node} (Score: {best_overall_score:.4f})")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error running broadcast query: {e}")


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

    wait_for_qdrant_indexing()

    initialize_meta_hnsw()
    
    print("--- Waiting 5s for background insertions to settle... ---")
    time.sleep(15)
    
    check_counts(insertions)
    run_queries()
    run_queries_debug()
    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"TEST COMPLETE IN {end_time - start_time:.2f} SECONDS")
    print("="*60)

if __name__ == "__main__":
    main_app()