import requests
import numpy as np
import uuid
import time
import json
import sys
import os
import utils
import msgpack
import ijson  # NEW: Import for streaming JSON parsing
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from node_assignment import compute_node_assignments # FIX: Import the missing function

# --- Configuration ---
try:
    NUM_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid argument. Using default of 3 nodes.")
    NUM_NODES = 3

print(f"--- Running Qdrant App for {NUM_NODES} nodes ---")

NUM_VECTORS = 20000
VECTOR_SIZE = 384

# --- Adaptive Batch Configuration ---
BATCH_TIERS = [800, 256, 64]
BATCH_SIZE_OPTIMAL = BATCH_TIERS[0]
BATCH_SIZE_MIN = BATCH_TIERS[-1]
MAX_RETRIES = len(BATCH_TIERS)

# --- Training Configuration ---
TRAINING_VECTORS = 10000
ASSIGNMENTS_FILE = 'node_assignments.json'
CENTROIDS_FILE = 'centroids.json' # FIX: Define the missing constant
FORCE_RETRAIN = False  # Set to True to force recomputation
# -----------------------------------

if VECTOR_SIZE < NUM_NODES:
    print(f"Error: VECTOR_SIZE ({VECTOR_SIZE}) must be >= NUM_NODES ({NUM_NODES})")
    print("Please increase VECTOR_SIZE in qdrant_app.py and server.py")
    sys.exit(1)

# --- Helper Functions ---
data = []
try:
    print("Loading embeddings from 'embeddings.json' using a streaming parser...")
    with open('embeddings.json', 'r') as f:
        # Use ijson to stream-load the large JSON file, preventing MemoryError.
        # We only load the number of vectors we actually need.
        # FIX: Use use_float=True for the C backend (yajl2_c) instead of parse_float.
        parser = ijson.items(f, 'item', use_float=True)
        for i, item in enumerate(parser):
            if i >= NUM_VECTORS + 1:
                break
            data.append(item)
    print(f"✅ Loaded {len(data)} embeddings from file.")

    # If the file has fewer vectors than needed, generate the rest.
    if len(data) < NUM_VECTORS + 1:
        print(f"Warning: embeddings.json has only {len(data)} items, but {NUM_VECTORS + 1} are needed.")
        print("Generating additional random data...")
        for i in range(len(data), NUM_VECTORS + 1):
            data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})

except FileNotFoundError:
    print("embeddings.json not found. Generating random data...")
    data = []  # Ensure data is a list
    for i in range(NUM_VECTORS + 1):
        data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})
except Exception as e:
    print(f"❌ Error loading embeddings.json: {e}")
    print("   Generating random data as a fallback...")
    data = []
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
        # NEW: Serialize with MessagePack instead of JSON
        binary_data = msgpack.packb(vectors_batch, use_bin_type=True)
        
        response = requests.post(
            f"{node_url}/add_vectors_bulk",
            data=binary_data,  # Send raw bytes
            headers={"Content-Type": "application/msgpack"},  # Set proper content type
            timeout=timeout
        )
        response.raise_for_status()
        
        # Response is still JSON (small payload, no need to change)
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
    """
    P2P Federated Search: Query via random entry node con routing server-side.
    Il nodo entry usa il SUO Meta-HNSW locale per trovare i migliori peer.
    """
    print("\n" + "="*60)
    print("4. Running P2P Federated Search (Server-Side Meta-HNSW Routing)")
    print("="*60)

    query_vector = np.array(data[NUM_VECTORS]['embedding'])
    k_nodes = min(3, NUM_NODES)  # Top-K nodi da interrogare
    k_results = 5  # Risultati per nodo

    # STEP 1: Scelta entry node CASUALE
    import random
    entry_node_idx = random.randint(0, NUM_NODES - 1)
    entry_node_url = NODE_URLS[entry_node_idx]
    entry_node_id = f"node{entry_node_idx + 1}"
    
    print(f"🎯 Using random entry node: {entry_node_id} ({entry_node_url})")
    print(f"📊 Query parameters: top_k_nodes={k_nodes}, top_k_results={k_results}")

    # STEP 2: Query l'entry node con endpoint P2P
    try:
        payload = {
            "query_vector": query_vector.tolist(),
            "top_k_nodes": k_nodes,
            "top_k_results": k_results
        }

        print(f"\n📡 Sending P2P query to {entry_node_id}...")
        response = requests.post(
            f"{entry_node_url}/search/p2p",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        results = response.json()
        
        # STEP 3: Mostra risultati aggregati
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

        print(f"\n💡 Efficiency: Entry node handled routing using local Meta-HNSW")
        print(f"   (No client-side Meta-HNSW needed - pure P2P architecture)")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during P2P query: {e}")
        import traceback
        print("\nDebug traceback:")
        traceback.print_exc()

def run_queries_debug():
    """
    Debug version: Query TUTTI i nodi per confrontare risultati.
    Usa ancora P2P ma con k_nodes = NUM_NODES.
    """
    print("\n" + "="*60)
    print("4 (DEBUG). Running Full P2P Search (All Nodes)")
    print("="*60)

    query_vector = np.array(data[NUM_VECTORS]['embedding'])
    k_nodes = NUM_NODES  # Query TUTTI i nodi
    k_results = 5

    import random
    entry_node_idx = random.randint(0, NUM_NODES - 1)
    entry_node_url = NODE_URLS[entry_node_idx]
    entry_node_id = f"node{entry_node_idx + 1}"
    
    print(f"🎯 Using entry node: {entry_node_id} (querying ALL {k_nodes} nodes for comparison)")

    try:
        payload = {
            "query_vector": query_vector.tolist(),
            "top_k_nodes": k_nodes,
            "top_k_results": k_results
        }

        response = requests.post(
            f"{entry_node_url}/search/p2p",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        results = response.json()
        
        print(f"\n✅ Full P2P query complete (ALL nodes):")
        print(f"  - Entry node: {results['entry_node']}")
        print(f"  - Nodes queried: {results['nodes_queried']}/{NUM_NODES}")
        print(f"  - Total results: {results['total_results']}")
        print(f"  - Best match: {results['best_match']['node']} (Score: {results['best_match']['score']:.4f})")

        print(f"\n📊 Per-node breakdown:")
        for node_name, node_results in results['results_per_node'].items():
            count = len(node_results)
            max_score = node_results[0]['score'] if count > 0 else -1
            min_score = node_results[-1]['score'] if count > 0 else -1
            print(f"  - {node_name}: {count} results (Best: {max_score:.4f}, Worst: {min_score:.4f})")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during debug query: {e}")
        import traceback
        print("\nDebug traceback:")
        traceback.print_exc()

def initialize_distributed_meta_hnsw():
    """
    Send Meta-HNSW initialization data to all nodes.
    Each node will build its own local Meta-HNSW instance.
    """
    print("\n" + "="*60)
    print("1.5. Initializing Distributed Meta-HNSW on Nodes")
    print("="*60)
    
    payload = {
        "dimension": VECTOR_SIZE,
        "max_clusters": max(len(CENTROIDS), NUM_NODES * 10),
        "centroids": CENTROIDS,
        "node_assignments": NODE_ASSIGNMENTS
    }
    
    success_count = 0
    for i, node_url in enumerate(NODE_URLS):
        node_id = f"node{i+1}"
        try:
            response = requests.post(
                f"{node_url}/init-meta-hnsw",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  ✓ {node_id}: Meta-HNSW initialized")
                success_count += 1
            else:
                print(f"  ✗ {node_id}: Failed (HTTP {response.status_code}) - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ✗ {node_id}: Error - {e}")
    
    print(f"\n✅ Distributed Meta-HNSW initialized on {success_count}/{NUM_NODES} nodes\n")
    
    if success_count < NUM_NODES:
        print("⚠️  Warning: Not all nodes initialized Meta-HNSW successfully!")
        print("   The system might not route queries correctly. Check server logs.")
        time.sleep(3)


def main_app():
    print("\n" + "="*60)
    print(f"QDRANT SMART SHARDING TEST ({NUM_NODES} NODES)")
    print("="*60)
    print(f"Test will insert {NUM_VECTORS} vectors with {REPLICATION_FACTOR}x replication.")
    print(f"Using {len(CENTROIDS)} clusters balanced across {NUM_NODES} nodes.\n")
    
    time.sleep(2)
    
    start_time = time.time()
    
    register_peers()
    
    # Inizializza Meta-HNSW sui SERVER (mantieni questa chiamata!)
    initialize_distributed_meta_hnsw()
    
    insertions = insert_vectors_bulk()

    #wait_for_qdrant_indexing()
    time.sleep(15)

    # REMOVED: initialize_meta_hnsw() ← DELETE questa chiamata (era per client Meta-HNSW)
    
    print("--- Waiting 15s for background insertions to settle... ---")
    time.sleep(15)
    
    check_counts(insertions)
    
    # Nuove query P2P
    run_queries()
    run_queries_debug()
    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"TEST COMPLETE IN {end_time - start_time:.2f} SECONDS")
    print("="*60)

if __name__ == "__main__":
    main_app()