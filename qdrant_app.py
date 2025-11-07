import requests
import numpy as np
import uuid
import time
import json
import sys
import os
import msgpack
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Configuration ---
try:
    NUM_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid argument. Using default of 3 nodes.")
    NUM_NODES = 3

print(f"--- Running Qdrant App for {NUM_NODES} nodes ---")

NUM_VECTORS = 20000
VECTOR_SIZE = 384

# --- NEW: Bootstrap Configuration ---
BOOTSTRAP_VECTORS = 10000  # First N vectors sent to ALL nodes for clustering
COORDINATOR_NODE_ID = "node1"  # Fixed coordinator (always first node)

# --- Adaptive Batch Configuration ---
BATCH_TIERS = [800, 256, 64]
BATCH_SIZE_OPTIMAL = BATCH_TIERS[0]
BATCH_SIZE_MIN = BATCH_TIERS[-1]
MAX_RETRIES = len(BATCH_TIERS)

if VECTOR_SIZE < NUM_NODES:
    print(f"Error: VECTOR_SIZE ({VECTOR_SIZE}) must be >= NUM_NODES ({NUM_NODES})")
    sys.exit(1)

# --- Load Data ---
data = []
try:
    with open('embeddings.json', 'r') as f:
        data = json.load(f)
    if len(data) < NUM_VECTORS + 1:
        print(f"Warning: embeddings.json has only {len(data)} items, need {NUM_VECTORS}+1")
        for i in range(len(data), NUM_VECTORS + 1):
            data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})
except FileNotFoundError:
    print("embeddings.json not found. Generating random data...")
    for i in range(NUM_VECTORS + 1):
        data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})

# --- Dynamic Node Configuration ---
NODE_URLS = [f"http://localhost:{8000 + i}" for i in range(1, NUM_NODES + 1)]

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
        
        # NEW: Handle 503 (clustering not ready) - wait and retry
        if response.status_code == 503:
            print(f"⏳ Node {sent_to_node_id} not ready (clustering in progress)")
            if retry_count == 0:
                print(f"   Waiting 5s for clustering to complete...")
                time.sleep(5)
                return send_batch_with_retry(vectors_batch, node_url, sent_to_node_id, 0)  # Retry same tier
            else:
                print(f"   Clustering still not ready after retry")
                return False, {}
        
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
    """Register nodes with peer information only (no clustering yet)."""
    print("\n" + "="*60)
    print("1. REGISTERING PEERS (NO CLUSTERING YET)")
    print("="*60)
    
    try:
        for node_idx in range(NUM_NODES):
            host_id = f"node{node_idx+1}"
            host_url = NODE_URLS[node_idx]
            
            peers_registered = 0
            for peer_idx in range(NUM_NODES):
                if node_idx == peer_idx:
                    continue
                
                peer_id = f"node{peer_idx+1}"
                peer_url = NODE_URLS[peer_idx]
                
                payload = {
                    "peer_id": peer_id,
                    "peer_url": peer_url,
                    "node_vectors": []  # Empty - clustering will set these
                }
                
                r_reg = requests.post(f"{host_url}/register_peer", json=payload, timeout=5)
                r_reg.raise_for_status()
                peers_registered += 1
            
            print(f"  {host_id}: Registered {peers_registered} peers")

        print("✅ Peer registration complete\n")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error registering peers: {e}")
        sys.exit(1)


def insert_vectors_bootstrap():
    """Broadcast bootstrap vectors to ALL nodes for clustering (PARALLEL)."""
    print("\n" + "="*60)
    print(f"2. BOOTSTRAP: Broadcasting {BOOTSTRAP_VECTORS} vectors to ALL nodes (PARALLEL)")
    print(f"   Fixed coordinator: {COORDINATOR_NODE_ID}")
    print("="*60)
    
    insertions = []
    batch_size = 256
    batch_num = 0
    
    start_time = time.time()
    
    # NEW: ThreadPoolExecutor per parallelizzare broadcast
    with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        for i in range(0, BOOTSTRAP_VECTORS, batch_size):
            batch_num += 1
            end_index = min(i + batch_size, BOOTSTRAP_VECTORS)
            
            # Prepara batch payload
            batch_payload = []
            for j in range(i, end_index):
                final_vec = data[j]['embedding']
                vector_data = {
                    "id": str(uuid.uuid4()),
                    "vector": final_vec,
                    "payload": {
                        "source_type": "bootstrap",
                        "index": j
                    }
                }
                batch_payload.append(vector_data)
            
            # SERIALIZZA UNA VOLTA (evita duplicazione memoria)
            binary_data = msgpack.packb(batch_payload, use_bin_type=True)
            
            # BROADCAST PARALLELO a TUTTI i nodi
            futures = []
            for node_idx in range(NUM_NODES):
                node_url = NODE_URLS[node_idx]
                node_id = f"node{node_idx + 1}"
                
                # Submit task al thread pool
                future = executor.submit(
                    _send_bootstrap_batch,
                    node_url,
                    node_id,
                    binary_data,
                    batch_num
                )
                futures.append((future, node_id, len(batch_payload)))
            
            # Aspetta che TUTTI i nodi completino questo batch
            # prima di passare al prossimo
            for future, node_id, vector_count in futures:
                try:
                    success = future.result(timeout=30)
                    if success:
                        insertions.extend([node_id] * vector_count)
                except Exception as e:
                    print(f"⚠️  Batch {batch_num} failed on {node_id}: {e}")
            
            if batch_num % 10 == 0:
                print(f"Batch {batch_num}: {end_index}/{BOOTSTRAP_VECTORS} vectors → {NUM_NODES} nodes (parallel)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Bootstrap complete in {total_time:.2f}s (parallel)")
    print(f"   Total insertions: {len(insertions)} ({BOOTSTRAP_VECTORS} × {NUM_NODES})")
    print(f"   Coordinator {COORDINATOR_NODE_ID} will now perform clustering...\n")
    
    return insertions


def _send_bootstrap_batch(node_url: str, node_id: str, binary_data: bytes, batch_num: int) -> bool:
    """
    Helper function per inviare batch a UN nodo (eseguito in thread separato).
    
    Args:
        node_url: URL del nodo target
        node_id: ID del nodo
        binary_data: Batch serializzato (MessagePack)
        batch_num: Numero batch (per logging)
    
    Returns:
        True se successo, False altrimenti
    """
    try:
        response = requests.post(
            f"{node_url}/add_vectors_bulk",
            data=binary_data,
            headers={"Content-Type": "application/msgpack"},
            timeout=30
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"⚠️  Batch {batch_num} → {node_id}: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch {batch_num} → {node_id}: {e}")
        return False


def wait_for_clustering():
    """Poll coordinator until clustering is complete."""
    print("\n" + "="*60)
    print(f"3. WAITING FOR {COORDINATOR_NODE_ID} TO COMPLETE CLUSTERING")
    print("="*60)
    
    coordinator_url = NODE_URLS[0]
    max_wait = 1800
    start_time = time.time()
    poll_interval = 5
    last_bootstrap_count = 0
    
    print(f"   Max wait time: {max_wait//60} minutes")
    print(f"   Polling every {poll_interval}s...\n")
    
    while time.time() - start_time < max_wait:
        elapsed = int(time.time() - start_time)
        
        try:
            response = requests.get(f"{coordinator_url}/clustering-status", timeout=5)
            
            if response.status_code == 200:
                status = response.json()
                
                bootstrap_count = status.get("bootstrap_vectors_received", 0)
                if bootstrap_count != last_bootstrap_count:
                    print(f"  [{elapsed}s] {COORDINATOR_NODE_ID}: {bootstrap_count}/{BOOTSTRAP_VECTORS} vectors")
                    last_bootstrap_count = bootstrap_count
                
                if status.get("clustering_complete", False):
                    print(f"\n✅ {COORDINATOR_NODE_ID}: Clustering complete after {elapsed}s!")
                    
                    # NEW: Verify cleanup completion on all nodes
                    print(f"\n🔍 Verifying cleanup completion across all nodes...")
                    all_cleanup_complete = True
                    
                    for i in range(NUM_NODES):
                        node_url = NODE_URLS[i]
                        node_id = f"node{i+1}"
                        try:
                            cleanup_resp = requests.get(f"{node_url}/cleanup-status", timeout=3)
                            if cleanup_resp.status_code == 200:
                                cleanup_data = cleanup_resp.json()
                                cleanup_complete = cleanup_data.get('cleanup_complete', False)
                                config_version = cleanup_data.get('config_version')
                                
                                if cleanup_complete:
                                    print(f"  ✅ {node_id}: Cleanup complete (version: {config_version})")
                                else:
                                    print(f"  ⏳ {node_id}: Cleanup in progress...")
                                    all_cleanup_complete = False
                        except:
                            print(f"  ⚠️  {node_id}: Cannot verify cleanup status")
                            all_cleanup_complete = False
                    
                    if not all_cleanup_complete:
                        print(f"\n⏳ Waiting for all nodes to complete cleanup...")
                        time.sleep(poll_interval)
                        continue
                    
                    # Verify config sync
                    print(f"\n🔍 Verifying config sync...")
                    all_versions = {}
                    for i in range(NUM_NODES):
                        node_url = NODE_URLS[i]
                        node_id = f"node{i+1}"
                        try:
                            cfg_resp = requests.get(f"{node_url}/config/status", timeout=3)
                            if cfg_resp.status_code == 200:
                                cfg_data = cfg_resp.json()
                                version = cfg_data.get('config_version')
                                all_versions[node_id] = version
                        except:
                            all_versions[node_id] = "ERROR"
                    
                    unique_versions = set(all_versions.values())
                    if len(unique_versions) == 1:
                        print(f"  ✅ All nodes synchronized on version: {list(unique_versions)[0]}")
                    else:
                        print(f"  ⚠️  Version mismatch detected: {unique_versions}")
                    
                    # NEW: Final verification - ensure ALL nodes accept routing
                    print(f"\n🔍 Verifying ALL nodes are ready to accept routed vectors...")
                    all_nodes_ready = True
                    max_readiness_wait = 30  # 30 seconds max
                    readiness_start = time.time()
                    
                    while time.time() - readiness_start < max_readiness_wait:
                        ready_count = 0
                        
                        for i in range(NUM_NODES):
                            node_url = NODE_URLS[i]
                            node_id = f"node{i+1}"
                            
                            try:
                                # Try a test request to see if node is ready
                                test_resp = requests.get(f"{node_url}/clustering-status", timeout=2)
                                if test_resp.status_code == 200:
                                    test_data = test_resp.json()
                                    if test_data.get('clustering_complete'):
                                        ready_count += 1
                            except:
                                pass
                        
                        if ready_count == NUM_NODES:
                            print(f"  ✅ All {NUM_NODES} nodes are ready for routing")
                            all_nodes_ready = True
                            break
                        
                        elapsed_readiness = int(time.time() - readiness_start)
                        if elapsed_readiness % 5 == 0:
                            print(f"  [{elapsed_readiness}s] Ready: {ready_count}/{NUM_NODES} nodes")
                        
                        time.sleep(1)
                    
                    if not all_nodes_ready:
                        print(f"  ⚠️  Some nodes may not be ready yet, but proceeding...")
                    
                    return True
                
                elif status.get("clustering_in_progress", False):
                    print(f"  [{elapsed}s] {COORDINATOR_NODE_ID}: 🔄 Clustering in progress...")
                else:
                    if elapsed % 30 == 0:
                        print(f"  [{elapsed}s] {COORDINATOR_NODE_ID}: Collecting bootstrap ({bootstrap_count}/{BOOTSTRAP_VECTORS})...")
            else:
                print(f"  ⚠️  [{elapsed}s] Failed to query coordinator: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  [{elapsed}s] Error querying coordinator: {e}")
        
        time.sleep(poll_interval)
    
    print(f"\n❌ Timeout: {COORDINATOR_NODE_ID} did not complete clustering in {max_wait}s\n")
    return False


def insert_vectors_routed():
    """Insert remaining vectors using smart routing (post-clustering)."""
    print("\n" + "="*60)
    print(f"4. SMART ROUTING: Inserting {NUM_VECTORS - BOOTSTRAP_VECTORS} remaining vectors")
    print("="*60)
    
    insertions = []
    vector_index = BOOTSTRAP_VECTORS
    batch_num = 0
    
    # NEW: Track replication statistics
    replication_stats = {
        'vectors_processed': 0,
        'total_replica_sends': 0,
        'replicas_per_vector': []
    }
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
        futures = {}
        
        while vector_index < NUM_VECTORS or futures:
            # Submit new batches
            while len(futures) < NUM_NODES and vector_index < NUM_VECTORS:
                batch_num += 1
                
                current_batch_size = batch_metrics.current_batch_size
                end_index = min(vector_index + current_batch_size, NUM_VECTORS)
                
                # Round-robin entry node
                node_url_index = (batch_num - 1) % NUM_NODES
                node_url = NODE_URLS[node_url_index]
                sent_to_node_id = f"node{node_url_index + 1}"
                
                batch_payload = []
                for j in range(vector_index, end_index):
                    final_vec = data[j]['embedding']
                    vector_data = {
                        "id": str(uuid.uuid4()),
                        "vector": final_vec,
                        "payload": {
                            "source_type": "routed",
                            "sent_to_node": sent_to_node_id,
                            "index": j,
                            "batch_num": batch_num  # NEW: Track batch
                        }
                    }
                    batch_payload.append(vector_data)
                    
                    # NEW: Track expected replicas per vector
                    replication_stats['vectors_processed'] += 1
                
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
                    'node_id': sent_to_node_id,
                    'batch_size': len(batch_payload)  # NEW
                }
                
                vector_index = end_index
            
            # Process completed batches
            if futures:
                done_futures = list(as_completed(futures.keys(), timeout=None))
                
                for future in done_futures:
                    batch_info = futures[future]
                    batch_num_completed, success, res_data = future.result()
                    
                    if success:
                        batches = res_data.get('batches', {})
                        
                        # NEW: Verbose logging for first few batches
                        if batch_num_completed <= 3:
                            print(f"\n🔍 Batch {batch_num_completed} Routing Detail:")
                            print(f"   Entry node: {batch_info['node_id']}")
                            print(f"   Batch size: {batch_info['batch_size']} vectors")
                            print(f"   Response batches: {batches}")
                            total_in_batch = sum(batches.values())
                            print(f"   Total routed: {total_in_batch}")
                            expected = batch_info['batch_size'] * 3
                            print(f"   Expected (3x): {expected}")
                            if total_in_batch < expected:
                                print(f"   ⚠️  MISSING: {expected - total_in_batch} replicas!")
                        
                        # Track replication stats
                        total_replicas = sum(batches.values())
                        batch_size = batch_info['batch_size']
                        avg_replicas = total_replicas / batch_size if batch_size > 0 else 0
                        
                        replication_stats['total_replica_sends'] += total_replicas
                        replication_stats['replicas_per_vector'].append(avg_replicas)
                        
                        with metrics_lock:
                            for node_id, count in batches.items():
                                insertions.extend([node_id] * count)
                        
                        if batch_num_completed % 30 == 0:
                            elapsed = time.time() - start_time
                            total_inserted = batch_info['end_index'] - BOOTSTRAP_VECTORS
                            vectors_per_sec = total_inserted / elapsed if elapsed > 0 else 0
                            
                            # NEW: Show replication info
                            print(f"Batch {batch_num_completed}: {total_inserted}/{NUM_VECTORS - BOOTSTRAP_VECTORS} "
                                  f"({vectors_per_sec:.0f} vec/s) | Replicas: {avg_replicas:.1f}x per vector")
                    
                    del futures[future]
    
    total_time = time.time() - start_time
    
    # NEW: Print replication analysis
    print(f"\n✅ Routed insertion complete in {total_time:.2f}s")
    print(f"\n📊 Replication Statistics:")
    print(f"  Vectors processed:     {replication_stats['vectors_processed']:>10,}")
    print(f"  Total replica sends:   {replication_stats['total_replica_sends']:>10,}")
    
    if replication_stats['vectors_processed'] > 0:
        actual_rep = replication_stats['total_replica_sends'] / replication_stats['vectors_processed']
        print(f"  Average replication:   {actual_rep:>10.2f}x")
        print(f"  Expected replication:  {3.0:>10.1f}x")
        
        if abs(actual_rep - 3.0) > 0.5:
            print(f"\n⚠️  WARNING: Replication factor is {actual_rep:.2f}x instead of 3.0x")
            print(f"     This indicates vectors are NOT being properly replicated!")
            print(f"     Expected total: {replication_stats['vectors_processed'] * 3:,}")
            print(f"     Actual total:   {replication_stats['total_replica_sends']:,}")
            print(f"     Missing:        {replication_stats['vectors_processed'] * 3 - replication_stats['total_replica_sends']:,}")
    
    print()
    
    return insertions


def check_counts(insertions: list):
    """Check final vector counts on each node with detailed statistics."""
    print("\n" + "="*60)
    print("5. FINAL VECTOR DISTRIBUTION ANALYSIS")
    print("="*60)
    
    try:
        all_counts = []
        total_count = 0
        
        # Get counts from all nodes
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_url = NODE_URLS[i]
            
            r = requests.get(f"{node_url}/count", timeout=5)
            r.raise_for_status()
            count = r.json().get('count', 0)
            
            all_counts.append((node_id, count))
            total_count += count

        # Print individual node counts
        print("\n📊 Vector Counts per Node:")
        print("-" * 60)
        for node_id, count in all_counts:
            percentage = (count / total_count * 100) if total_count > 0 else 0
            bar_length = int(percentage / 2)  # Scale bar to 50 chars max
            bar = "█" * bar_length
            print(f"  {node_id}: {count:>6,} vectors ({percentage:5.2f}%) {bar}")
        
        print("-" * 60)
        
        # Calculate statistics
        if all_counts:
            counts_only = [c for _, c in all_counts]
            max_count = max(counts_only)
            min_count = min(counts_only)
            avg_count = total_count / NUM_NODES
            median_count = sorted(counts_only)[len(counts_only) // 2]
            
            # Calculate variance and standard deviation
            variance = sum((c - avg_count) ** 2 for c in counts_only) / NUM_NODES
            std_dev = variance ** 0.5
            
            imbalance = ((max_count - min_count) / avg_count * 100) if avg_count > 0 else 0
            
            print(f"\n📈 Distribution Statistics:")
            print(f"  Total vectors:        {total_count:>10,}")
            
            # FIXED: Expected calculation (REPLICA-AWARE CLEANUP)
            # Bootstrap: 10k vettori × 3 repliche = 30k totali
            # Routed:    10k vettori × 3 repliche = 30k totali
            # TOTALE:    60k vettori
            
            bootstrap_kept_with_replicas = BOOTSTRAP_VECTORS * 3  # 10k × 3 = 30k
            routed_with_replicas = (NUM_VECTORS - BOOTSTRAP_VECTORS) * 3  # 10k × 3 = 30k
            expected_total = bootstrap_kept_with_replicas + routed_with_replicas  # 60k
            
            print(f"  Expected (REPLICA-AWARE): {expected_total:>10,}")
            print(f"  Calculation: ({BOOTSTRAP_VECTORS:,} bootstrap + {NUM_VECTORS - BOOTSTRAP_VECTORS:,} routed) × 3 rep = {expected_total:,}")
            print(f"  Breakdown:")
            print(f"    - Bootstrap with replicas: {bootstrap_kept_with_replicas:>10,}")
            print(f"    - Routed with replicas:    {routed_with_replicas:>10,}")
            
            print(f"  Average per node:     {avg_count:>10,.1f}")
            print(f"  Median per node:      {median_count:>10,}")
            print(f"  Min count:            {min_count:>10,} ({all_counts[counts_only.index(min_count)][0]})")
            print(f"  Max count:            {max_count:>10,} ({all_counts[counts_only.index(max_count)][0]})")
            print(f"  Range (max-min):      {max_count - min_count:>10,}")
            print(f"  Standard deviation:   {std_dev:>10,.2f}")
            print(f"  Imbalance:            {imbalance:>10.2f}%")
            
            # Load Balance Assessment (NEW)
            print(f"\n⚖️ Load Balance Assessment:")
            for node_id, count in all_counts:
                expected_count = avg_count
                diff = count - expected_count
                diff_percent = (diff / expected_count * 100) if expected_count != 0 else 0
                
                status = "✅" if abs(diff_percent) <= 10 else "⚠️" if abs(diff_percent) <= 25 else "❌"
                print(f"  {node_id}: {count:>6,} vectors (diff: {diff:+.1f}, {diff_percent:+.1f}%) {status}")
            
            # Replication factor estimation
            actual_replication = total_count / NUM_VECTORS if NUM_VECTORS > 0 else 0
            
            # NEW: Calculate routed-only replication (excluding bootstrap)
            routed_only_count = total_count  # After cleanup, all vectors are routed
            routed_vectors = NUM_VECTORS - BOOTSTRAP_VECTORS
            routed_replication = routed_only_count / routed_vectors if routed_vectors > 0 else 0
            
            # FIXED: Calcola expected replication (da beam search)
            expected_replication = 3.0  # Default from beam search (rep_factor)
            
            print(f"\n🔄 Replication Analysis:")
            print(f"  Total unique vectors:    {NUM_VECTORS:>10,}")
            print(f"  Bootstrap (deleted):     {BOOTSTRAP_VECTORS:>10,}")
            print(f"  Routed (kept):           {routed_vectors:>10,}")
            print(f"  Total stored:            {total_count:>10,}")
            print(f"  Overall replication:     {actual_replication:>10.2f}x (vs {NUM_VECTORS:,})")
            print(f"  Routed replication:      {routed_replication:>10.2f}x (vs {routed_vectors:,})")
            print(f"  Expected replication:   ~{expected_replication:.1f}x")
            
            replication_diff = abs(routed_replication - expected_replication)
            if replication_diff < 0.3:
                print(f"  ✅ Replication matches expected")
            elif replication_diff < 0.7:
                print(f"  ⚠️  Replication slightly off (diff: {replication_diff:.2f}x)")
            else:
                print(f"  ❌ Replication significantly different (diff: {replication_diff:.2f}x)")
                print(f"     Possible causes:")
                print(f"     - Cleanup incomplete (bootstrap not fully deleted)")
                print(f"     - Double routing (vectors routed twice)")
                print(f"     - Replication factor mismatch in beam search")
        
        print()
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error checking counts: {e}")


def run_queries():
    """Run P2P federated search queries using HNSW routing."""
    print("\n" + "="*60)
    print("6. P2P QUERY WITH META-HNSW ROUTING")
    print("="*60)

    # Use last vector as query
    query_vector = np.array(data[NUM_VECTORS]['embedding'])
    k_nodes = min(3, NUM_NODES)
    k_results = 5

    # Random entry node
    import random
    entry_node_idx = random.randint(0, NUM_NODES - 1)
    entry_node_url = NODE_URLS[entry_node_idx]
    entry_node_id = f"node{entry_node_idx + 1}"
    
    print(f"\n🎯 Query Setup:")
    print(f"  Entry node:       {entry_node_id}")
    print(f"  Query vector:     embeddings.json[{NUM_VECTORS}] (384-dim)")
    print(f"  Target nodes:     {k_nodes} (top-K via Meta-HNSW)")
    print(f"  Results per node: {k_results}")
    
    try:
        payload = {
            "query_vector": query_vector.tolist(),
            "top_k_nodes": k_nodes,
            "top_k_results": k_results
        }

        print(f"\n📡 Sending P2P query to {entry_node_id}...")
        print(f"   (Using local Meta-HNSW for intelligent routing)")
        
        start_time = time.time()
        response = requests.post(
            f"{entry_node_url}/search/p2p",
            json=payload,
            timeout=30
        )
        query_time = time.time() - start_time
        
        response.raise_for_status()
        results = response.json()
        
        print(f"\n✅ Query completed in {query_time:.3f}s")
        print(f"\n📊 Routing Information:")
        print(f"  Entry node:       {results['entry_node']}")
        print(f"  Routing method:   {results.get('routing_method', 'unknown').upper()}")
        print(f"  Target nodes:     {results['target_nodes']}")
        print(f"  Nodes queried:    {results['nodes_queried']}/{k_nodes}")
        print(f"  Total results:    {results['total_results']}")
        
        # Best match info
        best = results['best_match']
        print(f"\n🏆 Best Match:")
        print(f"  Node:             {best['node']}")
        print(f"  Cosine score:     {best['score']:.6f}")
        
        # Per-node breakdown
        print(f"\n📋 Results per Node:")
        print("-" * 60)
        
        for node_name in sorted(results['results_per_node'].keys()):
            node_results = results['results_per_node'][node_name]
            count = len(node_results)
            
            if count > 0:
                scores = [r['score'] for r in node_results]
                max_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                
                # Show if this node had the best match
                best_marker = "🏆 " if node_name == best['node'] else "   "
                
                print(f"{best_marker}{node_name}:")
                print(f"      Results: {count}")
                print(f"      Scores:  {max_score:.6f} (best) / {avg_score:.6f} (avg) / {min_score:.6f} (worst)")
                
                # Show top 3 results from this node
                print(f"      Top matches:")
                for i, result in enumerate(node_results[:3]):
                    vector_id = result.get('id', 'N/A')
                    score = result.get('score', 0.0)
                    payload_info = result.get('payload', {})
                    source = payload_info.get('source_type', 'unknown')
                    index = payload_info.get('index', '?')
                    
                    print(f"        #{i+1}: score={score:.6f} | id={vector_id[:8]}... | source={source} | idx={index}")
                
                print()
        
        # Meta-HNSW effectiveness analysis
        if results.get('routing_method') == 'meta_hnsw':
            print(f"🔍 Meta-HNSW Routing Effectiveness:")
            print(f"  ✅ Smart routing active")
            print(f"  ✅ Queried only {results['nodes_queried']}/{NUM_NODES} nodes ({results['nodes_queried']/NUM_NODES*100:.1f}%)")
            print(f"  ✅ Found best match on {best['node']}")
            print(f"  ✅ Estimated latency reduction: {(1 - results['nodes_queried']/NUM_NODES)*100:.1f}%")
        else:
            print(f"⚠️  Meta-HNSW Routing:")
            print(f"  Fallback to broadcast (Meta-HNSW unavailable)")
        
        print()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Query error: {e}")
        import traceback
        traceback.print_exc()


def print_post_cleanup_report():
    """Print detailed vector counts after cleanup on all nodes."""
    print("\n" + "="*80)
    print("POST-CLEANUP VECTOR DISTRIBUTION (BEFORE ROUTING)")
    print("="*80)
    
    all_counts = []
    total_vectors = 0
    
    # FIXED: Use simple /count endpoint like check_counts() does
    for i in range(NUM_NODES):
        node_url = NODE_URLS[i]
        node_id = f"node{i+1}"
        
        try:
            # Use the working /count endpoint
            response = requests.get(f"{node_url}/count", timeout=5)
            
            if response.status_code == 200:
                count = response.json().get('count', 0)
                all_counts.append((node_id, count))
                total_vectors += count
                print(f"  ✓ {node_id}: {count:>6,} vectors")
            else:
                print(f"  ⚠️  {node_id}: Failed (HTTP {response.status_code})")
                all_counts.append((node_id, 0))
                
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  {node_id}: Error - {e}")
            all_counts.append((node_id, 0))
    
    print("-" * 80)
    
    # Analysis
    if not all_counts:
        print("❌ No data collected")
        return
    
    counts_only = [c for _, c in all_counts]
    max_count = max(counts_only)
    min_count = min(counts_only)
    avg_count = total_vectors / NUM_NODES if NUM_NODES > 0 else 0
    
    print(f"\n📈 Post-Cleanup Analysis:")
    print(f"  Total vectors across all nodes:     {total_vectors:>10,}")
    
    # Expected: 10k bootstrap × 3 replicas = 30k
    expected_bootstrap = BOOTSTRAP_VECTORS * 3
    print(f"  Expected bootstrap (with replicas): {expected_bootstrap:>10,}")
    print(f"  Actual total:                       {total_vectors:>10,}")
    
    if total_vectors == expected_bootstrap:
        print(f"  ✅ Vector count matches expected (cleanup preserved correct replicas)")
    elif total_vectors < expected_bootstrap:
        diff = expected_bootstrap - total_vectors
        print(f"  ⚠️  Vector count LOWER than expected (missing: {diff:,})")
        print(f"     Some replicas may have been incorrectly deleted")
    else:
        diff = total_vectors - expected_bootstrap
        print(f"  ⚠️  Vector count HIGHER than expected (extra: {diff:,})")
        print(f"     Some non-replica vectors were not cleaned up")
    
    # Load balance
    print(f"\n⚖️  Load Balance (Post-Cleanup):")
    print(f"  Average per node:  {avg_count:>10,.1f}")
    print(f"  Min count:         {min_count:>10,} ({all_counts[counts_only.index(min_count)][0]})")
    print(f"  Max count:         {max_count:>10,} ({all_counts[counts_only.index(max_count)][0]})")
    print(f"  Range (max-min):   {max_count - min_count:>10,}")
    
    if avg_count > 0:
        imbalance_pct = ((max_count - min_count) / avg_count * 100)
        print(f"  Imbalance:         {imbalance_pct:>10.2f}%")
        
        if imbalance_pct <= 10:
            print(f"  ✅ Good balance (within 10%)")
        elif imbalance_pct <= 25:
            print(f"  ⚠️  Moderate imbalance (within 25%)")
        else:
            print(f"  ❌ Poor balance (exceeds 25%)")
    
    # Per-node breakdown
    print(f"\n📊 Per-Node Breakdown:")
    for node_id, count in all_counts:
        diff = count - avg_count
        diff_pct = (diff / avg_count * 100) if avg_count > 0 else 0
        status = "✅" if abs(diff_pct) <= 10 else "⚠️" if abs(diff_pct) <= 25 else "❌"
        bar_length = int((count / max_count * 30)) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"  {node_id}: {count:>6,} ({diff:+6.0f}, {diff_pct:+5.1f}%) {status} {bar}")
    
    print()
    print("="*80)
    print()


def main_app():
    print("\n" + "="*60)
    print(f"QDRANT SERVER-SIDE CLUSTERING ({NUM_NODES} NODES)")
    print("="*60)
    print(f"Phase 1: Register peers")
    print(f"Phase 2: Bootstrap {BOOTSTRAP_VECTORS} vectors (broadcast)")
    print(f"Phase 3: Wait for {COORDINATOR_NODE_ID} clustering")
    print(f"Phase 4: Smart routing {NUM_VECTORS - BOOTSTRAP_VECTORS} remaining vectors")
    print(f"Phase 5: Verify distribution")
    print(f"Phase 6: Test Meta-HNSW query routing\n")
    
    time.sleep(2)
    
    start_time = time.time()
    
    # PHASE 1: Register peers
    register_peers()
    
    # PHASE 2: Bootstrap broadcast
    bootstrap_insertions = insert_vectors_bootstrap()
    
    # PHASE 3: Wait for clustering
    if not wait_for_clustering():
        print("❌ Clustering failed, aborting")
        sys.exit(1)
    
    # NEW: Print detailed post-cleanup report
    print("\n" + "="*80)
    print("VERIFYING CLEANUP COMPLETION AND VECTOR DISTRIBUTION")
    print("="*80)
    print_post_cleanup_report()
    
    # NEW: Reduced wait time (cleanup already verified)
    print("--- Waiting 10s for final synchronization before routing... ---")
    time.sleep(10)
    
    # PHASE 4: Smart routing
    routed_insertions = insert_vectors_routed()
    
    # Wait for background ops
    print("\n--- Waiting 15s for background insertions... ---")
    time.sleep(15)
    
    # PHASE 5: Verification (ENHANCED)
    all_insertions = bootstrap_insertions + routed_insertions
    check_counts(all_insertions)
    
    # PHASE 6: Queries (ENHANCED with HNSW)
    run_queries()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*60)
    print(f"✅ TEST COMPLETE")
    print("="*60)
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Vectors inserted:     {NUM_VECTORS:,}")
    print(f"Nodes used:           {NUM_NODES}")
    print(f"Bootstrap phase:      {BOOTSTRAP_VECTORS:,} vectors")
    print(f"Routed phase:         {NUM_VECTORS - BOOTSTRAP_VECTORS:,} vectors")
    print("="*60)

if __name__ == "__main__":
    main_app()