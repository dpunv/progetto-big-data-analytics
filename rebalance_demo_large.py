"""
Rebalancing Demo Script (Large Scale)

Demonstrates the rebalancing mechanism using the standard server API with a larger cluster:
1. Creates 8 servers and establishes initial clustering
2. Then adds MORE vectors to just one server to create imbalance
3. Detects imbalance and triggers rebalancing
4. Verifies vectors are redistributed and search still works

Run with: python rebalance_demo_large.py
"""

import concurrent.futures
import math
import sys
import time

import numpy as np
import pandas as pd
import server as sv

print("=" * 70)
print("   REBALANCING MECHANISM DEMO (LARGE SCALE)")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================
num_servers = 8
initial_vectors = 8000  # Initial vectors to establish clustering
extra_vectors = 10000    # Extra vectors to add to one server (creates imbalance)
num_vectors_before_clustering = 2000 # Increased slightly for larger scale
replication_factor = 2 
batch_size = 256

print(f"\nConfiguration:")
print(f"  Servers: {num_servers}")
print(f"  Initial vectors: {initial_vectors}")
print(f"  Extra vectors (to create imbalance): {extra_vectors}")
print(f"  Replication factor: {replication_factor}")
print(f"  REBALANCE_THRESHOLD: {sv.REBALANCE_THRESHOLD} ({int(sv.REBALANCE_THRESHOLD*100)}%)")

# =============================================================================
# LOAD EMBEDDINGS
# =============================================================================
print("\n[1] Loading embeddings from parquet...")
try:
    df = pd.read_parquet("embeddings.parquet")
    data = [
        {"embedding": row["embedding"].tolist() if hasattr(row["embedding"], "tolist") else list(row["embedding"]),
         "text": row["sentence"]}
        for _, row in df.iterrows()
    ]
    # Ensure we have enough vectors
    needed = initial_vectors + extra_vectors
    if len(data) < needed:
         print(f"    WARNING: Not enough vectors in parquet ({len(data)}), generating synthetic extras...")
         synthetic_count = needed - len(data)
         np.random.seed(42)
         synthetic = [(np.random.randn(384).tolist(), f"synthetic_{i}") for i in range(synthetic_count)]
         all_vectors = [(d["embedding"], d["text"]) for d in data] + synthetic
    else:
        all_vectors = [(d["embedding"], d["text"]) for d in data[:needed]]
    print(f"    Loaded {len(all_vectors)} vectors (dim={len(all_vectors[0][0])})")
except Exception as e:
    print(f"    Could not load embeddings.parquet: {e}")
    print("    Generating synthetic vectors...")
    np.random.seed(42)
    all_vectors = [(np.random.randn(64).tolist(), f"text_{i}") for i in range(initial_vectors + extra_vectors)]
    print(f"    Generated {len(all_vectors)} vectors (dim=64)")

initial_batch = all_vectors[:initial_vectors]
# We generate extra vectors dynamically based on cluster centroid to guarantee correct routing
# So we don't pre-slice extra_batch here like in the simple demo, but we reserve count.

# =============================================================================
# CREATE SERVERS
# =============================================================================
print("\n[2] Creating servers...")
servers = []
for i in range(num_servers):
    servers.append(
        sv.Server(
            i,
            i == 0,
            num_vectors_before_clustering,
            replication_factor,
            port=9200 + i,
        )
    )

# Register peers
for server in servers:
    for peer in servers:
        if server.get_id() != peer.get_id():
            server.add_peer(peer)

# Wait for peer discovery and set active peers
time.sleep(2)
for server in servers:
    for other in servers:
        server.active_peers.add(other.get_id())
print(f"    {num_servers} servers created and connected")

# =============================================================================
# PHASE 1: INSERT INITIAL VECTORS (EVENLY DISTRIBUTED)
# =============================================================================
print("\n[3] Inserting initial vectors evenly...")

def send_batch(server_idx, batch_vectors, batch_idx, total_batches):
    try:
        data_packet = [(v, p) for v, p in batch_vectors]
        servers[server_idx].receive_from_client(data_packet)
    except Exception as e:
        print(f"    Error sending batch {batch_idx}: {e}")

# Round-robin insertion to simulate even load initially
threads = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    total_initial_batches = math.ceil(len(initial_batch) / batch_size)
    for i in range(total_initial_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(initial_batch))
        batch = initial_batch[start_idx:end_idx]
        
        # Round robin server selection
        target_server_idx = i % num_servers
        threads.append(executor.submit(send_batch, target_server_idx, batch, i, total_initial_batches))

# Wait for completion
for t in threads:
    t.result()

# Wait for eventual consistency
print("    Waiting for clustering to complete...")
# Wait until all servers are clustered
max_retries = 30
while max_retries > 0:
    all_clustered = all(s.status == "clustered" for s in servers)
    if all_clustered:
        break
    time.sleep(1)
    max_retries -= 1

if not all_clustered:
    print("    TIMEOUT: Not all servers transitioned to CLUSTERED status.")
    # Print status
    for s in servers:
        print(f"    Server {s.id}: {s.status}")

# Allow time for index building and initial replication
time.sleep(5) 

# Verify initial distribution
total_vectors = 0
print("\n[3.1] Initial Distribution:")
for s in servers:
    count = s.store.count()
    total_vectors += count
    print(f"    Server {s.id}: {count}")

expected_initial = initial_vectors * replication_factor
print(f"    Total in system: {total_vectors} (Expected ~{expected_initial})")

# =============================================================================
# PHASE 2: CREATE IMBALANCE (REALISTIC CLIENT SIMULATION)
# =============================================================================
print("\n[4] Creating imbalance via realistic client insertion...")

# 1. Pick a target server and a target cluster
target_cluster_id = servers[0].clusters[0][0] # Just pick the first cluster of the first server (likely owns it)
print(f"    Targeting Cluster {target_cluster_id} on Server 0 with {extra_vectors} vectors")

# Find the centroid of this cluster
centroid = None
# Robust iteration for both list (tuples) and dict
if isinstance(servers[0].clusters, dict):
    iterator = servers[0].clusters.items()
else:
    iterator = servers[0].clusters

for c_data in iterator:
    if c_data[0] == target_cluster_id:
        val = c_data[1]
        if isinstance(val, dict):
             centroid = val["center"]
        else: # tuple or list
             centroid = val
        break

if centroid is None:
    print("    ERROR: Could not find centroid for target cluster!")
    sys.exit(1)

# 2. Generate noisy vectors near this centroid
print(f"    Generating {extra_vectors} noisy vectors around centroid...")
np.random.seed(999) # Different seed
dim = len(centroid)
noise_scale = 0.1 # Small noise to ensure they fall into the same cluster

noisy_vectors = []
for i in range(extra_vectors):
    noise = np.random.normal(0, noise_scale, dim)
    noisy_vec = [c + n for c, n in zip(centroid, noise)]
    noisy_vectors.append((noisy_vec, f"imbalance_{i}"))

# 3. Insert these vectors via public API (receive_from_client)
# We send them to Server 0, but since they belong to a cluster Server 0 owns (likely), 
# they should stay there + be replicated.
# Even if sent to another server, they should be routed to Server 0.
print(f"    Inserting {extra_vectors} vectors in batches of {batch_size}...")

threads = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    total_extra_batches = math.ceil(len(noisy_vectors) / batch_size)
    for i in range(total_extra_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(noisy_vectors))
        batch = noisy_vectors[start_idx:end_idx]
        
        # Send strictly to Server 0 to simulate client hitting one endpoint
        threads.append(executor.submit(send_batch, 0, batch, i, total_extra_batches))
        
        if i % 10 == 0:
            print(f"    Inserted {end_idx}...")

for t in threads:
    t.result()
print(f"    Inserted all {extra_vectors} vectors.")


# Wait for replication
time.sleep(5)

# =============================================================================
# PHASE 3: CHECK IMBALANCE AND REBALANCE
# =============================================================================
print("\n" + "-"*70)
print("BEFORE REBALANCING:")
print("-"*70)

# Calculate load stats
counts = [s.store.count() for s in servers]
total_loaded = sum(counts)
avg_load = total_loaded / num_servers
imbalance_threshold = avg_load * (1 + sv.REBALANCE_THRESHOLD)

print(f"\n  Load Statistics:")
for i, count in enumerate(counts):
    print(f"    Server {i}: {count} vectors" + (" ⚠️ OVERLOADED" if count > imbalance_threshold else ""))
print(f"    Total: {total_loaded}")
print(f"    Average: {avg_load}")
print(f"\n  Imbalance Detection:")
print(f"    Threshold: > {int(imbalance_threshold)} vectors (average × {1 + sv.REBALANCE_THRESHOLD})")

print("\n" + "-"*70)
print("TRIGGERING REBALANCING...")
print("-"*70)

rebalance_occured = False
rebalance_occured = False
for s in servers:
    if s.trigger_rebalance():
        rebalance_occured = True

if not rebalance_occured:
    print("\n  ℹ  No rebalancing needed (load is balanced)")
else:
    print("\n  ✓ Rebalancing executed!")
    
# Wait for rebalancing effects (async broadcast)
time.sleep(5)

print("\n" + "-"*70)
print("AFTER REBALANCING:")
print("-"*70)

new_counts = [s.store.count() for s in servers]
total_after = sum(new_counts)

print(f"\n  Load Statistics:")
for i, count in enumerate(new_counts):
    diff = count - counts[i]
    diff_str = f"({'+' if diff > 0 else ''}{diff})" if diff != 0 else ""
    print(f"    Server {i}: {count} vectors {diff_str}")

print(f"    Total: {total_after}")

expected_total = (initial_vectors + extra_vectors) * replication_factor
if abs(total_after - expected_total) < (num_servers * 5): # Allow small epsilon for inflight
    print(f"    ✓ Vector count preserved!")
else:
    print(f"    ❌ VECTOR COUNT MISMATCH! Expected {expected_total}, got {total_after}")

# =============================================================================
# PHASE 4: VERIFY SEARCH
# =============================================================================
print("\n" + "-"*70)
print("VERIFYING SEARCH FUNCTIONALITY:")
print("-"*70)

query_vec = initial_batch[0][0]
print(f"\n  Query: first vector from dataset")

# Query random server
query_server = servers[np.random.randint(0, num_servers)]
try:
    # Pass list containing the vector, as done in rebalance_demo.py
    results = query_server.query_from_client([query_vec], top_k=100)
    print(f"  Results: {len(results)} vectors found")
    if results:
         print(f"  Top 3 results:")
         for r in results[:3]:
             sim = r[1]
             text = r[2]
             print(f"    - similarity={sim:.4f}: \"{text[:45]}...\"")
except Exception as e:
    print(f"  Search failed: {e}")

print("\n" + "="*70)
print("   DEMO COMPLETE")
print("="*70)
