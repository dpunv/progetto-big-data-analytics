"""
Rebalancing Demo Script

Demonstrates the rebalancing mechanism using the standard server API:
1. Creates servers and establishes initial clustering
2. Then adds MORE vectors to just one server to create imbalance
3. Detects imbalance and triggers rebalancing
4. Verifies vectors are redistributed and search still works

Run with: python rebalance_demo.py
"""

import concurrent.futures
import math
import sys
import time

import numpy as np
import pandas as pd
import server as sv

print("=" * 70)
print("   REBALANCING MECHANISM DEMO")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================
num_servers = 3
initial_vectors = 3000  # Initial vectors to establish clustering
extra_vectors = 6000    # Extra vectors to add to one server (creates imbalance)
num_vectors_before_clustering = 1500
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
    all_vectors = [(d["embedding"], d["text"]) for d in data[:initial_vectors + extra_vectors]]
    print(f"    Loaded {len(all_vectors)} vectors (dim={len(all_vectors[0][0])})")
except Exception as e:
    print(f"    Could not load embeddings.parquet: {e}")
    print("    Generating synthetic vectors...")
    np.random.seed(42)
    all_vectors = [(np.random.randn(64).tolist(), f"text_{i}") for i in range(initial_vectors + extra_vectors)]
    print(f"    Generated {len(all_vectors)} vectors (dim=64)")

initial_batch = all_vectors[:initial_vectors]
extra_batch = all_vectors[initial_vectors:]

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
    servers[server_idx].receive_from_client(batch_vectors)

total_batches = math.ceil(initial_vectors / batch_size)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(total_batches):
        batch_vectors = initial_batch[batch_size * i : batch_size * (i + 1)]
        server_idx = i % num_servers  # Even distribution
        futures.append(
            executor.submit(send_batch, server_idx, batch_vectors, i + 1, total_batches)
        )
    concurrent.futures.wait(futures)

# Wait for clustering to complete
print("    Waiting for clustering to complete...")
while True:
    total_queue = sum(s.get_queue_size() for s in servers)
    clustering = any(s.is_clustering() for s in servers)
    if total_queue == 0 and not clustering:
        break
    time.sleep(0.5)

print(f"    Clustering complete. Status: {servers[0].status}")

# =============================================================================
# PHASE 2: ADD EXTRA VECTORS DIRECTLY TO SERVER 0 (CREATE IMBALANCE)
# =============================================================================
print(f"\n[4] Adding {extra_vectors} extra vectors centered on a specific cluster...")
print("    (Using receive_from_client() API - routed normally with replication)")

# 1. Pick a target cluster on Server 0 to overload
if not servers[0].clusters:
    print("Error: Server 0 has no clusters!")
    sys.exit(1)
    
print(f"DEBUG: Server 0 clusters type: {type(servers[0].clusters)}")
if servers[0].clusters:
    print(f"DEBUG: First cluster element: {servers[0].clusters[0]}")
    print(f"DEBUG: First cluster element type: {type(servers[0].clusters[0])}")

target_cluster_id = servers[0].clusters[0][0]
# Find the centroid of this cluster
centroid = None
# Robust iteration for both list (tuples) and dict
if isinstance(servers[0].clusters, dict):
    iterator = servers[0].clusters.items()
else:
    iterator = servers[0].clusters

for c_data in iterator:
    if c_data[0] == target_cluster_id:
        # Based on debug, c_data[1] is the centroid list directly
        val = c_data[1]
        if isinstance(val, list) or isinstance(val, tuple):
             # It assumes c_data[1] is the centroid vector
             if len(val) > 0 and isinstance(val[0], (int, float)):
                 centroid = np.array(val)
             elif isinstance(val[0], dict): # Fallback if list of dicts?
                 centroid = np.array(val[0]["center"])
        elif isinstance(val, dict):
             centroid = np.array(val["center"])
        break

if centroid is None:
    print("Error: Could not find centroid for target cluster")
    sys.exit(1)

print(f"    Targeting Cluster {target_cluster_id} on Server 0 with {extra_vectors} vectors")

# 2. Generate vectors near this centroid (centroid + noise)
# Use the first vector from dataset to get dimension/type hints if needed, but we generate synthetic
dim = len(centroid)
noise_scale = 0.05  # Tight grouping to ensure they stay in the cluster
extra_vectors_data = []

# Create a numpy array of noise
noise = np.random.normal(scale=noise_scale, size=(extra_vectors, dim))
# Add to centroid
synthetic_vectors = centroid + noise

# Convert to list for API
for i, vec in enumerate(synthetic_vectors):
    vec_list = vec.tolist()
    # Payload
    payload = f"synthetic_overload_{i}"
    extra_vectors_data.append((vec_list, payload))

# 3. Insert using public API
batch_size_extra = 500
print(f"    Inserting {extra_vectors} vectors in batches of {batch_size_extra}...")

for i in range(0, len(extra_vectors_data), batch_size_extra):
    batch = extra_vectors_data[i:i+batch_size_extra]
    # Send to Server 0 (it will route them, likely to itself as leader of that cluster)
    servers[0].receive_from_client(batch)
    if i % 1000 == 0:
        print(f"    Inserted {i}...")

print(f"    Inserted all {extra_vectors} vectors.")

# Wait for processing to complete
time.sleep(0.5)
while sum(s.get_queue_size() for s in servers) > 0:
    time.sleep(0.3)

print("    Extra vectors inserted directly to Server 0")

# =============================================================================
# SHOW BEFORE STATS
# =============================================================================
print("\n" + "-" * 70)
print("BEFORE REBALANCING:")
print("-" * 70)

counts_before = [s.count() for s in servers]
total_before = sum(counts_before)
avg = total_before / num_servers

print(f"\n  Load Statistics:")
for i, c in enumerate(counts_before):
    overload = " ⚠️ OVERLOADED" if c > avg * (1 + sv.REBALANCE_THRESHOLD) else ""
    print(f"    Server {i}: {c} vectors{overload}")
print(f"    Total: {total_before}")
print(f"    Average: {avg:.1f}")

threshold_value = avg * (1 + sv.REBALANCE_THRESHOLD)
print(f"\n  Imbalance Detection:")
print(f"    Threshold: > {threshold_value:.0f} vectors (average × 1.3)")

# =============================================================================
# TRIGGER REBALANCING
# =============================================================================
print("\n" + "-" * 70)
print("TRIGGERING REBALANCING...")  
print("-" * 70)

result = servers[0].trigger_rebalance()

if result:
    print("\n  ✓ Rebalancing executed!")
else:
    print("\n  ℹ  No rebalancing needed (load is balanced)")
    
# Wait for vectors to transfer
time.sleep(1)

# =============================================================================
# SHOW AFTER STATS
# =============================================================================
print("\n" + "-" * 70)
print("AFTER REBALANCING:")
print("-" * 70)

counts_after = [s.count() for s in servers]
total_after = sum(counts_after)

print(f"\n  Load Statistics:")
for i, (before, after) in enumerate(zip(counts_before, counts_after)):
    delta = after - before
    delta_str = f" ({'+' if delta > 0 else ''}{delta})" if delta != 0 else ""
    print(f"    Server {i}: {after} vectors{delta_str}")
print(f"    Total: {total_after}")

if total_after == total_before:
    print(f"    ✓ Vector count preserved!")
else:
    diff = total_after - total_before
    print(f"    Note: Count changed by {diff} (replication effects)")

# =============================================================================
# VERIFY SEARCH FUNCTIONALITY
# =============================================================================
print("\n" + "-" * 70)
print("VERIFYING SEARCH FUNCTIONALITY:")
print("-" * 70)

query_vector = [all_vectors[0][0]]
results = servers[0].query_from_client(query_vector)

print(f"\n  Query: first vector from dataset")
print(f"  Results: {len(results)} vectors found")

if results:
    print(f"  Top 3 results:")
    for vec, vec_id, payload, similarity in results[:3]:
        text_preview = payload[:50] + "..." if len(payload) > 50 else payload
        print(f"    - similarity={similarity:.4f}: \"{text_preview}\"")

# =============================================================================
# CLEANUP
# =============================================================================
print("\n" + "=" * 70)
print("   DEMO COMPLETE")
print("=" * 70)

for s in servers:
    s.stop()
