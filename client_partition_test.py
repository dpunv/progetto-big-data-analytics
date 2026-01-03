"""
Client that simulates network partitions during execution to test partition tolerance.

This client does the same operations as client.py but introduces network partitions
at various points to verify the system maintains eventual consistency with no lost vectors.
"""

import server as sv
import json
import sys
from typing import List, Set
import concurrent.futures
import time
import random

start_time = time.time()

print("=" * 60)
print("PARTITION TOLERANCE TEST CLIENT")
print("=" * 60)

# read the data
print("\n[1] Loading data...")
data = []
with open('embeddings_big.json', 'r') as f:
    data = json.load(f)
vectors = [(d['embedding'], d['text']) for d in data]
print(f"    Loaded {len(data)} vectors")

# configuration
num_vectors = 500000  # Smaller for faster testing
num_vectors_before_clustering = 8192
num_servers = 8
replication_factor = 4
batch_size = 256
print(f"\n[2] Configuration:")
print(f"    Vectors: {num_vectors}")
print(f"    Servers: {num_servers}")
print(f"    Replication factor: {replication_factor}")
print(f"    Clustering threshold: {num_vectors_before_clustering}")

# create servers
print("\n[3] Starting servers with network awareness...")
servers = []
for i in range(num_servers):
    # Server with highest ID (7) is initial coordinator
    servers.append(sv.Server(i, i == num_servers - 1, num_vectors_before_clustering, 
                             replication_factor, qdrant_url=":memory:"))

# register peers
for server in servers:
    for peer in servers:
        if server.get_id() == peer.get_id():
            continue
        server.add_peer(peer)
print(f"    {num_servers} servers started and connected")
time.sleep(2) # Wait for heartbeats to propagate and elect coordinator

# ============ Network Simulation Helpers ============

def partition_network(partitions: List[List[int]]):
    """
    Simulate network partition by blocking communication between different groups.
    """
    # Create valid partition sets
    partition_sets = [set(p) for p in partitions]
    
    # Ensure every node is in a partition (or isolated)
    all_nodes = set(range(num_servers))
    covered_nodes = set().union(*partition_sets)
    for node in all_nodes:
        if node not in covered_nodes:
            partition_sets.append({node})
    
    print(f"    Applying partition scheme: {[list(p) for p in partition_sets]}")
    
    # Apply blocking rules
    for i in range(num_servers):
        my_partition = None
        for p in partition_sets:
            if i in p:
                my_partition = p
                break
        
        # Block everyone NOT in my partition
        for j in range(num_servers):
            if i == j: continue
            
            if j not in my_partition:
                servers[i].block_peer(j)
            else:
                servers[i].unblock_peer(j)
    
    # Wait for heartbeats to propagate
    print("    Waiting for heartbeats to stabilize partitions...")
    time.sleep(2) 

def heal_network(merge_indices=None):
    """
    Heal network partitions. 
    If merge_indices provided, only merge those groups (Not implemented here for simplicity, doing full heal).
    To support partial heal we'd need to track current state better.
    For this test, 'heal' means full connectivity unless specified.
    """
    if merge_indices:
        # Complex partial heal logic - skipping for now as main flow uses full heal mostly
        # or specific partition reconfiguration
        print("    Partial heal requested (not fully implemented helper, unblocking manually)")
        pass 
    
    print("    Restoring full connectivity...")
    for server in servers:
        for j in range(num_servers):
            if server.id != j:
                server.unblock_peer(j)
    
    print("    Waiting for heartbeats to discover peers...")
    time.sleep(2)

def get_partition_for_node(node_id):
    """Infer partition based on reachable peers."""
    server = servers[node_id]
    reachable = set([p.get_id() for p in server.get_reachable_peers()])
    reachable.add(node_id)
    return reachable

def get_partition_index(node_id):
    # Helper to find which partition a node is in (0 for left, 1 for right etc)
    # This is slightly tricky without global state, so we approximate
    # partitions usually 0..k
    pass

# ============ Helpers End ============

def print_server_status():
    """Print current status of all servers."""
    print("\n    Server Status:")
    total = 0
    for s in servers:
        coord_status = "COORD" if s.is_coordinator else "     "
        hints = s.hinted_handoff.count()
        count = s.count()
        qsize = s.get_queue_size()
        total += count
        hint_str = f"(hints: {hints})" if hints > 0 else ""
        print(f"      [{s.id}] {coord_status} vectors: {count:5d} (queue: {qsize:5d}) {hint_str}")
    print(f"    Total vectors stored: {total}")
    return total

def wait_for_queues(timeout=300):
    """Wait for all server queues to drain."""
    print(f"    Waiting for server queues to drain (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        total_q = sum(s.get_queue_size() for s in servers)
        any_clustering = any(s.is_clustering() for s in servers)
        if total_q == 0 and not any_clustering:
            print("    All queues drained.")
            return True
        sys.stdout.write(f"\r    Queues pending: {total_q}, Clustering: {any_clustering}   ")
        sys.stdout.flush()
        time.sleep(0.5)
    print("\n    WARNING: Timeout waiting for queues to drain!")
    return False

def send_batch(server_idx, batch_vectors, batch_idx, total_batches):
    """Send a batch of vectors to a server."""
    servers[server_idx].receive_from_client(batch_vectors)
    return batch_idx

# ============ PHASE 1: Normal operation (no partitions) ============
print("\n" + "=" * 60)
print("PHASE 1: Normal operation (inserting first 25%)")
print("=" * 60)

phase1_vectors = int(num_vectors * 0.25)
phase1_batches = phase1_vectors // batch_size

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(phase1_batches):
        batch_vectors = vectors[batch_size*i:batch_size*(i+1)]
        server_idx = i % len(servers)
        futures.append(executor.submit(send_batch, server_idx, batch_vectors, i+1, phase1_batches))
    concurrent.futures.wait(futures)
    
    # Insert remaining vectors that don't fill a full batch
    remainder_start = phase1_batches * batch_size
    if remainder_start < phase1_vectors:
        remainder_vectors = vectors[remainder_start:phase1_vectors]
        servers[0].receive_from_client(remainder_vectors)
    
    wait_for_queues()
    print(f"    Inserted {phase1_vectors} vectors")
    print_server_status()

# ============ PHASE 2: Partition during insertion ============
print("\n" + "=" * 60)
print("PHASE 2: Network partition during insertion")
print("=" * 60)

# Create partition: [0,1,2,3] | [4,5,6,7]
print("\n    >>> CREATING PARTITION: [0,1,2,3] | [4,5,6,7] <<<")
partition_network([[0, 1, 2, 3], [4, 5, 6, 7]])

# Show coordinator status
print("\n    Partition coordinators:")
for s in servers:
    if s.is_coordinator:
        partition = get_partition_for_node(s.id)
        print(f"      Server {s.id} is coordinator of partition {sorted(partition)}")

# Insert more vectors during partition (next 25%)
phase2_start = phase1_vectors
phase2_end = int(num_vectors * 0.5)
phase2_batches = (phase2_end - phase2_start) // batch_size

print(f"\n    Inserting vectors during partition...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(phase2_batches):
        idx = phase2_start // batch_size + i
        batch_vectors = vectors[phase2_start + batch_size*i:phase2_start + batch_size*(i+1)]
        server_idx = i % len(servers)
        futures.append(executor.submit(send_batch, server_idx, batch_vectors, i+1, phase2_batches))
    concurrent.futures.wait(futures)

    # Insert remaining vectors that don't fill a full batch
    remainder_start = phase2_start + phase2_batches * batch_size
    if remainder_start < phase2_end:
        remainder_vectors = vectors[remainder_start:phase2_end]
        servers[0].receive_from_client(remainder_vectors)

    wait_for_queues()
    print(f"    Inserted {phase2_end - phase2_start} vectors during partition")
    print_server_status()

# ============ PHASE 3: Heal and observe reconciliation ============
print("\n" + "=" * 60)
print("PHASE 3: Healing partition")
print("=" * 60)

print("\n    >>> HEALING PARTITION <<<")
heal_network()
wait_for_queues()  # Reconciliation takes time

print("\n    After healing:")
print_server_status()

# ============ PHASE 4: Three-way partition ============
print("\n" + "=" * 60)
print("PHASE 4: Three-way partition")
print("=" * 60)

print("\n    >>> CREATING 3-WAY PARTITION: [0,1] | [2,3,4] | [5,6,7] <<<")
partition_network([[0, 1], [2, 3, 4], [5, 6, 7]])

# Show coordinator status
print("\n    Partition coordinators:")
for s in servers:
    if s.is_coordinator:
        partition = get_partition_for_node(s.id)
        print(f"      Server {s.id} is coordinator of partition {sorted(partition)}")

# Insert next 25% during 3-way partition
phase4_start = phase2_end
phase4_end = int(num_vectors * 0.75)
phase4_batches = (phase4_end - phase4_start) // batch_size

print(f"\n    Inserting vectors during 3-way partition...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(phase4_batches):
        batch_vectors = vectors[phase4_start + batch_size*i:phase4_start + batch_size*(i+1)]
        # Target specific partitions
        if i % 3 == 0:
            server_idx = random.choice([0, 1])  # Left partition
        elif i % 3 == 1:
            server_idx = random.choice([2, 3, 4])  # Middle partition
        else:
            server_idx = random.choice([5, 6, 7])  # Right partition
        futures.append(executor.submit(send_batch, server_idx, batch_vectors, i+1, phase4_batches))
    concurrent.futures.wait(futures)

    # Insert remaining vectors that don't fill a full batch
    remainder_start = phase4_start + phase4_batches * batch_size
    if remainder_start < phase4_end:
        remainder_vectors = vectors[remainder_start:phase4_end]
        servers[0].receive_from_client(remainder_vectors)

    wait_for_queues()
    print(f"    Inserted {phase4_end - phase4_start} vectors during 3-way partition")
    print_server_status()

# ============ PHASE 5: Partial heal (merge left and middle) ============
print("\n" + "=" * 60)
print("PHASE 5: Partial heal (merging some partitions)")
print("=" * 60)

# Merging [0,1] and [2,3,4] -> [0,1,2,3,4] | [5,6,7]
print(f"\n    >>> PARTIAL HEAL: Merging [0,1] with [2,3,4] <<<")
partition_network([[0, 1, 2, 3, 4], [5, 6, 7]])

wait_for_queues()

print("\n    Partition state after partial heal:")
for s in servers:
    if s.is_coordinator:
        partition = get_partition_for_node(s.id)
        print(f"      Server {s.id} is coordinator of partition {sorted(partition)}")

print_server_status()

# ============ PHASE 6: Full heal ============
print("\n" + "=" * 60)
print("PHASE 6: Full heal and final insertions")
print("=" * 60)

print("\n    >>> FULL HEAL <<<")
heal_network()
wait_for_queues()

# Insert final 25%
phase6_start = phase4_end
phase6_end = num_vectors
phase6_batches = (phase6_end - phase6_start) // batch_size

print(f"\n    Inserting final {phase6_end - phase6_start} vectors...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(phase6_batches):
        batch_vectors = vectors[phase6_start + batch_size*i:phase6_start + batch_size*(i+1)]
        server_idx = i % len(servers)
        futures.append(executor.submit(send_batch, server_idx, batch_vectors, i+1, phase6_batches))
    concurrent.futures.wait(futures)
    
    # Insert remaining vectors that don't fill a full batch
    remainder_start = phase6_start + phase6_batches * batch_size
    if remainder_start < phase6_end:
        remainder_vectors = vectors[remainder_start:phase6_end]
        servers[0].receive_from_client(remainder_vectors)
    
    wait_for_queues()
    print(f"    Inserted {phase6_end - phase6_start} vectors")
total_stored = print_server_status()

# ============ VERIFICATION ============
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

# Count unique vector IDs across all servers
all_vector_ids = set()
for s in servers:
    for v in s.store.get_all():
        all_vector_ids.add(v[1])

print(f"\n    Unique vectors across cluster: {len(all_vector_ids)}")
print(f"    Expected unique vectors: {num_vectors}")
print(f"    Total replicated vectors: {total_stored}")
print(f"    Expected replicated (approx): {num_vectors * replication_factor}")

# Check for any remaining hints (should be 0 after full heal)
remaining_hints = sum(s.hinted_handoff.count() for s in servers)
print(f"    Remaining undelivered hints: {remaining_hints}")

# Query test
print("\n    Testing query functionality...")
query_vector = [vectors[0][0]]
results = servers[0].query_from_client(query_vector)
if results:
    print(f"    Query returned {len(results)} results ✓")
else:
    print("    WARNING: Query returned no results!")

# Final verdict
print("\n" + "=" * 60)
if len(all_vector_ids) == num_vectors and remaining_hints == 0:
    print("✓ SUCCESS: All vectors preserved through partitions!")
    print("✓ SUCCESS: No lost data, eventual consistency achieved!")
else:
    missing = num_vectors - len(all_vector_ids)
    if missing > 0:
        print(f"✗ WARNING: {missing} vectors missing!")
    if remaining_hints > 0:
        print(f"✗ WARNING: {remaining_hints} hints not yet delivered")
print("=" * 60)

# Stop servers
for s in servers:
    s.stop()

end_time = time.time()
print(f"\nTotal time: {end_time - start_time:.2f}s", file=sys.stderr)

