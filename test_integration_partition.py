
import pytest
import time
import random
import concurrent.futures
import numpy as np
from server import Server, Peer
from compound_types import *

def partition_network(servers, partitions):
    """
    Simulate network partition by blocking communication between different groups.
    """
    num_servers = len(servers)
    partition_sets = [set(p) for p in partitions]
    
    # Ensure every node is in a partition (or isolated)
    all_nodes = set(range(num_servers))
    covered_nodes = set().union(*partition_sets)
    for node in all_nodes:
        if node not in covered_nodes:
            partition_sets.append({node})
    
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
    time.sleep(2) 

def heal_network(servers):
    """
    Heal network partitions to restore full connectivity.
    """
    num_servers = len(servers)
    for server in servers:
        for j in range(num_servers):
            if server.id != j:
                server.unblock_peer(j)
    
    # Wait for heartbeats to discover peers
    time.sleep(2)


def wait_for_queues(servers, timeout=None, check_hints=False):
    """Wait for all server queues to drain."""
    if timeout is None:
        timeout = 3600

    print(f"    Waiting for queues to drain (timeout={timeout}s)...")
    start = time.time()
    last_print = 0
    while time.time() - start < timeout:
        total_q = sum(s.get_queue_size() for s in servers)
        any_clustering = any(s.is_clustering() for s in servers)
        
        # Check for explicitly dropped vectors
        total_dropped = sum(s.dropped_vectors for s in servers)
        if total_dropped > 0:
            print(f"    FAILURE: Detected {total_dropped} dropped vectors!")
            for s in servers:
                if s.dropped_vectors > 0:
                    print(f"        Server {s.id} dropped {s.dropped_vectors} vectors")
            return False

        # Fail fast if we see hints when we shouldn't
        if check_hints:
             total_hints = sum(s.hinted_handoff.count() for s in servers)
             if total_hints > 0:
                 for s in servers:
                     if s.hinted_handoff.count() > 0:
                         print(f"    FAILURE: Server {s.id} has {s.hinted_handoff.count()} hints!")
                 return False

        # Log progress every 5 seconds
        if time.time() - last_print > 5:
            total_stored = sum(s.count() for s in servers)
            total_hints = sum(s.hinted_handoff.count() for s in servers)
            print(f"    [Status] Queues: {total_q}, Clustering: {any_clustering}, Vectors stored: {total_stored}, Hints: {total_hints}, Dropped: {total_dropped}")
            last_print = time.time()

        if total_q == 0 and not any_clustering:
            print("    Queues drained.")
            return True
        time.sleep(0.1)
    
    print(f"    TIMEOUT: Queues={total_q}, Clustering={any_clustering}")
    return False

def count_unique_vectors(servers):
    """Count unique vector IDs across all servers."""
    all_vector_ids = set()
    for s in servers:
        for v in s.store.get_all():
            all_vector_ids.add(v[1])
    return len(all_vector_ids)

def get_replicated_count(servers):
    count = 0
    for s in servers:
        count += s.count()
    return count

@pytest.fixture
def cluster_setup():
    num_servers = 8
    replication_factor = 4
    # Must be small enough to trigger clustering quickly with test data
    vectors_before_clustering = 10 
    
    import qdrant_module
    
    servers = []
    for i in range(num_servers):
        # Clean up stale collections from previous runs
        qdrant_module.delete_collection(":memory:", f"node_{i}_vectors")
        
        # Last server determines initial coordinator
        is_coord = (i == num_servers - 1)
        # Use :memory: Qdrant for integration tests
        s = Server(i, is_coord, vectors_before_clustering, replication_factor, port=8000+i, qdrant_url=":memory:")
        servers.append(s)
        
    # Fully connect
    for s in servers:
        for peer in servers:
            if s.id != peer.id:
                s.add_peer(peer)
                
    time.sleep(1) # gossip startup
    
    yield servers
    
    for s in servers:
        s.stop()
        # Clean up after test
        qdrant_module.delete_collection(":memory:", f"node_{s.id}_vectors")

def wait_for_network_convergence(servers, timeout=30):
    """Wait for all servers to see each other as active."""
    print(f"    Waiting for network convergence (timeout={timeout}s)...")
    start = time.time()
    num_servers = len(servers)
    expected_peers = num_servers - 1 # Each sees everyone else (excluding self?? No, active_peers includes self)
    # wait server.py: 257 self.active_peers: Set[int] = {id}
    # and 395: currently_active.add(peer_id) so it includes self.
    expected_active_count = num_servers 

    while time.time() - start < timeout:
        converged = True
        for s in servers:
            # Check reachable peers count
            reachable = s.get_reachable_peers()
            # s.get_reachable_peers() implementation: return [p for p in self.peers if p.get_id() in self.active_peers]
            # self.peers includes self??
            # server.py line 244: self.peers = [Peer(self)] -> Yes, initially. 
            # line 298 add_peer appends.
            # So len(s.peers) should be num_servers.
            
            # actually checking s.active_peers set length is safer/direct
            with s.lock:
                active_count = len(s.active_peers)
                
            if active_count < expected_active_count:
                converged = False
                break
        
        if converged:
            print("    Network converged.")
            return True
            
        time.sleep(0.5)
    
    print("    TIMEOUT waiting for convergence.")
    # Print state for debugging
    for s in servers:
        with s.lock:
            print(f"    Server {s.id} sees: {sorted(list(s.active_peers))}")
    return False

def test_partition_tolerance_with_verification(cluster_setup):
    servers = cluster_setup
    num_servers = len(servers)
    
    # Ensure network is fully connected before starting
    assert wait_for_network_convergence(servers, timeout=120), "Network failed to converge at startup"
    
    # Generate synthetic data
    # Increased count for better stress testing
    total_vectors = 1_000_000
    
    # Vector format: (embedding, id, payload)
    # Use circle coordinates to ensure distinctness and norm=1
    # [sin(alpha), cos(alpha)]
    data = []
    for i in range(total_vectors):
        # Evenly spread around the circle to maximize distinctness
        alpha = 2 * np.pi * i / total_vectors
        # Create 384-dim vector to match Server default
        vec = np.zeros(384)
        vec[0] = np.sin(alpha)
        vec[1] = np.cos(alpha)
        vec = vec.tolist() 
        payload = f"vec_{i}"
        data.append((vec, payload))
        
    
    # --- PHASE 1: Normal Insertion (First 40%) ---
    batch1_size = total_vectors * 40 // 100
    batch1_data = data[:batch1_size]
    
    # Send to random servers (client load balancing)
    for i, (vec, payload) in enumerate(batch1_data):
        target = servers[i % num_servers]
        # receive_from_client takes List[(vec, payload)]
        target.receive_from_client([(vec, payload)])
        
    wait_for_queues(servers, check_hints=True)
    
    # Check consistency
    # Since ids are assigned by server, we just count unique
    unique_count = count_unique_vectors(servers)
    assert unique_count == batch1_size, f"Expected {batch1_size} vectors, found {unique_count}"
    
    # --- PHASE 2: Partition [0-3] | [4-7] ---
    partition_network(servers, [[0, 1, 2, 3], [4, 5, 6, 7]])
    
    # --- PHASE 3: Insert during partition (Next 40%) ---
    batch2_size = total_vectors * 40 // 100
    batch2_data = data[batch1_size:batch1_size+batch2_size]
    
    # Send evenly to both partitions
    for i, (vec, payload) in enumerate(batch2_data):
        # Target specific partition based on index to ensure both get writes
        if i % 2 == 0:
            target = servers[i % 4] # 0,1,2,3
        else:
            target = servers[4 + (i % 4)] # 4,5,6,7
            
        target.receive_from_client([(vec, payload)])
        
    wait_for_queues(servers)
    
    # Note: Counts might be disjoint. Total unique should be 80.
    # Hinted handoff should be storing hints for cross-partition replication.
    unique_count_p2 = count_unique_vectors(servers)
    assert unique_count_p2 == batch1_size + batch2_size, "Lost vectors during partition write"
    
    # --- PHASE 4: Heal ---
    heal_network(servers)
    wait_for_queues(servers, timeout=60)
    
    # --- PHASE 5: Insert remaining (20%) ---
    batch3_data = data[batch1_size+batch2_size:]
    for i, (vec, payload) in enumerate(batch3_data):
        target = servers[i % num_servers]
        target.receive_from_client([(vec, payload)])
        
    wait_for_queues(servers)
    
    # --- VERIFICATION 1: Total Count ---
    final_unique = count_unique_vectors(servers)
    assert final_unique == total_vectors, f"Expected {total_vectors} total vectors, found {final_unique}"
    
    # --- VERIFICATION 2: Replication Factor ---
    # Expected total stored nodes approx total_vectors * replication_factor
    total_stored = get_replicated_count(servers)
    # Exact check might be tricky due to async replication timing or duplicates during reconcile
    # But it should be at least target.
    expected_replicated = total_vectors * 4
    assert total_stored >= expected_replicated, f"Replication looks low: {total_stored} < {expected_replicated}"
    
    
    # --- VERIFICATION 3: Query Accuracy ---
    # Pick random vectors and query them.
    # Since routing uses similarity, querying with the exact vector should return it as top result.
    
    # Test 10 random probes
    probes = random.sample(data, 10)
    for vec, payload in probes:
        # Query random server
        inquirer = random.choice(servers)
        results = inquirer.query_from_client([vec])
        
        # Expect at least one result
        assert len(results) > 0, f"Query for {payload} returned no results"
        
        # Check top results
        # Result format: (Vector, VectorId, VectorPayload, Similarity)
        # Note: query_from_client returns sorted by similarity desc
        
        # Due to float32 precision, adjacent vectors on a 100k circle might have similarity 1.0
        # So we check if the target is in the top few results.
        found_target = False
        # results contains top_k candidates (now 20).
        for res in results:
            if res[2] == payload:
                found_target = True
                assert res[3] > 0.99, "Similarity should be near 1.0 for exact match"
                break
        
        if not found_target:
             # Debug print top results
             debug_info = [(r[2], r[3]) for r in results]
             assert False, f"Query mismatch using {inquirer.id}. Expected {payload}, got top results: {debug_info}"

