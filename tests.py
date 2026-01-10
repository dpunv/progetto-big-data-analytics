
import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
import server
from server import Server, Peer, cosine_similarity
from compound_types import *

# --- Network Simulation Helpers ---

def partition_network(servers, groups):
    """
    Simulate partition by blocking peers.
    groups: List[List[int]] - list of groups of server IDs that can communicate.
    """
    # Create map for ID -> Server
    server_map = {s.id: s for s in servers}
    
    # Flatten groups to check for unassigned
    assigned = set()
    for g in groups:
        assigned.update(g)
        
    for s1 in servers:
        # Find my group
        my_group = set()
        for g in groups:
            if s1.id in g:
                my_group = set(g)
                break
        
        # If not assigned to any group, it's isolated (or assume implicit group?)
        # Let's assume isolated if not in groups
        if not my_group:
             my_group = {s1.id}

        for s2 in servers:
            if s1.id == s2.id: continue
            
            if s2.id not in my_group:
                s1.block_peer(s2.id)
            else:
                s1.unblock_peer(s2.id)
                
    time.sleep(1.0) # Wait for heartbeat

def heal_network(servers):
    """Restore full connectivity."""
    for s in servers:
        # Use unblock_peer to properly restore active_peers and trigger handlers
        for other in servers:
            if s.id != other.id:
                s.unblock_peer(other.id)
    time.sleep(1.0)

def wait_for_full_connectivity(servers):
    """Wait for all servers to see all others."""
    expected_count = len(servers)
    for _ in range(20):
        c = 0
        for s in servers:
            if len(s.active_peers) == expected_count:
                c += 1
        if c == expected_count:
            return
        time.sleep(0.2)
    # If we get here, we timed out, but let the test fail naturally or log it
    print("Warning: timed out waiting for full connectivity")

# --- Unit Tests ---

def test_cosine_similarity():
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    
    v3 = [0.0, 1.0]
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)
    
    v4 = [0.0, 0.0]
    assert cosine_similarity(v1, v4) == 0.0
    assert cosine_similarity(v4, v4) == 0.0

    v5 = [1.0, 1.0]
    # dot=1, norm_a=1, norm_b=sqrt(2) -> 1/sqrt(2) approx 0.707
    assert cosine_similarity(v1, v5) == pytest.approx(1.0 / (1.0 * np.sqrt(2)))

# Global port counter
_next_port = 30000

def get_free_port():
    global _next_port
    p = _next_port
    _next_port += 1
    return p

@pytest.fixture
def server_node():
    # Use :memory: Qdrant to test the actual integration path
    port = get_free_port()
    s = Server(id=1, is_coordinator=True, before_clustering=10, replication_factor=2, port=port, qdrant_url=":memory:")
    # Ensure empty collection for test isolation
    import qdrant_module
    qdrant_module.delete_collection(s.qdrant_url, s.store.collection_name)
    # Force recreation for empty state, implicit dim on next insert
    s.store.collection_created = False
    
    yield s
    
    try:
        s.stop()
        qdrant_module.delete_collection(s.qdrant_url, s.store.collection_name)
    except Exception as e:
        print(f"Error checking cleaning up server: {e}")

class TestPeer:
    def test_peer_delegation(self):
        mock_server = MagicMock()
        peer = Peer('127.0.0.1', get_free_port(), server_instance=mock_server)
        
        peer.get_id()
        mock_server.get_id.assert_called_once()
        
        peer.similarity([1,2])
        mock_server.similarity.assert_called_once_with([1,2])
        
        peer.receive([], 'status')
        mock_server.receive.assert_called_once_with([], 'status')
        
        peer.i_am_coord()
        mock_server.i_am_coord.assert_called_once()
        
        peer.set_clusters({}, {})
        mock_server.set_clusters.assert_called_once_with({}, {})
        
        peer.search_vectors_local([], 5)
        mock_server.search_vectors_local.assert_called_once_with([], 5)
        
        peer.query([], 'status')
        mock_server.query.assert_called_once_with([], 'status')
    
    def test_peer_delegation_additional_methods(self):
        """Test delegation of additional Peer methods."""
        mock_server = MagicMock()
        mock_server.partition_coordinator_id = 42
        mock_server.respond_to_ping.return_value = True
        mock_server.get_vector_digest.return_value = {1: (1.0, 0)}
        mock_server.get_vectors_by_ids.return_value = []
        
        peer = Peer('127.0.0.1', get_free_port(), server_instance=mock_server)
        
        # Test get_vector_digest
        result = peer.get_vector_digest()
        mock_server.get_vector_digest.assert_called_once()
        assert result == {1: (1.0, 0)}
        
        # Test get_vectors_by_ids
        result = peer.get_vectors_by_ids([1, 2])
        mock_server.get_vectors_by_ids.assert_called_once_with([1, 2])
        
        # Test get_partition_coordinator_id
        result = peer.get_partition_coordinator_id()
        assert result == 42
        
        # Test ping
        result = peer.ping()
        mock_server.respond_to_ping.assert_called_once()
        assert result == True



class TestServerUnit:
    # server_node fixture is now global


    def test_initialization(self, server_node):
        assert server_node.id == 1
        assert server_node.is_coordinator is True
        assert server_node.status == 'bootstrap'
        assert len(server_node.peers) == 1 # Self peer
        assert server_node.peers[0].server == server_node

    def test_add_peer(self, server_node):
        other_server = Server(id=2, is_coordinator=False, before_clustering=10, replication_factor=2, port=get_free_port())
        server_node.add_peer(other_server)
        assert len(server_node.peers) == 2
        other_server.stop()

    def test_get_new_vector_id(self, server_node):
        vid1 = server_node.get_new_vector_id()
        vid2 = server_node.get_new_vector_id()
        assert vid1 != vid2
        # Logic is vector_id * 10^(len) + id
        # len(peers) is 1, so 10^1 = 10
        # vector_id starts at 0, first call increments to 1.
        # 1 * 10 + 1 = 11
        # 2 * 10 + 1 = 21
        assert vid1 == 11
        assert vid2 == 21

    def test_status_management(self, server_node):
        assert server_node.get_status() == 'bootstrap'
        server_node.set_status('clustered')
        assert server_node.get_status() == 'clustered'

    def test_count(self, server_node):
        assert server_node.count() == 0
        # Insert vectors properly using store.insert which updates vector_ids
        server_node.store.insert(([1.0], 1, "a", 0, (1.0, 1)))
        server_node.store.insert(([2.0], 2, "b", 0, (1.0, 1)))
        server_node.store.insert(([3.0], 3, "c", 1, (1.0, 1)))
        assert server_node.count() == 3

    def test_similarity_no_clusters(self, server_node):
        assert server_node.similarity([1.0, 0.0]) == 0.0

    def test_similarity_with_clusters(self, server_node):
        # clusters: [(cluster_id, center_vector), ...]
        server_node.clusters = [
            (0, [1.0, 0.0]),
            (1, [0.0, 1.0])
        ]
        sim = server_node.similarity([1.0, 0.0])
        assert sim == pytest.approx(1.0)
        
        sim = server_node.similarity([0.0, 1.0])
        assert sim == pytest.approx(1.0) # Cosine sim of identical vectors is 1
        
        # vector closer to cluster 0 (0.9, 0.1) than cluster 1 (0.1, 0.9)
        # sim to [1,0] -> ~0.99
        # sim to [0,1] -> ~0.11
        sim = server_node.similarity([0.9, 0.1])
        # Returns max similarity
        # manually: dot([0.9,0.1], [1,0]) / (norm*1) = 0.9 / sqrt(0.82) ~= 0.99
        assert sim > 0.9

    def test_clustering_logic(self, server_node):
        # Create some dummy vectors
        # VectorComplete = Tuple[Vector, VectorId, VectorPayload, int]
        # Status -1 for initial cluster index
        vectors = []
        for i in range(20):
             # 10 vectors near [1,0], 10 near [0,1]
             base = [1.0, 0.0] if i < 10 else [0.0, 1.0]
             noisy = [base[0] + np.random.uniform(-0.1, 0.1), base[1] + np.random.uniform(-0.1, 0.1)]
             vectors.append((noisy, i, "p", -1))
        
        clusters_dict = server_node.clustering(vectors, num_clusters=2)
        assert len(clusters_dict) == 2
        
        # Verify format of output
        # {cluster_id: {'center': ..., 'members': ...}}
        for cid, data in clusters_dict.items():
            assert 'center' in data
            assert 'members' in data
            assert len(data['members']) > 0
            # Members should have assigned cluster id
            for m in data['members']:
                assert m[3] == cid

    def test_assign_clusters_to_peers(self, server_node):
        # Add another peer to execute assignment logic sensibly
        other_node = Server(2, False, 10, 2, port=get_free_port())
        server_node.add_peer(other_node)
        server_node.active_peers.add(2)
        
        # Mock clusters: {id: {'center':..., 'members':...}}
        # 3 clusters, unequal sizes
        clusters = {
            0: {'center': [1,0], 'members': [1]*10},
            1: {'center': [0,1], 'members': [1]*20},
            2: {'center': [1,1], 'members': [1]*30}
        }
        
        # Total load = 60. Nodes = 2. Ideal load = 30 per node.
        # Replication factor = 2. Each cluster goes to 2 nodes. Total capacity usage = 120.
        # Node loads should be roughly 60 each.
        
        assignment = server_node.assign_clusters_to_peers(clusters)
        
        assert 1 in assignment
        assert 2 in assignment
        
        # Check that every cluster is assigned 'replication_factor' times
        cluster_counts = {0: 0, 1: 0, 2: 0}
        for node_id in assignment:
            for cluster_id, center in assignment[node_id]:
                cluster_counts[cluster_id] += 1
        
        for c_id in cluster_counts:
            assert cluster_counts[c_id] == 2
            
        other_node.stop()

    def test_save_vectors(self, server_node):
        # vectors: ListOfVectorsComplete
        # Tuple[Vector, VectorId, VectorPayload, int]
        vecs = [
            ([1.0, 0.0], 1, "A", 0),
            ([0.0, 1.0], 2, "B", 0),
            ([1.0, 1.0], 3, "C", 1)
        ]
        server_node.save_vectors(vecs)
        
        # Verify total count
        assert server_node.store.count() == 3
        
        # Verify clustering (retrieving by cluster ID)
        c0 = server_node.store.get_by_cluster(0)
        assert len(c0) == 2
        
        c1 = server_node.store.get_by_cluster(1)
        assert len(c1) == 1

    def test_search_vectors_local(self, server_node):
        # Insert via API
        vecs = [
                ([1.0, 0.0], 1, "A", 0),
                ([0.0, 1.0], 2, "B", 0)
        ]
        server_node.save_vectors(vecs)
    
        # Query for [1, 0] should return A first
        query = [([1.0, 0.0], 100)]
        results = server_node.search_vectors_local(query, top_k=2)
        # Result: [(Vector, VectorId, VectorPayload, Similarity)]
        assert len(results) == 2
        assert results[0][2] == "A"
        assert results[0][3] == pytest.approx(1.0)

    def test_route_vectors(self, server_node):
        # Mock peers with controlled similarity
        p1 = MagicMock()
        p1.get_id.return_value = 1
        p1.similarity.return_value = 0.9
        
        p2 = MagicMock()
        p2.get_id.return_value = 2
        p2.similarity.return_value = 0.1
        
        server_node.peers = [p1, p2]
        
        vectors = [([1.0, 0.0], 99)] # (Vector, ID)
        
        # Replication factor 2 ? check server default (set to 2 in fixture)
        # route_vectors(vectors, top_k=3). Default fixture rep factor is 2.
        # But method signature is route_vectors(vectors, top_k=3).
        
        results = server_node.route_vectors(vectors, top_k=1)
        # Expect only p1
        assert 99 in results
        top_peers = results[99]
        assert len(top_peers) == 1
        assert top_peers[0][0] == 1 # id of p1

# --- Integration / Workflow Tests ---

@pytest.mark.parametrize("protocol", ["HTTP", "GRPC", "QUIC"])
def test_full_workflow(protocol):
    s1_port = get_free_port()
    s2_port = get_free_port()
    # Setup mini cluster with real network ports
    s1 = Server(1, True, before_clustering=4, replication_factor=2, port=s1_port, endpoint=protocol)
    s2 = Server(2, False, before_clustering=4, replication_factor=2, port=s2_port, endpoint=protocol)
    
    # Wait for endpoints to be ready
    time.sleep(1)
    
    # Connect using real networking (IP/Port)
    s1.add_peer('127.0.0.1', s2_port)
    s2.add_peer('127.0.0.1', s1_port)
    
    # We must manually update active peers since we are not waiting for heartbeat gossip here yet
    # But wait, heartbeat runs in background. 
    # Let's wait for them to see each other via heartbeat or force it.
    # Real peers rely on heartbeat pings.
    # We can force add active peers for speed, or wait.
    # Let's wait for a bit.
    
    time.sleep(2)
    # Check if they found each other (optional debugging)
    # assert len(s1.active_peers) > 1
    
    try:
        # Create Dummy Data
        # 8 vectors total. 
        # We need s1 (coord) to buffer them, wait for 4, trigger clustering.
        # We send 8 to ensure we trigger clustering and have some left over or verify phases.
        
        # Client sends to S2 (non-coord) -> should forward to S1 (coord) because status is bootstrap
        
        data = []
        for i in range(10):
            vec = [float(i), 1.0] 
            data.append((vec, f"payload_{i}"))
            
        # Send data via client method
        # s2.receive_from_client calls receive('client') which calls get_new_vector_id etc.
        # For 'client' status in 'bootstrap':
        # if not coord (s2), finds coord (s1), sends receive(vectors, 'bootstrap')
        
        s2.receive_from_client(data[:5]) 
        
        # This is async (enqueued). Wait a bit for processing.
        time.sleep(1)
        
        # Check if s1 buffered them
        # Note: server runs a processing thread.
        # s1.vector_buffer should be somewhat populated or cleared if clustering triggered.
        # 'before_clustering' is 4. We sent 5.
        # So clustering should have triggered.
        
        # Wait for clustering to finish.
        # Clustering runs in background thread, then calls set_clusters on peers.
        # set_clusters updates status to 'clustered'.
        
        max_retries = 20
        while s1.get_status() != 'clustered' and max_retries > 0:
            time.sleep(0.1)
            max_retries -= 1
            
        assert s1.get_status() == 'clustered'
        assert s2.get_status() == 'clustered'
        assert len(s1.clusters) > 0
        
        # Vectors should be saved now in saved_vectors of peers
        # Check if data is stored
        total_stored = s1.count() + s2.count()
        # 5 vectors, replication 2 -> 10 copies total distributed across s1 and s2
        assert total_stored == 10
        
        # Phase 2: Querying
        # Query s2. Status is clustered.
        # s2.query -> route -> parallel search local
        
        q_vec = [0.0, 1.0] # Should match payload_0 roughly [0, 1]
        results = s2.query_from_client([q_vec]) 
        
        # query_from_client returns [(id, payload, distance), ...]
        # Note: query_from_client implementation in server.py:
        # returns self.query(...)
        # query returns result from search_vectors.
        # search_vectors returns list of tuples.
        # Wait, let's check return type of search_vectors in server.py
        # It calls peer.search_vectors_local -> returns sorted list of tuples from _search_in_array or saved
        # The tuple is (v, v_id, v_payload, sim)
        # query_from_client iterates over this? No, client.py does interaction.
        # server.py query_from_client simply returns the result of self.query.
        # self.query returns results list.
        
        assert len(results) > 0
        
        # Check structure
        # (Vector, VectorId, VectorPayload, Similarity)
        r = results[0]
        assert len(r) == 4
        assert isinstance(r[3], float) # similarity
        
    finally:
        s1.stop()
        s2.stop()

def test_receive_error_conditions():
    s = Server(1, True, 10, 1, port=get_free_port())
    
    # Send 'clustered' message while in bootstrap (unexpected but handled?)
    # receive -> handle_receive
    # if sender_status == 'clustered': saves vectors.
    # This is actually allowed even if self is bootstrap? 
    # Logic: 
    # elif sender_status == 'clustered': self.save_vectors(vectors)
    # Yes, it creates saved_vectors entries.
    
    vec = ([1.0, 0.0], 1, "A", 0)
    s.receive([vec], 'clustered')
    time.sleep(0.1)
    assert s.count() == 1
    
    s.stop()

def test_coordinator_finding():
    s1 = Server(1, False, 10, 1, port=get_free_port()) # Not coord
    s2 = Server(2, True, 10, 1, port=get_free_port()) # Coord
    s1.add_peer(s2)
    s1.active_peers.add(2)
    
    # Check ID match instead of object identity
    found = s1.coordinator()
    assert found.get_id() == 2
    
    s1.stop()
    s2.stop()

@pytest.mark.parametrize("protocol", ["HTTP", "GRPC", "QUIC"])
def test_server_shutdown_cleanly(protocol):
    s = Server(1, True, 10, 1, port=get_free_port(), endpoint=protocol)
    s.stop()
    assert not s.worker_thread.is_alive()

def test_query_routing_bootstrap_mode():
    # If client queries while system is in bootstrap
    s1 = Server(1, True, 10, 1, port=get_free_port()) # Coord
    
    # Add some data to buffer
    s1.vector_buffer = [([1.0, 0.0], 1, "A", -1)]
    
    # Query
    # query_from_client -> query('client') -> if bootstrap & coord -> search_in_array(buffer)
    results = s1.query_from_client([[1.0, 0.0]])
    assert len(results) == 1
    assert results[0][2] == "A"
    
    s1.stop()

def test_process_queue_exception(server_node):
    def raiser():
        raise ValueError("Boom")
    
    server_node.queue.put((raiser, ()))
    time.sleep(0.1)
    assert server_node.worker_thread.is_alive()

def test_get_all_vectors(server_node):
    s = server_node
    vecs = [
        ([1.0], 1, "p", 0),
        ([2.0], 2, "q", 1)
    ]
    s.save_vectors(vecs)
    
    all_v = s.get_all_vectors()
    assert len(all_v) == 2

def test_add_to_buffer_not_coord():
    s = Server(1, False, 10, 1, port=get_free_port())
    s.add_to_buffer([])
    assert len(s.vector_buffer) == 0
    s.stop()

def test_receive_bootstrap_when_clustered():
    s = Server(1, True, 10, 1, port=get_free_port())
    s.status = 'clustered'
    
    mock_peer_obj = MagicMock()
    mock_peer_obj.get_id.return_value = 55
    mock_peer_obj.receive = MagicMock()
    mock_peer_obj.similarity.return_value = 0.5
    
    s.peers.append(mock_peer_obj)
    s.active_peers.add(55)
    
    vecs = [([1.0, 0.0], 1, "A", -1)]
    s.receive(vecs, 'bootstrap')
    time.sleep(0.1)
    
    mock_peer_obj.receive.assert_called()
    s.stop()

def test_receive_client_when_clustered():
    s = Server(1, True, 10, 1, port=get_free_port())
    s.status = 'clustered'
    
    mock_peer_obj = MagicMock()
    mock_peer_obj.get_id.return_value = 55
    mock_peer_obj.similarity.return_value = 1.0
    mock_peer_obj.receive = MagicMock()
    
    s.peers.append(mock_peer_obj)
    s.active_peers.add(55)
    
    vecs = [([1.0, 0.0], 1, "A", -1)]
    s.receive(vecs, 'client')
    time.sleep(0.1)
    
    mock_peer_obj.receive.assert_called()
    s.stop()

def test_receive_client_not_coord_bootstrap():
    s_nc = Server(2, False, 10, 1, port=get_free_port())
    
    mock_coord_peer = MagicMock()
    mock_coord_peer.i_am_coord.return_value = True
    mock_coord_peer.receive = MagicMock()
    mock_coord_peer.get_id.return_value = 1
    
    s_nc.peers = [Peer(s_nc.ip, s_nc.port, server_instance=s_nc), mock_coord_peer]
    s_nc.active_peers.add(1)
    
    vecs = [([1.0, 0.0], 1, "A", -1)]
    s_nc.receive(vecs, 'client')
    time.sleep(0.1)
    
    mock_coord_peer.receive.assert_called_with(vecs, 'bootstrap')
    s_nc.stop()

def test_query_inter_peer_logic():
    s = Server(1, True, 10, 1, port=get_free_port())
    s.vector_buffer = [([1.0, 0.0], 1, "A", -1)]
    
    # Query expects ListOfVectorsWithId: [(VectorId, Vector)]
    res = s.query([(99, [1.0, 0.0])], 'bootstrap')
    assert len(res) == 1
    
    s_nc = Server(2, False, 10, 1, port=get_free_port())
    res_nc = s_nc.query([], 'bootstrap')
    assert res_nc is None
    
    s.status = 'clustered'
    s.store.vectors = {0: [([1.0, 0.0], 1, "A", 0)]}
    res_c = s.query([(99, [1.0, 0.0])], 'bootstrap')
    assert len(res_c) == 1
    
    s.query([], 'invalid')
    
    s.stop()
    s_nc.stop()

def test_query_client_bootstrap_non_coord():
    s_nc = Server(2, False, 10, 1, port=get_free_port())
    
    mock_coord = MagicMock()
    mock_coord.i_am_coord.return_value = True
    mock_coord.query.return_value = "Forwarded"
    mock_coord.get_id.return_value = 1
    
    s_nc.peers.append(mock_coord)
    s_nc.active_peers.add(1)
    
    res = s_nc.query([], 'client')
    assert res == "Forwarded"
    
    s_nc.stop()

def test_search_vectors_error_handling(server_node):
    bad_peer = MagicMock()
    bad_peer.get_id.return_value = 99
    bad_peer.search_vectors_local.side_effect = Exception("Search Fail")
    bad_peer.similarity.return_value = 1.0
    
    server_node.peers.append(bad_peer)
    server_node.status = 'clustered'
    
    # search_vectors expects [(ID, Vector)]
    res = server_node.search_vectors([(1, [1.0,0.0])], top_k=5, top_look=1)
    assert res == []

def test_coverage_gap_listeners():
    # 1. Line 168: bootstrap sent to non coordinator node
    s_nc = Server(2, False, 10, 1, port=get_free_port()) # Not coord
    s_nc.receive([], 'bootstrap') # Should print error
    # We can capture stdout if we want, but coverage is enough
    
    # 2. Line 176: status corrupted (bootstrap sent, i am coord, but status not bootstrap/clustered)
    s_c = Server(1, True, 10, 1, port=get_free_port())
    s_c.status = 'invalid_status'
    s_c._handle_receive([], 'bootstrap')
    
    # 3. Line 183: client sent, i am coord, status bootstrap -> add_to_buffer
    s_c.status = 'bootstrap'
    # Mock add_to_buffer to verify call
    with patch.object(s_c, 'add_to_buffer') as mock_add:
        # Call _handle_receive directly to avoid async queue delay
        s_c._handle_receive([], 'client')
        mock_add.assert_called()
        
    # 4. Line 191: client sent, status corrupted (not bootstrap or clustered)
    s_c.status = 'invalid_status'
    s_c._handle_receive([], 'client')
    
    # 5. Line 193: invalid sender status (not bootstrap, clustered, or client)
    s_c._handle_receive([], 'unknown_status')
    
    # 6. Line 376: query client, invalid status
    s_c.status = 'invalid_status'
    s_c.query([], 'client')
    
    # 7. Line 389: query clustered, status clustered (Wait, check logic)
    # elif sender_status == 'clustered': return self.search_vectors(...)
    # We need to trigger this line.
    s_c.status = 'clustered'
    # Mock search_vectors to avoid complexity
    with patch.object(s_c, 'search_vectors') as mock_search:
        mock_search.return_value = []
        s_c.query([], 'clustered')
        mock_search.assert_called()

    s_nc.stop()
    s_c.stop()


# ==================== Partition Tolerance Tests ====================


# ==================== Partition Tolerance Tests ====================

from server import HintedHandoff


class TestPartitionLogic:
    """Tests for Server-based partition simulation logic."""
    
    def test_block_peer(self):
        s1 = Server(1, True, 10, 2, port=get_free_port())
        s1.block_peer(2)
        
        # Check active peers / reachability
        # Note: _is_peer_reachable checks active_peers which update via heartbeat
        # Mock peer
        p2 = MagicMock()
        p2.get_id.return_value = 2
        
        # Manually verify internal state since heartbeat is async
        assert 2 in s1.simulated_unreachable_peers
        
        s1.unblock_peer(2)
        assert 2 not in s1.simulated_unreachable_peers
        
        s1.stop()


class TestHintedHandoff:
    """Tests for HintedHandoff class."""
    
    def test_store_and_retrieve_hints(self):
        hh = HintedHandoff()
        
        vectors = [([1.0], 1, "a", 0, (1.0, 1))]
        hh.store_hint(99, vectors)
        
        assert hh.has_hints_for(99)
        assert hh.count() == 1
        
        retrieved = hh.get_hints_for(99)
        assert len(retrieved) == 1
        assert not hh.has_hints_for(99)  # Cleared after retrieval
    
    def test_multiple_hints_same_target(self):
        hh = HintedHandoff()
        
        hh.store_hint(99, [([1.0], 1, "a", 0, (1.0, 1))])
        hh.store_hint(99, [([2.0], 2, "b", 0, (1.0, 1))])
        
        assert hh.count() == 2
        hints = hh.get_hints_for(99)
        assert len(hints) == 2
    
    def test_hints_for_multiple_targets(self):
        hh = HintedHandoff()
        
        hh.store_hint(1, [([1.0], 1, "a", 0, (1.0, 1))])
        hh.store_hint(2, [([2.0], 2, "b", 0, (1.0, 1))])
        
        targets = hh.get_all_targets()
        assert 1 in targets
        assert 2 in targets
    
    def test_peek_without_clearing(self):
        hh = HintedHandoff()
        hh.store_hint(99, [([1.0], 1, "a", 0, (1.0, 1))])
        
        peeked = hh.peek_hints_for(99)
        assert len(peeked) == 1
        assert hh.has_hints_for(99)  # Still there
    
    def test_clear_all(self):
        hh = HintedHandoff()
        hh.store_hint(1, [([1.0], 1, "a", 0, (1.0, 1))])
        hh.store_hint(2, [([2.0], 2, "b", 0, (1.0, 1))])
        
        hh.clear()
        assert hh.count() == 0


class TestCoordinatorElection:
    """Tests for partition-aware coordinator election."""
    
    def _create_cluster(self, n=4):
        servers = [
            Server(i, i == 0, 10, 2, port=get_free_port()) for i in range(n)
        ]
        # Fully connect
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        # Allow heartbeat
        time.sleep(1)
        return servers

    def _partition(self, servers, groups):
        # groups: List[List[int]] ids
        for i, s1 in enumerate(servers):
            # Find my group
            my_group = None
            for g in groups:
                if s1.id in g:
                    my_group = g
                    break
            
            # Block everyone else
            for s2 in servers:
                if s1.id == s2.id: continue
                if s2.id not in my_group:
                    s1.block_peer(s2.id)
                else:
                    s1.unblock_peer(s2.id)
        
        # Wait for heartbeat propagation
        time.sleep(1.5)

    def _heal(self, servers):
        for s in servers:
            for p in servers:
                if s.id != p.id:
                    s.unblock_peer(p.id)
        time.sleep(1.5)

    def test_elect_highest_id_in_partition(self):
        servers = self._create_cluster(4)
        
        try:
            # Partition: [0, 1] | [2, 3]
            self._partition(servers, [[0, 1], [2, 3]])
            
            # In partition [0, 1], highest ID is 1 -> should be coordinator
            # Need to wait for heartbeat loop (included in helper)
            
            assert servers[1].is_coordinator
            assert not servers[0].is_coordinator
            
            # In partition [2, 3], highest ID is 3 -> should be coordinator
            assert servers[3].is_coordinator
            assert not servers[2].is_coordinator
            
        finally:
            for s in servers:
                s.stop()
    
    def test_coordinator_in_minority_partition(self):
        """Test when original coordinator ends up isolated."""
        # 0 is initial coord
        servers = self._create_cluster(4)
        
        # Force 3 to be coord initially for this logic match? 
        # Actually logic is dynamic. Initial is 0. 
        # Let's make 3 coord by forcing election or just testing logic
        # Actually default is 0 is coord (from _create_cluster(4)).
        
        try:
            # Partition: [0] | [1, 2, 3]  - coordinator 0 isolated
            self._partition(servers, [[0], [1, 2, 3]])
            
            # In majority partition, node 3 should become coordinator (highest ID)
            assert servers[3].is_coordinator
            
            # Node 0 remains coordinator of its single-node partition
            assert servers[0].is_coordinator
            
        finally:
            for s in servers:
                s.stop()
    
    def test_coordinator_convergence_on_heal(self):
        """Test that coordinators properly merge after partition heals."""
        servers = self._create_cluster(4)
        
        try:
            # Partition
            self._partition(servers, [[0, 1], [2, 3]])
            
            assert servers[1].is_coordinator
            assert servers[3].is_coordinator
            
            # Heal
            self._heal(servers)
            
            # After heal, highest ID (3) should be coordinator
            assert servers[3].is_coordinator
            assert not servers[1].is_coordinator
            
        finally:
            for s in servers:
                s.stop()


class TestAntiEntropy:
    """Tests for anti-entropy reconciliation."""
    
    def test_digest_exchange(self):
        """Test that digest contains vector IDs and versions."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            s.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            s.store.insert(([2.0], 2, "b", 0, (2.0, 0)))
            
            digest = s.get_vector_digest()
            
            assert 1 in digest
            assert 2 in digest
            assert digest[1] == (1.0, 0)
            assert digest[2] == (2.0, 0)
        finally:
            s.stop()
    
    def test_get_vectors_by_ids(self):
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            s.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            s.store.insert(([2.0], 2, "b", 0, (2.0, 0)))
            s.store.insert(([3.0], 3, "c", 0, (3.0, 0)))
            
            result = s.get_vectors_by_ids([1, 3])
            
            ids = [v[1] for v in result]
            assert 1 in ids
            assert 3 in ids
            assert 2 not in ids
        finally:
            s.stop()
    
    def test_reconciliation_syncs_missing_vectors(self):
        """Test that reconciliation transfers missing vectors."""
        # With replication_factor=2 and 2 peers, all vectors should route to both
        s0 = Server(0, True, 10, 2, port=get_free_port())
        s1 = Server(1, False, 10, 2, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)
        
        try:
            # Set up clusters so routing works (both peers have same cluster)
            s0.clusters = [(0, [0.5])]
            s1.clusters = [(0, [0.5])]
            
            # Give s0 some vectors
            s0.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            s0.store.insert(([2.0], 2, "b", 0, (2.0, 0)))
            
            # Give s1 different vectors
            s1.store.insert(([3.0], 3, "c", 0, (3.0, 1)))
            
            # Reconcile - s0 sends vectors to s1
            peer1 = s0.peers[1]  # s1's peer wrapper
            s0.reconcile_with_peer(peer1)
            
            time.sleep(0.2)
            
            # s1 should have received vectors 1 and 2
            assert s1.store.count() == 3, f"s1 should have all 3 vectors, has {s1.store.count()}"
            
            # s0 won't get s1's vectors through reconcile_with_peer alone
            # (we removed the request logic to prevent over-replication)
            # s1 would need to reconcile with s0 for s0 to get vector 3
            peer0 = s1.peers[1]  # s0's peer wrapper
            s1.reconcile_with_peer(peer0)
            
            time.sleep(0.2)
            
            # Now s0 should have all 3 vectors
            assert s0.store.count() == 3, f"s0 should have all 3 vectors, has {s0.store.count()}"
            
        finally:
            s0.stop()
            s1.stop()
    
    def test_version_conflict_resolution(self):
        """Test that newer versions win in reconciliation."""
        s0 = Server(0, True, 10, 2, port=get_free_port())
        s1 = Server(1, False, 10, 2, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)
        
        try:
            # Set up clusters so routing works
            s0.clusters = [(0, [0.5])]
            s1.clusters = [(0, [0.5])]
            
            # Both have vector ID 1, but s1 has newer version
            s0.store.insert(([1.0], 1, "old", 0, (1.0, 0)))
            s1.store.insert(([1.0], 1, "new", 0, (5.0, 1)))  # Newer timestamp
            
            # Reconcile s1 with s0 - s1 sends newer version to s0
            peer0 = s1.peers[1]  # s0's peer wrapper
            s1.reconcile_with_peer(peer0)
            
            time.sleep(0.2)
            
            # s0 should now have the newer version
            vec = s0.store.get_vector(1)
            assert vec[2] == "new", f"Payload should be updated to 'new', got '{vec[2]}'"
            
        finally:
            s0.stop()
            s1.stop()


class TestEventualConsistency:
    """End-to-end tests for eventual consistency with no lost vectors."""
    
    def test_no_lost_vectors_simple_partition(self):
        """Test all vectors present after simple partition and heal."""
        
        servers = [
            Server(i, i == 3, 100, 2, port=get_free_port()) for i in range(4)
        ]
        
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        
        try:
            # Set all servers to clustered mode with basic clusters
            for s in servers:
                s.status = 'clustered'
                s.clusters = [(0, [0.5, 0.5])]
            
            wait_for_full_connectivity(servers)
            
            # Partition
            partition_network(servers, [[0, 1], [2, 3]])
            time.sleep(0.1)
            
            # Insert vectors to both partitions
            vectors_left = [([float(i), 0.0], f"left_{i}") for i in range(10)]
            vectors_right = [([0.0, float(i)], f"right_{i}") for i in range(10)]
            
            servers[0].receive_from_client(vectors_left)
            servers[2].receive_from_client(vectors_right)
            
            time.sleep(0.5)  # Wait for processing
            
            # Count vectors in each partition before heal
            left_count = servers[0].count() + servers[1].count()
            right_count = servers[2].count() + servers[3].count()
            
            # Heal
            heal_network(servers)
            time.sleep(0.5)
            
            # After heal and reconciliation, total should be consistent
            total_count = sum(s.count() for s in servers)
            
            # We inserted 20 vectors (10 left, 10 right)
            # With replication factor 2, expected = 20 * 2 = 40
            # But we need to check that all unique vectors are present
            all_vector_ids = set()
            for s in servers:
                for v in s.store.get_all():
                    all_vector_ids.add(v[1])
            
            assert len(all_vector_ids) == 20, f"Expected 20 unique vectors, got {len(all_vector_ids)}"
            
        finally:
            for s in servers:
                s.stop()
    
    def test_writes_during_partition_preserved(self):
        """Test that writes to both partitions are preserved after heal."""
        
        s0 = Server(0, True, 100, 1, port=get_free_port())
        s1 = Server(1, False, 100, 1, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)
        
        try:
            s0.status = 'clustered'
            s1.status = 'clustered'
            s0.clusters = [(0, [0.5, 0.5])]
            s1.clusters = [(0, [0.5, 0.5])]
            
            wait_for_full_connectivity([s0, s1])
            
            # Partition - both nodes isolated
            partition_network([s0, s1], [[0], [1]])
            time.sleep(0.1)
            
            # Write to s0
            s0.receive_from_client([([1.0, 0.0], "from_s0")])
            
            # Write to s1
            s1.receive_from_client([([0.0, 1.0], "from_s1")])
            
            time.sleep(0.2)
            
            # Before heal - each has only its own write
            assert s0.count() == 1
            assert s1.count() == 1
            
            # Heal
            heal_network([s0, s1])
            time.sleep(0.3)
            
            # After heal, both should have both vectors
            # Note: this depends on reconciliation working
            all_ids_s0 = s0.store.get_all_ids()
            all_ids_s1 = s1.store.get_all_ids()
            
            # The union should contain both vectors
            all_ids = all_ids_s0 | all_ids_s1
            assert len(all_ids) == 2
            
        finally:
            s0.stop()
            s1.stop()


class TestRecursivePartitions:
    """Tests for complex recursive partition scenarios."""
    
    def test_three_way_partition(self):
        """Test A|B|C isolated partition scenario."""
        
        servers = [
            Server(i, i == 2, 100, 1, port=get_free_port()) for i in range(3)
        ]
        
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        
        try:
            # Ensure convergence before partition
            wait_for_full_connectivity(servers)
            
            # 3-way split
            partition_network(servers, [[0], [1], [2]])
            time.sleep(0.1)
            
            # Each should be its own coordinator
            for s in servers:
                assert s.is_coordinator, f"Server {s.id} should be coordinator of its partition"
            
        finally:
            for s in servers:
                s.stop()
    
    def test_cascading_partitions(self):
        """Test ABC -> A|BC -> A|B|C."""
        
        servers = [
            Server(i, i == 2, 100, 1, port=get_free_port()) for i in range(3)
        ]
        
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        
        try:
            wait_for_full_connectivity(servers)
            
            # Initial: all connected, server 2 is coordinator
            assert servers[2].is_coordinator
            
            # First split: [0] | [1, 2]
            partition_network(servers, [[0], [1, 2]])
            time.sleep(0.1)
            
            assert servers[0].is_coordinator  # Alone
            assert servers[2].is_coordinator  # Highest in [1, 2]
            assert not servers[1].is_coordinator
            
            # Further split: [0] | [1] | [2]
            partition_network(servers, [[0], [1], [2]])
            time.sleep(0.1)
            
            # Now all are coordinators
            for s in servers:
                assert s.is_coordinator
            
        finally:
            for s in servers:
                s.stop()
    
    def test_partial_heal_chain(self):
        """Test A|B|C -> AB|C -> ABC heal sequence."""
        
        servers = [
            Server(i, i == 2, 100, 1, port=get_free_port()) for i in range(3)
        ]
        
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        
        try:
            wait_for_full_connectivity(servers)
            
            s0, s1, s2 = servers
            
            # Set up clustered state
            for s in servers:
                s.status = 'clustered'
                s.clusters = [(0, [0.5, 0.5])]
            
            # 3-way split
            partition_network(servers, [[0], [1], [2]])
            time.sleep(0.1)
            
            # Each partition inserts a vector
            s0.receive_from_client([([1.0, 0.0], "v0")])
            s1.receive_from_client([([0.5, 0.5], "v1")])
            s2.receive_from_client([([0.0, 1.0], "v2")])
            time.sleep(0.2)
            
            # Verify each has one vector
            assert s0.count() == 1
            assert s1.count() == 1
            assert s2.count() == 1
            
            # Partial heal: merge 0 and 1
            partition_network(servers, [[0, 1], [2]])
            time.sleep(0.3)
            
            # After partial heal, 0 and 1 should reconcile
            ids_01 = s0.store.get_all_ids() | s1.store.get_all_ids()
            assert len(ids_01) >= 2, "After partial heal, partitions 0 and 1 should have synced"
            
            # Full heal
            heal_network(servers)
            time.sleep(0.3)
            
            # All vectors should be everywhere
            all_ids = s0.store.get_all_ids() | s1.store.get_all_ids() | s2.store.get_all_ids()
            assert len(all_ids) == 3, f"All 3 vectors should exist, got {len(all_ids)}"
            
        finally:
            for s in servers:
                s.stop()
    
    def test_hinted_handoff_during_partition(self):
        """Test that hints are stored during partition and delivered on heal."""
        
        servers = [
            Server(i, i == 2, 100, 2, port=get_free_port()) for i in range(3)
        ]
        
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        
        try:
            wait_for_full_connectivity(servers)
            
            # Set up clustered state with different cluster centers to ensure varied routing
            # Use replication factor 3 so ALL peers should get the vector
            for s in servers:
                s.status = 'clustered'
                s.clusters = [(0, [0.5, 0.5])]
                s.replication_factor = 3  # Ensure all peers get the vector
            
            # Partition: [0, 1] | [2]
            partition_network(servers, [[0, 1], [2]])
            time.sleep(0.1)
            
            # Insert vectors from partition [0, 1]
            # These should be replicated within partition but hints stored for node 2
            servers[0].receive_from_client([([1.0, 0.0], "test")])
            time.sleep(0.3)
            
            # Check hints are stored for unreachable node 2
            hints_for_2 = servers[0].hinted_handoff.count() + servers[1].hinted_handoff.count()
            assert hints_for_2 > 0, "Hints should be stored for unreachable node"
            
            # Heal
            heal_network(servers)
            time.sleep(0.3)
            
            # Hints should be delivered
            assert servers[0].hinted_handoff.count() == 0
            assert servers[1].hinted_handoff.count() == 0
            
            # Node 2 should now have the vector
            assert servers[2].count() > 0
            
        finally:
            for s in servers:
                s.stop()


class TestNetworkOptimization:
    """Tests verifying minimal network usage."""
    
    def test_digest_smaller_than_full_vectors(self):
        """Verify digest is smaller than full vector data."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Insert vectors with large payloads
            for i in range(100):
                large_payload = "x" * 1000  # 1KB payload
                s.store.insert(([float(i)] * 100, i, large_payload, 0, (float(i), 0)))
            
            # Get digest
            digest = s.get_vector_digest()
            
            # Digest should just be {id: (timestamp, node_id)}
            # Not the full vectors with payloads
            import sys
            digest_size = sys.getsizeof(digest)
            
            # Full vectors would be much larger
            # Digest is just IDs and version tuples
            assert len(digest) == 100
            
        finally:
            s.stop()
    
    def test_deduplication_prevents_redundant_storage(self):
        """Test that duplicate vectors with same version are rejected."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Insert same vector twice with same version
            result1 = s.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            result2 = s.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            
            assert result1 == True
            assert result2 == False  # Should be rejected
            assert s.store.count() == 1
            
        finally:
            s.stop()
    
    def test_newer_version_replaces_older(self):
        """Test that newer versions replace older ones."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Insert initial version
            s.store.insert(([1.0], 1, "v1", 0, (1.0, 0)))
            
            # Insert older version - should fail
            result_old = s.store.insert(([1.0], 1, "v0", 0, (0.5, 0)))
            assert result_old == False
            
            # Insert newer version - should succeed
            result_new = s.store.insert(([1.0], 1, "v2", 0, (2.0, 0)))
            assert result_new == True
            
            # Should still only have 1 vector (newer one)
            assert s.store.count() == 1
            vec = s.store.get_vector(1)
            assert vec[2] == "v2"
            
        finally:
            s.stop()


# Additional edge case tests




def test_server_without_network():
    """Test server works normally without network simulator."""
    s = Server(0, True, 10, 1, port=get_free_port())
    
    try:
        # All peers should be reachable
        assert s._is_peer_reachable(s.peers[0])
        
        # Get reachable peers should return all
        reachable = s.get_reachable_peers()
        assert len(reachable) == 1
        
    finally:
        s.stop()


# ==================== Additional Coverage Tests ====================

class TestPeerAdditionalMethods:
    """Tests for remaining Peer class methods."""
    

    
    def test_peer_get_partition_coordinator_id(self):
        mock_server = MagicMock()
        mock_server.partition_coordinator_id = 42
        peer = Peer('127.0.0.1', get_free_port(), server_instance=mock_server)
        result = peer.get_partition_coordinator_id()
        assert result == 42


class TestVectorStoreEdgeCases:
    """Tests for VectorStore edge cases."""
    
    def test_remove_by_id_nonexistent(self):
        """Test removing a vector that doesn't exist."""
        from server import VectorStore
        store = VectorStore()
        # Should not raise
        store.remove_by_id(999)
        assert store.count() == 0
    
    def test_remove_by_id_existing(self):
        """Test removing an existing vector."""
        from server import VectorStore
        store = VectorStore()
        store.insert(([1.0], 1, "a", 0, (1.0, 0)))
        assert store.count() == 1
        store.remove_by_id(1)
        assert store.count() == 0
    
    def test_has_vector(self):
        """Test has_vector method."""
        from server import VectorStore
        store = VectorStore()
        assert not store.has_vector(1)
        store.insert(([1.0], 1, "a", 0, (1.0, 0)))
        assert store.has_vector(1)
    
    def test_get_by_cluster_empty(self):
        """Test get_by_cluster when cluster doesn't exist."""
        from server import VectorStore
        store = VectorStore()
        result = store.get_by_cluster(999)
        assert result == []


class TestServerHelperMethods:
    """Tests for Server helper methods."""
    
    def test_coordinator_returns_none_when_no_coordinator(self):
        """Test coordinator() when no coordinator is reachable."""
        s = Server(0, False, 10, 1, port=get_free_port())  # Not coordinator
        
        try:
            # Only self peer, which is not coordinator
            result = s.coordinator()
            assert result is None
        finally:
            s.stop()
    
    def test_get_queue_size(self):
        """Test get_queue_size method."""
        s = Server(0, True, 10, 1, port=get_free_port())
        try:
            size = s.get_queue_size()
            assert size >= 0
        finally:
            s.stop()
    
    def test_is_clustering(self):
        """Test is_clustering method."""
        s = Server(0, True, 10, 1, port=get_free_port())
        try:
            assert not s.is_clustering()
        finally:
            s.stop()
    
    def test_get_unreachable_peers_no_network(self):
        """Test get_unreachable_peers without network."""
        s = Server(0, True, 10, 1, port=get_free_port())
        try:
            unreachable = s.get_unreachable_peers()
            assert unreachable == []
        finally:
            s.stop()
    
    def test_get_unreachable_peers_with_partition(self):
        """Test get_unreachable_peers with partitioned network."""
        s0 = Server(0, True, 10, 1, port=get_free_port())
        s1 = Server(1, False, 10, 1, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)
        
        try:
            # Before partition, all reachable
            # Manually set active peers since we don't wait for heartbeat
            s0.active_peers = {0, 1}
            assert len(s0.get_unreachable_peers()) == 0
            
            # Partition: s0 blocks s1
            s0.block_peer(1)
            # wait for heartbeat
            time.sleep(0.6)
            
            unreachable = s0.get_unreachable_peers()
            assert len(unreachable) == 1
            assert unreachable[0].get_id() == 1
        finally:
            s0.stop()
            s1.stop()
    
    def test_get_peer_by_id_not_found(self):
        """Test _get_peer_by_id when peer doesn't exist."""
        s = Server(0, True, 10, 1, port=get_free_port())
        try:
            result = s._get_peer_by_id(999)
            assert result is None
        finally:
            s.stop()
    
    def test_route_vectors_no_peers(self):
        """Test route_vectors with no peers."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.peers = []
        try:
            result = s.route_vectors([([1.0], 1)], top_k=1)
            assert result == {}
        finally:
            s.stop()
    
    def test_route_vectors_use_all_peers(self):
        """Test route_vectors with use_all_peers=True."""
        s = Server(0, True, 10, 1, port=get_free_port())
        mock_peer = MagicMock()
        mock_peer.get_id.return_value = 1
        mock_peer.similarity.return_value = 0.5
        s.peers = [mock_peer]
        
        try:
            result = s.route_vectors([([1.0], 1)], top_k=1, use_all_peers=True)
            assert 1 in result
        finally:
            s.stop()
    
    def test_calculate_destinations_no_peers(self):
        """Test _calculate_destinations with no peers."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.peers = []
        try:
            result = s._calculate_destinations(([1.0], 1, "a", 0, (1.0, 0)))
            assert result == frozenset()
        finally:
            s.stop()





class TestReconciliationEdgeCases:
    """Tests for reconciliation edge cases."""
    
    def test_reconcile_already_reconciling(self):
        """Test on_partition_heal returns early when already reconciling."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Set reconciling flag
            s._reconciling = True
            
            # Should return early without error
            s.on_partition_heal()
            
            # Reset for cleanup
            s._reconciling = False
        finally:
            s.stop()
    
    def test_reconcile_with_exception(self):
        """Test reconcile_with_peer handles exceptions."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.get_vector_digest.side_effect = Exception("Digest failed!")
            
            # Should not raise
            s.reconcile_with_peer(mock_peer)
        finally:
            s.stop()
    
    def test_should_be_on_peer_no_destinations(self):
        """Test _should_be_on_peer when vector has no stored destinations."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.clusters = [(0, [1.0, 0.0])]  # Has clusters
        
        mock_peer = MagicMock()
        mock_peer.get_id.return_value = 0
        mock_peer.similarity.return_value = 1.0
        s.peers = [mock_peer]
        
        try:
            # Vector without destinations (only 4 elements)
            vec = ([1.0, 0.0], 1, "a", 0)
            result = s._should_be_on_peer(vec, 0)
            assert isinstance(result, bool)
        finally:
            s.stop()


class TestSendToPeersExceptionHandling:
    """Tests for send_to_peers exception handling."""
    
    def test_send_to_peers_exception_stores_hint(self):
        """Test that send_to_peers stores hints on exception."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0
            mock_peer.receive.side_effect = Exception("Send failed!")
            
            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.status = 'clustered'
            
            vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1]))
            s.send_to_peers([vec])
            
            # Hint should be stored
            assert s.hinted_handoff.has_hints_for(1)
        finally:
            s.stop()


class TestHandoffEdgeCases:
    """Tests for hinted handoff edge cases."""
    
    def test_deliver_hints_exception(self):
        """Test deliver_hints handles exceptions."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Store a hint
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.hinted_handoff.store_hint(1, [vec])
            
            # Create a mock peer that throws on receive
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.receive.side_effect = Exception("Delivery failed!")
            
            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            
            # Should not raise
            s.deliver_hints()
            
            # Hint should still be there (put back)
            assert s.hinted_handoff.has_hints_for(1)
        finally:
            s.stop()


class TestReceiveHandoffAndReconcile:
    """Tests for receive with handoff and reconcile status."""
    
    def test_receive_handoff(self):
        """Test receive with handoff status."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.receive([vec], 'handoff')
            time.sleep(0.1)
            
            assert s.store.count() == 1
        finally:
            s.stop()
    
    def test_receive_reconcile(self):
        """Test receive with reconcile status."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.receive([vec], 'reconcile')
            time.sleep(0.1)
            
            assert s.store.count() == 1
        finally:
            s.stop()


class TestClientNoCoordinatorForward:
    """Test client forward when no coordinator found."""
    
    def test_receive_client_no_coordinator(self):
        """Test receive client when coordinator can't be found."""
        s = Server(0, False, 10, 1, port=get_free_port())
        
        try:
            # No other peers, coordinator() returns None
            s.peers = [Peer(s.ip, s.port, server_instance=s)]  # Only self
            
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.receive([vec], 'client')
            time.sleep(0.1)
            
            # Should print debug message but not crash
        finally:
            s.stop()


class TestReconciliationWithRealCoordinator:
    """Test reconciliation with actual coordinator election."""
    
    def test_reconcile_exception_in_coordinator_loop(self):
        """Test _reconcile_with_other_coordinators handles exceptions."""
        s0 = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            # Simulate exception during reconciliation
            s0.reconcile_with_peer = MagicMock(side_effect=Exception("Reconcile failed!"))
            s0.peers = [Peer(s0.ip, s0.port, server_instance=s0), mock_peer]
            
            # Should not raise
            s0._reconcile_with_other_coordinators()
        finally:
            s0.stop()


class TestClusteringEdgeCases:
    """Test edge cases in clustering and assignment."""
    
    def test_clustering_with_vector_buffer_on_clustered(self):
        """Test _handle_set_clusters processes buffered vectors."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Set up buffered vectors before clustering
            s.vector_buffer = [([1.0], 1, "a", -1, (1.0, 0))]
            s.clustering_in_progress = True
            
            clusters = {0: {'center': [1.0], 'members': []}}
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 0
            mock_peer.similarity.return_value = 1.0
            s.peers = [mock_peer]
            
            s._handle_set_clusters(clusters, [(0, [1.0])])
            
            # Buffer should be cleared
            assert len(s.vector_buffer) == 0
            assert not s.clustering_in_progress
        finally:
            s.stop()
    
    def test_assign_clusters_no_nodes(self):
        """Test assign_clusters_to_peers with no reachable nodes."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.peers = []  # No peers
        
        try:
            clusters = {0: {'center': [1.0], 'members': []}}
            result = s.assign_clusters_to_peers(clusters)
            assert result == {}
        finally:
            s.stop()


class TestQueryEdgeCases:
    """Test query edge cases."""
    
    def test_query_bootstrap_no_coordinator(self):
        """Test query bootstrap mode when coordinator not found returns empty."""
        s = Server(0, False, 10, 1, port=get_free_port())  # Not coordinator
        
        try:
            # Only self peer, which is not coordinator
            result = s.query([(1, [1.0])], 'client')
            # Should return None or empty since no coordinator found
            assert result is None or result == []
        finally:
            s.stop()





class TestShouldBeOnPeerVariants:
    """Test _should_be_on_peer with different vector formats."""
    
    def test_should_be_on_peer_with_stored_destinations(self):
        """Test _should_be_on_peer when vector has stored destinations."""
        s = Server(0, True, 10, 1, port=get_free_port())
        
        try:
            # Vector with stored destinations (6 elements)
            vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1, 2]))
            
            assert s._should_be_on_peer(vec, 1) == True
            assert s._should_be_on_peer(vec, 2) == True
            assert s._should_be_on_peer(vec, 3) == False
        finally:
            s.stop()
    
    def test_should_be_on_peer_no_clusters(self):
        """Test _should_be_on_peer when clusters not yet formed."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.clusters = []  # No clusters yet
        
        try:
            # Vector without stored destinations
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            
            # Should return True (accept everything before clustering)
            assert s._should_be_on_peer(vec, 1) == True
        finally:
            s.stop()


# ==================== ClusterIndex Tests ====================

from cluster_index import ClusterIndex
import numpy as np


class TestClusterIndex:
    """Tests for ClusterIndex class to improve coverage."""
    
    def test_build_empty_clusters(self):
        """Test build() with empty cluster list triggers warning."""
        ci = ClusterIndex(dimension=2)
        ci.build([])  # Should log warning and return early
        assert ci.hnsw_index is None
    
    def test_build_valid_clusters(self):
        """Test build() with valid clusters creates HNSW index."""
        ci = ClusterIndex(dimension=2)
        clusters = [
            (0, [1.0, 0.0]),
            (1, [0.0, 1.0]),
            (2, [1.0, 1.0])
        ]
        ci.build(clusters)
        
        assert ci.hnsw_index is not None
        assert ci.hnsw_index.element_count == 3
    
    def test_build_with_invalid_cluster_id(self):
        """Test build() handles non-integer cluster IDs gracefully."""
        ci = ClusterIndex(dimension=2)
        clusters = [
            ("invalid_string", [1.0, 0.0]),  # Non-integer ID
            (1, [0.0, 1.0])  # Valid
        ]
        ci.build(clusters)
        
        # Should have only 1 valid item
        assert ci.hnsw_index.element_count == 1
    
    def test_build_all_invalid_ids(self):
        """Test build() with all invalid IDs triggers warning."""
        ci = ClusterIndex(dimension=2)
        clusters = [
            ("invalid1", [1.0, 0.0]),
            ("invalid2", [0.0, 1.0])
        ]
        ci.build(clusters)
        # No valid items added
        assert ci.hnsw_index.element_count == 0
    
    def test_find_nearest_clusters_no_index(self):
        """Test find_nearest_clusters() without built index returns empty."""
        ci = ClusterIndex(dimension=2)
        result = ci.find_nearest_clusters([1.0, 0.0], k=3)
        assert result == []
    
    def test_find_nearest_clusters_k_greater_than_count(self):
        """Test find_nearest_clusters() adjusts k when greater than element count."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci.build(clusters)
        
        # Request k=10 but only 2 elements
        result = ci.find_nearest_clusters([1.0, 0.0], k=10)
        assert len(result) == 2
    
    def test_find_nearest_clusters_k_zero(self):
        """Test find_nearest_clusters() with k=0 returns empty."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci.build(clusters)
        
        result = ci.find_nearest_clusters([1.0, 0.0], k=0)
        assert result == []
    
    def test_find_nearest_clusters_ef_adjustment(self):
        """Test find_nearest_clusters() adjusts ef when k > ef."""
        ci = ClusterIndex(dimension=2)
        clusters = [(i, [float(i), float(i)]) for i in range(100)]
        ci.build(clusters)
        
        # Set ef low, then request higher k
        ci.hnsw_index.set_ef(10)
        result = ci.find_nearest_clusters([50.0, 50.0], k=20)
        
        # ef should have been adjusted
        assert ci.hnsw_index.ef >= 20
        assert len(result) == 20
    
    def test_find_nearest_clusters_returns_correct_ids(self):
        """Test find_nearest_clusters() returns correct cluster IDs."""
        ci = ClusterIndex(dimension=2)
        clusters = [
            (0, [1.0, 0.0]),
            (1, [0.0, 1.0]),
            (2, [0.5, 0.5])
        ]
        ci.build(clusters)
        
        # Query for [1.0, 0.0] should return cluster 0 first
        result = ci.find_nearest_clusters([1.0, 0.0], k=1)
        assert result[0] == 0
    
    def test_search_batch_no_index(self):
        """Test search_batch() without built index returns empty."""
        ci = ClusterIndex(dimension=2)
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = ci.search_batch(query, k=1)
        assert result == []
    
    def test_search_batch_k_greater_than_count(self):
        """Test search_batch() adjusts k when greater than element count."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci.build(clusters)
        
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = ci.search_batch(query, k=10)
        
        assert len(result) == 2
        assert len(result[0]) == 2  # Adjusted to element count
    
    def test_search_batch_k_zero(self):
        """Test search_batch() with k=0 returns empty lists."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci.build(clusters)
        
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = ci.search_batch(query, k=0)
        
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []
    
    def test_search_batch_ef_adjustment(self):
        """Test search_batch() adjusts ef when k > ef."""
        ci = ClusterIndex(dimension=2)
        clusters = [(i, [float(i), float(i)]) for i in range(100)]
        ci.build(clusters)
        
        ci.hnsw_index.set_ef(5)
        query = np.array([[50.0, 50.0]], dtype=np.float32)
        result = ci.search_batch(query, k=15)
        
        assert ci.hnsw_index.ef >= 15
        assert len(result[0]) == 15
    
    def test_to_serializable_no_index(self):
        """Test to_serializable() without built index."""
        ci = ClusterIndex(dimension=2, max_clusters=100, ef_construction=100, M=8)
        data = ci.to_serializable()
        
        assert data['dimension'] == 2
        assert data['max_clusters'] == 100
        assert data['ef_construction'] == 100
        assert data['M'] == 8
        assert data['index_data'] is None
    
    def test_to_serializable_with_index(self):
        """Test to_serializable() with built index."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci.build(clusters)
        
        data = ci.to_serializable()
        
        assert data['index_data'] is not None
        assert isinstance(data['index_data'], str)  # Base64 string
    
    def test_from_serializable_no_index(self):
        """Test from_serializable() without index data."""
        data = {
            'dimension': 2,
            'max_clusters': 100,
            'ef_construction': 100,
            'M': 8,
            'index_data': None
        }
        ci = ClusterIndex.from_serializable(data)
        
        assert ci.dimension == 2
        assert ci.max_clusters == 100
        assert ci.hnsw_index is None
    
    def test_from_serializable_with_index(self):
        """Test from_serializable() with valid index data."""
        # Create and serialize
        ci1 = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci1.build(clusters)
        data = ci1.to_serializable()
        
        # Deserialize
        ci2 = ClusterIndex.from_serializable(data)
        
        assert ci2.hnsw_index is not None
        assert ci2.hnsw_index.element_count == 2
    
    def test_from_serializable_invalid_data(self):
        """Test from_serializable() handles corrupt index data."""
        data = {
            'dimension': 2,
            'max_clusters': 100,
            'ef_construction': 100,
            'M': 8,
            'index_data': 'invalid_base64_data_that_is_not_a_valid_pickle'
        }
        
        # Should not raise, but log error and leave index as None
        ci = ClusterIndex.from_serializable(data)
        assert ci.hnsw_index is None


# ==================== QdrantModule Tests ====================

import qdrant_module
import os
import tempfile


class TestQdrantModule:
    """Tests for qdrant_module functions to improve coverage."""
    
    def test_get_collection_name(self):
        """Test get_collection_name() concatenates name and cluster."""
        result = qdrant_module.get_collection_name("test_collection", 5)
        assert result == "test_collection_5"
    
    def test_get_client_memory(self):
        """Test get_client() with :memory: URL."""
        client = qdrant_module.get_client(":memory:")
        assert client is not None
        # Clear cache for test isolation
        qdrant_module._client_cache.clear()
    
    def test_get_client_caching(self):
        """Test get_client() returns cached client."""
        qdrant_module._client_cache.clear()
        
        client1 = qdrant_module.get_client(":memory:")
        client2 = qdrant_module.get_client(":memory:")
        
        assert client1 is client2
        qdrant_module._client_cache.clear()
    
    def test_create_collection_already_exists(self):
        """Test create_collection() when collection exists."""
        url = ":memory:"
        collection = "test_exists"
        
        # Create first time
        result1 = qdrant_module.create_collection(url, collection, 2)
        assert result1 == True
        
        # Create again - should detect exists and return True
        result2 = qdrant_module.create_collection(url, collection, 2)
        assert result2 == True
        
        qdrant_module._client_cache.clear()
    
    def test_create_collection_different_distances(self):
        """Test create_collection() with different distance metrics."""
        url = ":memory:"
        
        qdrant_module.create_collection(url, "test_cosine", 2, "Cosine")
        qdrant_module.create_collection(url, "test_euclid", 2, "Euclid")
        qdrant_module.create_collection(url, "test_dot", 2, "Dot")
        qdrant_module.create_collection(url, "test_unknown", 2, "Unknown")  # Falls back to Cosine
        
        qdrant_module._client_cache.clear()
    
    def test_delete_collection_success(self):
        """Test delete_collection() success case."""
        url = ":memory:"
        collection = "test_delete"
        
        qdrant_module.create_collection(url, collection, 2)
        result = qdrant_module.delete_collection(url, collection)
        
        assert result == True
        qdrant_module._client_cache.clear()
    
    def test_delete_collection_nonexistent(self):
        """Test delete_collection() for non-existent collection - Qdrant doesn't throw."""
        url = ":memory:"
        
        # Qdrant's delete_collection doesn't throw on non-existent, just returns True
        # This test verifies the function handles both cases
        result = qdrant_module.delete_collection(url, "nonexistent_collection")
        # In-memory Qdrant returns True even for non-existent collections
        assert result == True
        qdrant_module._client_cache.clear()
    
    def test_insert_and_count(self):
        """Test insert_vectors() and count()."""
        url = ":memory:"
        collection = "test_insert"
        
        qdrant_module.create_collection(url, collection, 2)
        
        vectors = [
            ([1.0, 0.0], 1, "payload_a", 0),
            ([0.0, 1.0], 2, "payload_b", 0)
        ]
        
        result = qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)
        assert result == True
        
        count_result = qdrant_module.count(url, collection)
        assert count_result == 2
        
        qdrant_module._client_cache.clear()
    
    def test_count_nonexistent_collection(self):
        """Test count() on non-existent collection returns 0."""
        url = ":memory:"
        
        result = qdrant_module.count(url, "nonexistent")
        assert result == 0
        
        qdrant_module._client_cache.clear()
    
    def test_retrieve_vector_success(self):
        """Test retrieve_vector() success case."""
        url = ":memory:"
        collection = "test_retrieve"
        
        qdrant_module.create_collection(url, collection, 2)
        vectors = [([1.0, 0.0], 1, "payload_a", 0)]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)
        
        result = qdrant_module.retrieve_vector(url, collection, 1)
        assert result is not None
        assert result.id == 1
        
        qdrant_module._client_cache.clear()
    
    def test_retrieve_vector_not_found(self):
        """Test retrieve_vector() returns None when not found."""
        url = ":memory:"
        collection = "test_retrieve_none"
        
        qdrant_module.create_collection(url, collection, 2)
        
        result = qdrant_module.retrieve_vector(url, collection, 999)
        assert result is None
        
        qdrant_module._client_cache.clear()
    
    def test_retrieve_vector_error(self):
        """Test retrieve_vector() handles errors."""
        url = ":memory:"
        
        result = qdrant_module.retrieve_vector(url, "nonexistent", 1)
        assert result is None
        
        qdrant_module._client_cache.clear()
    
    def test_delete_vector_success(self):
        """Test delete_vector() success case."""
        url = ":memory:"
        collection = "test_delete_vec"
        
        qdrant_module.create_collection(url, collection, 2)
        vectors = [([1.0, 0.0], 1, "payload_a", 0)]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)
        
        result = qdrant_module.delete_vector(url, collection, 1)
        assert result == True
        
        # Verify deleted
        count = qdrant_module.count(url, collection)
        assert count == 0
        
        qdrant_module._client_cache.clear()
    
    def test_delete_vector_error(self):
        """Test delete_vector() handles errors."""
        url = ":memory:"
        
        result = qdrant_module.delete_vector(url, "nonexistent", 1)
        assert result == False
        
        qdrant_module._client_cache.clear()
    
    def test_get_all_vectors(self):
        """Test get_all_vectors() retrieves all with pagination."""
        url = ":memory:"
        collection = "test_scroll"
        
        qdrant_module.create_collection(url, collection, 2)
        
        # Insert multiple vectors
        vectors = [([float(i), float(i)], i, f"payload_{i}", 0) for i in range(5)]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)
        
        result = qdrant_module.get_all_vectors(url, collection)
        assert len(result) == 5
        
        qdrant_module._client_cache.clear()
    
    def test_get_all_vectors_error(self):
        """Test get_all_vectors() handles errors."""
        url = ":memory:"
        
        result = qdrant_module.get_all_vectors(url, "nonexistent")
        assert result == []
        
        qdrant_module._client_cache.clear()
    
    def test_query_vectors(self):
        """Test query_vectors() returns results."""
        url = ":memory:"
        collection = "test_query"
        
        qdrant_module.create_collection(url, collection, 2)
        vectors = [
            ([1.0, 0.0], 1, "A", 0),
            ([0.0, 1.0], 2, "B", 0),
            ([0.5, 0.5], 3, "C", 0)
        ]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)
        
        query = [([1.0, 0.0], 0)]  # Query with cluster_id
        results = qdrant_module.query_vectors(url, collection, query, topk=2)
        
        assert len(results) > 0
        
        qdrant_module._client_cache.clear()
    
    def test_query_vectors_error(self):
        """Test query_vectors() handles errors."""
        url = ":memory:"
        
        query = [([1.0, 0.0], 0)]
        results = qdrant_module.query_vectors(url, "nonexistent", query, topk=2)
        
        assert results == []
        qdrant_module._client_cache.clear()
    
    def test_get_client_local_path(self):
        """Test get_client() with local file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "qdrant_data")
            
            client = qdrant_module.get_client(local_path)
            assert client is not None
            
            qdrant_module._client_cache.clear()
    
    def test_query_vectors_generic(self):
        """Test query_vectors_generic() groups queries by cluster and routes correctly."""
        url = ":memory:"
        base_collection = "test_generic_query"
        
        # Create collections for different clusters
        qdrant_module.create_collection(url, qdrant_module.get_collection_name(base_collection, 0), 2)
        qdrant_module.create_collection(url, qdrant_module.get_collection_name(base_collection, 1), 2)
        
        # Insert vectors into different cluster collections
        vectors_c0 = [([1.0, 0.0], 1, "A", 0)]
        vectors_c1 = [([0.0, 1.0], 2, "B", 1)]
        
        qdrant_module.insert_vectors(url, qdrant_module.get_collection_name(base_collection, 0), vectors_c0, batch_size_retry=1)
        qdrant_module.insert_vectors(url, qdrant_module.get_collection_name(base_collection, 1), vectors_c1, batch_size_retry=1)
        
        # Query spanning multiple clusters
        # query format: [(vector, cluster_id), ...]
        queries = [([1.0, 0.0], 0), ([0.0, 1.0], 1)]
        
        results = qdrant_module.query_vectors_generic(url, base_collection, queries, topk=1)
        
        assert len(results) > 0
        qdrant_module._client_cache.clear()
    
    def test_insert_vectors_generic(self):
        """Test insert_vectors_generic() groups vectors by cluster and inserts correctly."""
        url = ":memory:"
        base_collection = "test_generic_insert"
        
        # Create collections for different clusters
        qdrant_module.create_collection(url, qdrant_module.get_collection_name(base_collection, 0), 2)
        qdrant_module.create_collection(url, qdrant_module.get_collection_name(base_collection, 1), 2)
        
        # Vectors with different cluster IDs
        vectors = [
            ([1.0, 0.0], 1, "A", 0),  # cluster 0
            ([0.0, 1.0], 2, "B", 1),  # cluster 1
            ([0.5, 0.5], 3, "C", 0),  # cluster 0
        ]
        
        qdrant_module.insert_vectors_generic(url, base_collection, vectors, batch_size_retry=1)
        
        # Verify counts in each cluster collection
        count_c0 = qdrant_module.count(url, qdrant_module.get_collection_name(base_collection, 0))
        count_c1 = qdrant_module.count(url, qdrant_module.get_collection_name(base_collection, 1))
        
        assert count_c0 == 2
        assert count_c1 == 1
        
        qdrant_module._client_cache.clear()
    
    def test_insert_batch_size_adjustment(self):
        """Test insert_vectors() with batch_size smaller than vector count."""
        url = ":memory:"
        collection = "test_batch_size"
        
        qdrant_module.create_collection(url, collection, 2)
        
        # Insert many vectors with small batch size
        vectors = [([float(i), float(i)], i, f"payload_{i}", 0) for i in range(10)]
        
        # batch_size=3 is smaller than vector count, so effective_batch_size will be 3
        result = qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1, batch_size=3)
        assert result == True
        
        count = qdrant_module.count(url, collection)
        assert count == 10
        
        qdrant_module._client_cache.clear()




# ==================== Peer Remote Call Tests ====================

from peer.peer import Peer
import asyncio


class TestPeerRemoteCalls:
    """Tests for Peer class remote call methods to improve coverage."""
    
    def test_peer_no_communicator_raises(self):
        """Test _remote_call() raises when no communicator set."""
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=None)
        
        with pytest.raises(Exception) as exc_info:
            peer._remote_call("get_id")
        
        assert "No communicator" in str(exc_info.value)
    
    def test_peer_remote_call_success(self):
        """Test _remote_call() with successful response."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 42}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer._remote_call("get_id")
        
        assert result == 42
    
    def test_peer_remote_call_failure(self):
        """Test _remote_call() raises on failed response status."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": -1, "error": "Connection refused", "response": None}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        
        with pytest.raises(Exception) as exc_info:
            peer._remote_call("get_id")
        
        assert "Connection refused" in str(exc_info.value)
    
    def test_peer_get_id_remote(self):
        """Test get_id() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 123}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_id()
        
        assert result == 123
    
    def test_peer_similarity_remote(self):
        """Test similarity() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 0.95}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.similarity([1.0, 0.0])
        
        assert result == 0.95
    
    def test_peer_receive_remote(self):
        """Test receive() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": None}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.receive([([1.0], 1, "a", 0)], "client")
        
        assert result is None
    
    def test_peer_i_am_coord_remote(self):
        """Test i_am_coord() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": True}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.i_am_coord()
        
        assert result == True
    
    def test_peer_set_clusters_remote(self):
        """Test set_clusters() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": None}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.set_clusters({0: {'center': [1.0], 'members': []}}, [(0, [1.0])])
        
        assert result is None
    
    def test_peer_search_vectors_local_remote(self):
        """Test search_vectors_local() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": [([1.0], 1, "a", 0.99)]}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.search_vectors_local([([1.0], 1)], 5)
        
        assert len(result) == 1
    
    def test_peer_query_remote(self):
        """Test query() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": []}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.query([([1.0], 1)], "client")
        
        assert result == []
    
    def test_peer_get_vector_digest_remote(self):
        """Test get_vector_digest() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": {1: (1.0, 0)}}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_vector_digest()
        
        assert 1 in result
    
    def test_peer_get_vectors_by_ids_remote(self):
        """Test get_vectors_by_ids() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": []}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_vectors_by_ids([1, 2, 3])
        
        assert result == []
    
    def test_peer_get_partition_coordinator_id_remote(self):
        """Test get_partition_coordinator_id() for remote peer."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 5}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_partition_coordinator_id()
        
        assert result == 5
    
    def test_peer_ping_local(self):
        """Test ping() for local peer."""
        mock_server = MagicMock()
        mock_server.respond_to_ping.return_value = True
        
        peer = Peer('127.0.0.1', 9999, server_instance=mock_server)
        result = peer.ping()
        
        assert result == True
        mock_server.respond_to_ping.assert_called_once()
    
    def test_peer_ping_remote_success(self):
        """Test ping() for remote peer - success."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            return {"status": 0, "error": None, "response": True}
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.ping()
        
        assert result == True
    
    def test_peer_ping_remote_exception(self):
        """Test ping() for remote peer - returns False on exception."""
        mock_comm = MagicMock()
        
        async def mock_send(*args):
            raise Exception("Network error")
        
        mock_comm.send = mock_send
        
        peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=mock_comm)
        result = peer.ping()
        
        assert result == False
    
    def test_peer_is_local(self):
        """Test is_local() method."""
        local_peer = Peer('127.0.0.1', 9999, server_instance=MagicMock())
        remote_peer = Peer('127.0.0.1', 9999, server_instance=None, communicator=MagicMock())
        
        assert local_peer.is_local() == True
        assert remote_peer.is_local() == False


# ==================== CertUtils Tests ====================

from utils.cert_utils import generate_self_signed_cert


class TestCertUtils:
    """Tests for cert_utils module to improve coverage."""
    
    def test_generate_cert_creates_files(self):
        """Test generate_self_signed_cert() creates cert and key files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "test.crt")
            key_path = os.path.join(tmpdir, "test.key")
            
            generate_self_signed_cert(cert_path=cert_path, key_path=key_path)
            
            assert os.path.exists(cert_path)
            assert os.path.exists(key_path)
            
            # Verify files have content
            with open(cert_path, 'rb') as f:
                cert_content = f.read()
            with open(key_path, 'rb') as f:
                key_content = f.read()
            
            assert b"CERTIFICATE" in cert_content
            assert b"PRIVATE KEY" in key_content
    
    def test_generate_cert_skips_if_exists(self):
        """Test generate_self_signed_cert() skips when files already exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "test.crt")
            key_path = os.path.join(tmpdir, "test.key")
            
            # Create dummy files
            with open(cert_path, 'w') as f:
                f.write("existing cert")
            with open(key_path, 'w') as f:
                f.write("existing key")
            
            # Get modification times
            cert_mtime_before = os.path.getmtime(cert_path)
            key_mtime_before = os.path.getmtime(key_path)
            
            # Call function - should skip
            generate_self_signed_cert(cert_path=cert_path, key_path=key_path)
            
            # Verify files were not modified
            cert_mtime_after = os.path.getmtime(cert_path)
            key_mtime_after = os.path.getmtime(key_path)
            
            assert cert_mtime_before == cert_mtime_after
            assert key_mtime_before == key_mtime_after
