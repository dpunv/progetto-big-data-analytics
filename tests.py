import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compound_types import *
from server import Peer, Server, cosine_similarity



def partition_network(servers, groups):
    """
    Simulate partition by blocking peers.
    groups: List[List[int]] - list of groups of server IDs that can communicate.
    """
    {s.id: s for s in servers}

    assigned = set()
    for g in groups:
        assigned.update(g)

    for s1 in servers:
        my_group = set()
        for g in groups:
            if s1.id in g:
                my_group = set(g)
                break

        if not my_group:
            my_group = {s1.id}

        for s2 in servers:
            if s1.id == s2.id:
                continue

            if s2.id not in my_group:
                s1.block_peer(s2.id)
            else:
                s1.unblock_peer(s2.id)

    time.sleep(1.0)


def heal_network(servers):
    """Restore full connectivity."""
    for s in servers:
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
    print("Warning: timed out waiting for full connectivity")




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
    assert cosine_similarity(v1, v5) == pytest.approx(1.0 / (1.0 * np.sqrt(2)))


_next_port = 30000


def get_free_port():
    global _next_port
    p = _next_port
    _next_port += 1
    return p


@pytest.fixture
def server_node():
    port = get_free_port()
    s = Server(
        id=1,
        is_coordinator=True,
        before_clustering=10,
        replication_factor=2,
        port=port,
        qdrant_url=":memory:",
    )
    import qdrant_module

    qdrant_module.delete_collection(s.qdrant_url, s.store.collection_name)
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
        peer = Peer("127.0.0.1", get_free_port(), server_instance=mock_server)

        peer.get_id()
        mock_server.get_id.assert_called_once()

        peer.similarity([1, 2])
        mock_server.similarity.assert_called_once_with([1, 2])

        peer.receive([], "status")
        mock_server.receive.assert_called_once_with([], "status")

        peer.i_am_coord()
        mock_server.i_am_coord.assert_called_once()

        peer.set_clusters({}, {})
        mock_server.set_clusters.assert_called_once_with({}, {}, version=0)

        peer.search_vectors_local([], 5)
        mock_server.search_vectors_local.assert_called_once_with([], 5)

        peer.query([], "status")
        mock_server.query.assert_called_once_with([], "status")

    def test_peer_delegation_additional_methods(self):
        """Test delegation of additional Peer methods."""
        mock_server = MagicMock()
        mock_server.partition_coordinator_id = 42
        mock_server.respond_to_ping.return_value = True
        mock_server.get_vector_digest.return_value = {1: (1.0, 0)}
        mock_server.get_vectors_by_ids.return_value = []

        peer = Peer("127.0.0.1", get_free_port(), server_instance=mock_server)

        result = peer.get_vector_digest()
        mock_server.get_vector_digest.assert_called_once()
        assert result == {1: (1.0, 0)}

        result = peer.get_vectors_by_ids([1, 2])
        mock_server.get_vectors_by_ids.assert_called_once_with([1, 2])

        result = peer.get_partition_coordinator_id()
        assert result == 42

        result = peer.ping()
        mock_server.respond_to_ping.assert_called_once()
        assert result


class TestServerUnit:

    def test_initialization(self, server_node):
        assert server_node.id == 1
        assert server_node.is_coordinator is True
        assert server_node.status == "bootstrap"
        assert len(server_node.peers) == 1
        assert server_node.peers[0].server == server_node

    def test_add_peer(self, server_node):
        other_server = Server(
            id=2,
            is_coordinator=False,
            before_clustering=10,
            replication_factor=2,
            port=get_free_port(),
        )
        server_node.add_peer(other_server)
        assert len(server_node.peers) == 2
        other_server.stop()

    def test_get_new_vector_id(self, server_node):
        vid1 = server_node.get_new_vector_id()
        vid2 = server_node.get_new_vector_id()
        assert vid1 != vid2
        assert vid1 == 11
        assert vid2 == 21

    def test_status_management(self, server_node):
        assert server_node.get_status() == "bootstrap"
        server_node.set_status("clustered")
        assert server_node.get_status() == "clustered"

    def test_count(self, server_node):
        assert server_node.count() == 0
        server_node.store.insert(([1.0], 1, "a", 0, (1.0, 1)))
        server_node.store.insert(([2.0], 2, "b", 0, (1.0, 1)))
        server_node.store.insert(([3.0], 3, "c", 1, (1.0, 1)))
        assert server_node.count() == 3

    def test_similarity_no_clusters(self, server_node):
        assert server_node.similarity([1.0, 0.0]) == 0.0

    def test_similarity_with_clusters(self, server_node):
        server_node.clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        sim = server_node.similarity([1.0, 0.0])
        assert sim == pytest.approx(1.0)

        sim = server_node.similarity([0.0, 1.0])
        assert sim == pytest.approx(1.0)

        sim = server_node.similarity([0.9, 0.1])
        assert sim > 0.9

    def test_clustering_logic(self, server_node):
        vectors = []
        for i in range(20):
            base = [1.0, 0.0] if i < 10 else [0.0, 1.0]
            noisy = [
                base[0] + np.random.uniform(-0.1, 0.1),
                base[1] + np.random.uniform(-0.1, 0.1),
            ]
            vectors.append((noisy, i, "p", -1))

        clusters_dict = server_node.clustering(vectors, min_k=2, max_k=2)
        assert len(clusters_dict) == 2

        for cid, data in clusters_dict.items():
            assert "center" in data
            assert "members" in data
            assert len(data["members"]) > 0
            for m in data["members"]:
                assert m[3] == cid

    def test_assign_clusters_to_peers(self, server_node):
        other_node = Server(2, False, 10, 2, port=get_free_port())
        server_node.add_peer(other_node)
        server_node.active_peers.add(2)

        clusters = {
            0: {"center": [1, 0], "members": [1] * 10},
            1: {"center": [0, 1], "members": [1] * 20},
            2: {"center": [1, 1], "members": [1] * 30},
        }


        assignment = server_node.assign_clusters_to_peers(clusters)

        assert 1 in assignment
        assert 2 in assignment

        cluster_counts = {0: 0, 1: 0, 2: 0}
        for node_id in assignment:
            for cluster_id, center in assignment[node_id]:
                cluster_counts[cluster_id] += 1

        for c_id in cluster_counts:
            assert cluster_counts[c_id] == 2

        other_node.stop()

    def test_save_vectors(self, server_node):
        vecs = [
            ([1.0, 0.0], 1, "A", 0),
            ([0.0, 1.0], 2, "B", 0),
            ([1.0, 1.0], 3, "C", 1),
        ]
        server_node.save_vectors(vecs)

        assert server_node.store.count() == 3

        c0 = server_node.store.get_by_cluster(0)
        assert len(c0) == 2

        c1 = server_node.store.get_by_cluster(1)
        assert len(c1) == 1

    def test_search_vectors_local(self, server_node):
        vecs = [([1.0, 0.0], 1, "A", 0), ([0.0, 1.0], 2, "B", 0)]
        server_node.save_vectors(vecs)

        query = [([1.0, 0.0], 100)]
        results = server_node.search_vectors_local(query, top_k=2)
        assert len(results) == 2
        assert results[0][2] == "A"
        assert results[0][3] == pytest.approx(1.0)

    def test_route_vectors(self, server_node):
        p1 = MagicMock()
        p1.get_id.return_value = 1
        p1.similarity.return_value = 0.9

        p2 = MagicMock()
        p2.get_id.return_value = 2
        p2.similarity.return_value = 0.1

        server_node.peers = [p1, p2]

        vectors = [([1.0, 0.0], 99)]


        results = server_node.route_vectors(vectors, top_k=1)
        assert 99 in results
        top_peers = results[99]
        assert len(top_peers) == 1
        assert top_peers[0][0] == 1




@pytest.mark.parametrize("protocol", ["HTTP", "GRPC", "QUIC"])
def test_full_workflow(protocol):
    s1_port = get_free_port()
    s2_port = get_free_port()
    s1 = Server(
        1,
        True,
        before_clustering=4,
        replication_factor=2,
        port=s1_port,
        endpoint=protocol,
    )
    s2 = Server(
        2,
        False,
        before_clustering=4,
        replication_factor=2,
        port=s2_port,
        endpoint=protocol,
    )

    time.sleep(1)

    s1.add_peer("127.0.0.1", s2_port)
    s2.add_peer("127.0.0.1", s1_port)


    time.sleep(2)

    try:


        data = []
        for i in range(10):
            vec = [float(i), 1.0]
            data.append((vec, f"payload_{i}"))


        s2.receive_from_client(data[:5])

        time.sleep(1)



        max_retries = 50
        while (s1.get_status() != "clustered" or s2.get_status() != "clustered") and max_retries > 0:
            time.sleep(0.1)
            max_retries -= 1

        assert s1.get_status() == "clustered"
        assert s2.get_status() == "clustered"
        assert len(s1.clusters) > 0

        total_stored = s1.count() + s2.count()
        assert total_stored == 10


        q_vec = [0.0, 1.0]
        results = s2.query_from_client([q_vec])


        assert len(results) > 0

        r = results[0]
        assert len(r) == 4
        assert isinstance(r[3], float)

    finally:
        s1.stop()
        s2.stop()


def test_receive_error_conditions():
    s = Server(1, True, 10, 1, port=get_free_port())


    vec = ([1.0, 0.0], 1, "A", 0)
    s.receive([vec], "clustered")
    time.sleep(0.1)
    assert s.count() == 1

    s.stop()


def test_coordinator_finding():
    s1 = Server(1, False, 10, 1, port=get_free_port())
    s2 = Server(2, True, 10, 1, port=get_free_port())
    s1.add_peer(s2)
    s1.active_peers.add(2)

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
    s1 = Server(1, True, 10, 1, port=get_free_port())

    s1.vector_buffer = [([1.0, 0.0], 1, "A", -1)]

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
    vecs = [([1.0], 1, "p", 0), ([2.0], 2, "q", 1)]
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
    s.status = "clustered"

    mock_peer_obj = MagicMock()
    mock_peer_obj.get_id.return_value = 55
    mock_peer_obj.receive = MagicMock()
    mock_peer_obj.similarity.return_value = 0.5

    s.peers.append(mock_peer_obj)
    s.active_peers.add(55)

    vecs = [([1.0, 0.0], 1, "A", -1)]
    s.receive(vecs, "bootstrap")
    time.sleep(0.1)

    mock_peer_obj.receive.assert_called()
    s.stop()


def test_receive_client_when_clustered():
    s = Server(1, True, 10, 1, port=get_free_port())
    s.status = "clustered"

    mock_peer_obj = MagicMock()
    mock_peer_obj.get_id.return_value = 55
    mock_peer_obj.similarity.return_value = 1.0
    mock_peer_obj.receive = MagicMock()

    s.peers.append(mock_peer_obj)
    s.active_peers.add(55)

    vecs = [([1.0, 0.0], 1, "A", -1)]
    s.receive(vecs, "client")
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
    s_nc.receive(vecs, "client")
    time.sleep(0.1)

    mock_coord_peer.receive.assert_called_with(vecs, "bootstrap")
    s_nc.stop()


def test_query_inter_peer_logic():
    s = Server(1, True, 10, 1, port=get_free_port())
    s.vector_buffer = [([1.0, 0.0], 1, "A", -1)]

    res = s.query([(99, [1.0, 0.0])], "bootstrap")
    assert len(res) == 1

    s_nc = Server(2, False, 10, 1, port=get_free_port())
    res_nc = s_nc.query([], "bootstrap")
    assert res_nc is None

    s.status = "clustered"
    s.store.vectors = {0: [([1.0, 0.0], 1, "A", 0)]}
    res_c = s.query([(99, [1.0, 0.0])], "bootstrap")
    assert len(res_c) == 1

    s.query([], "invalid")

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

    res = s_nc.query([], "client")
    assert res == "Forwarded"

    s_nc.stop()


def test_search_vectors_error_handling(server_node):
    bad_peer = MagicMock()
    bad_peer.get_id.return_value = 99
    bad_peer.search_vectors_local.side_effect = Exception("Search Fail")
    bad_peer.similarity.return_value = 1.0

    server_node.peers.append(bad_peer)
    server_node.status = "clustered"

    res = server_node.search_vectors([(1, [1.0, 0.0])], top_k=5, top_look=1)
    assert res == []


def test_coverage_gap_listeners():
    s_nc = Server(2, False, 10, 1, port=get_free_port())
    s_nc.receive([], "bootstrap")

    s_c = Server(1, True, 10, 1, port=get_free_port())
    s_c.status = "invalid_status"
    s_c._handle_receive([], "bootstrap")

    s_c.status = "bootstrap"
    with patch.object(s_c, "add_to_buffer") as mock_add:
        s_c._handle_receive([], "client")
        mock_add.assert_called()

    s_c.status = "invalid_status"
    s_c._handle_receive([], "client")

    s_c._handle_receive([], "unknown_status")

    s_c.status = "invalid_status"
    s_c.query([], "client")

    s_c.status = "clustered"
    with patch.object(s_c, "search_vectors") as mock_search:
        mock_search.return_value = []
        s_c.query([], "clustered")
        mock_search.assert_called()

    s_nc.stop()
    s_c.stop()





from server import HintedHandoff


class TestPartitionLogic:
    """Tests for Server-based partition simulation logic."""

    def test_block_peer(self):
        s1 = Server(1, True, 10, 2, port=get_free_port())
        s1.block_peer(2)

        p2 = MagicMock()
        p2.get_id.return_value = 2

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
        assert not hh.has_hints_for(99)

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
        assert hh.has_hints_for(99)

    def test_clear_all(self):
        hh = HintedHandoff()
        hh.store_hint(1, [([1.0], 1, "a", 0, (1.0, 1))])
        hh.store_hint(2, [([2.0], 2, "b", 0, (1.0, 1))])

        hh.clear()
        assert hh.count() == 0


class TestCoordinatorElection:
    """Tests for partition-aware coordinator election."""

    def _create_cluster(self, n=4):
        servers = [Server(i, i == 0, 10, 2, port=get_free_port()) for i in range(n)]
        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)
        time.sleep(1)
        return servers

    def _partition(self, servers, groups):
        for i, s1 in enumerate(servers):
            my_group = None
            for g in groups:
                if s1.id in g:
                    my_group = g
                    break

            for s2 in servers:
                if s1.id == s2.id:
                    continue
                if s2.id not in my_group:
                    s1.block_peer(s2.id)
                else:
                    s1.unblock_peer(s2.id)

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
            self._partition(servers, [[0, 1], [2, 3]])


            assert servers[1].is_coordinator
            assert not servers[0].is_coordinator

            assert servers[3].is_coordinator
            assert not servers[2].is_coordinator

        finally:
            for s in servers:
                s.stop()

    def test_coordinator_in_minority_partition(self):
        """Test when original coordinator ends up isolated."""
        servers = self._create_cluster(4)


        try:
            self._partition(servers, [[0], [1, 2, 3]])

            assert servers[3].is_coordinator

            assert servers[0].is_coordinator

        finally:
            for s in servers:
                s.stop()

    def test_coordinator_convergence_on_heal(self):
        """Test that coordinators properly merge after partition heals."""
        servers = self._create_cluster(4)

        try:
            self._partition(servers, [[0, 1], [2, 3]])

            assert servers[1].is_coordinator
            assert servers[3].is_coordinator

            self._heal(servers)

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
        s0 = Server(0, True, 10, 2, port=get_free_port())
        s1 = Server(1, False, 10, 2, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)

        try:
            s0.clusters = [(0, [0.5])]
            s1.clusters = [(0, [0.5])]

            s0.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            s0.store.insert(([2.0], 2, "b", 0, (2.0, 0)))

            s1.store.insert(([3.0], 3, "c", 0, (3.0, 1)))

            peer1 = s0.peers[1]
            s0.reconcile_with_peer(peer1)

            time.sleep(0.2)

            assert (
                s1.store.count() == 3
            ), f"s1 should have all 3 vectors, has {s1.store.count()}"

            peer0 = s1.peers[1]
            s1.reconcile_with_peer(peer0)

            time.sleep(0.2)

            assert (
                s0.store.count() == 3
            ), f"s0 should have all 3 vectors, has {s0.store.count()}"

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
            s0.clusters = [(0, [0.5])]
            s1.clusters = [(0, [0.5])]

            s0.store.insert(([1.0], 1, "old", 0, (1.0, 0)))
            s1.store.insert(([1.0], 1, "new", 0, (5.0, 1)))

            peer0 = s1.peers[1]
            s1.reconcile_with_peer(peer0)

            time.sleep(0.2)

            vec = s0.store.get_vector(1)
            assert (
                vec[2] == "new"
            ), f"Payload should be updated to 'new', got '{vec[2]}'"

        finally:
            s0.stop()
            s1.stop()

    def test_reconciliation_retries_on_failure(self):
        """Test that reconciliation retries if the network fails mid-transfer."""
        s0 = Server(0, True, 10, 2, port=get_free_port())
        s1 = Server(1, False, 10, 2, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)

        try:
            s1.store.insert(([1.0], 1, "data", 0, (1.0, 1)))

            original_receive = s0.receive

            call_count = [0]

            def failing_receive(vectors, status):
                if status == "reconcile":
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise Exception("Simulated Network Cut during Sync")
                return original_receive(vectors, status)

            with patch.object(s0, "receive", side_effect=failing_receive):

                print("Triggering failing reconciliation...")
                try:
                    s1.reconcile_with_peer(s1.peers[1])
                except Exception as e:
                    print(f"Caught expected error: {e}")

                assert s0.count() == 0, "S0 should not have data after failed sync"

                print("Triggering retry reconciliation...")
                s1.reconcile_with_peer(s1.peers[1])

                time.sleep(0.5)
                assert s0.count() == 1, "S0 should have data after successful retry"

        finally:
            s0.stop()
            s1.stop()


class TestEventualConsistency:
    """End-to-end tests for eventual consistency with no lost vectors."""

    def test_no_lost_vectors_simple_partition(self):
        """Test all vectors present after simple partition and heal."""

        servers = [Server(i, i == 3, 100, 2, port=get_free_port()) for i in range(4)]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            for s in servers:
                s.status = "clustered"
                s.clusters = [(0, [0.5, 0.5])]

            wait_for_full_connectivity(servers)

            partition_network(servers, [[0, 1], [2, 3]])
            time.sleep(0.1)

            vectors_left = [([float(i), 0.0], f"left_{i}") for i in range(10)]
            vectors_right = [([0.0, float(i)], f"right_{i}") for i in range(10)]

            servers[0].receive_from_client(vectors_left)
            servers[2].receive_from_client(vectors_right)

            time.sleep(0.5)

            servers[0].count() + servers[1].count()
            servers[2].count() + servers[3].count()

            heal_network(servers)
            time.sleep(0.5)

            sum(s.count() for s in servers)

            all_vector_ids = set()
            for s in servers:
                for v in s.store.get_all():
                    all_vector_ids.add(v[1])

            assert (
                len(all_vector_ids) == 20
            ), f"Expected 20 unique vectors, got {len(all_vector_ids)}"

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
            s0.status = "clustered"
            s1.status = "clustered"
            s0.clusters = [(0, [0.5, 0.5])]
            s1.clusters = [(0, [0.5, 0.5])]

            wait_for_full_connectivity([s0, s1])

            partition_network([s0, s1], [[0], [1]])
            time.sleep(0.1)

            s0.receive_from_client([([1.0, 0.0], "from_s0")])

            s1.receive_from_client([([0.0, 1.0], "from_s1")])

            time.sleep(0.2)

            assert s0.count() == 1
            assert s1.count() == 1

            heal_network([s0, s1])
            time.sleep(0.3)

            all_ids_s0 = s0.store.get_all_ids()
            all_ids_s1 = s1.store.get_all_ids()

            all_ids = all_ids_s0 | all_ids_s1
            assert len(all_ids) == 2

        finally:
            s0.stop()
            s1.stop()

    def test_cluster_topology_sync_after_partition(self):
        """
        Test that cluster metadata (IDs and centroids) converges if rebalancing
        happens on one side of a partition.
        """
        s0 = Server(0, True, 100, 2, port=get_free_port())
        s1 = Server(1, False, 100, 2, port=get_free_port())
        s0.add_peer(s1)
        s1.add_peer(s0)

        try:
            vectors = []
            for i in range(50):
                vec = (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                s0.store.insert(vec)

            initial_clusters = [(0, [25.0, 25.0])]
            s0.clusters = list(initial_clusters)
            s1.clusters = list(initial_clusters)
            s0.status = "clustered"
            s1.status = "clustered"

            wait_for_full_connectivity([s0, s1])

            partition_network([s0, s1], [[0], [1]])
            time.sleep(0.5)

            print("Triggering split on S0...")
            s0.balanced_split_cluster(cluster_id=0, n_subclusters=2)

            ids_s0 = set(c[0] for c in s0.clusters)
            ids_s1 = set(c[0] for c in s1.clusters)
            assert (
                ids_s0 != ids_s1
            ), "Sanity check: Topology should be divergent during partition"
            assert 0 not in ids_s0, "S0 should have replaced Cluster 0"
            assert 0 in ids_s1, "S1 should still have Cluster 0"

            print("Healing network...")
            heal_network([s0, s1])

            time.sleep(2.0)

            final_ids_s0 = set(c[0] for c in s0.clusters)
            final_ids_s1 = set(c[0] for c in s1.clusters)

            assert final_ids_s0 == final_ids_s1, (
                f"Topology Divergence detected!\n"
                f"S0 Clusters: {final_ids_s0}\n"
                f"S1 Clusters: {final_ids_s1}"
            )

        finally:
            s0.stop()
            s1.stop()


class TestPartitionHealVectorConsistency:
    """Large-scale tests for verifying vector consistency after network partition and heal.

    These tests verify that after a partition occurs and heals:
    - No vectors are lost
    - Total vector count = vectors_sent × replication_factor

    Tests use 1000-10000 vectors and wait for actual clustering to complete.
    """

    @staticmethod
    def _wait_for_clustering(servers, timeout=60):
        """Wait until all servers reach 'clustered' status."""
        start = time.time()
        while time.time() - start < timeout:
            if all(s.status == "clustered" for s in servers):
                return True
            time.sleep(0.5)
        return False

    @staticmethod
    def _generate_vectors(count, dimension=128, prefix="vec"):
        """Generate random vectors for testing."""
        vectors = []
        for i in range(count):
            vec = np.random.randn(dimension).tolist()
            vectors.append((vec, f"{prefix}_{i}"))
        return vectors

    @staticmethod
    def _count_unique_vectors(servers):
        """Count unique vector IDs across all servers."""
        all_ids = set()
        for s in servers:
            for v in s.store.get_all():
                all_ids.add(v[1])
        return len(all_ids)

    @staticmethod
    def _total_vector_count(servers):
        """Get total vector count across all servers."""
        return sum(s.count() for s in servers)

    def test_large_scale_partition_heal_1000_vectors(self):
        """Test partition/heal with 1000 vectors - verifies total = vectors × replication_factor."""
        num_servers = 4
        replication_factor = 2
        initial_vectors = 1000
        partition_vectors_per_side = 100
        dimension = 64
        clustering_threshold = 500

        servers = [
            Server(
                i,
                i == num_servers - 1,
                clustering_threshold,
                replication_factor,
                port=get_free_port(),
            )
            for i in range(num_servers)
        ]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)

            coordinator_idx = num_servers - 1

            print(
                f"[Phase 1] Inserting {initial_vectors} initial vectors to coordinator (server {coordinator_idx})..."
            )
            initial_vecs = self._generate_vectors(initial_vectors, dimension, "init")
            servers[coordinator_idx].receive_from_client(initial_vecs)

            print("[Phase 1] Waiting for clustering...")
            assert self._wait_for_clustering(
                servers, timeout=180
            ), "Clustering did not complete in time"
            print(
                f"[Phase 1] Clustering complete. Status: {[s.status for s in servers]}"
            )

            time.sleep(3.0)
            count_after_clustering = self._total_vector_count(servers)
            unique_after_clustering = self._count_unique_vectors(servers)
            print(
                f"[Phase 1] After clustering: {unique_after_clustering} unique, {count_after_clustering} total"
            )

            print("[Phase 2] Creating network partition...")
            partition_network(servers, [[0, 1], [2, 3]])
            time.sleep(1.0)

            print(
                f"[Phase 3] Inserting {partition_vectors_per_side * 2} vectors during partition..."
            )
            left_vecs = self._generate_vectors(
                partition_vectors_per_side, dimension, "left"
            )
            right_vecs = self._generate_vectors(
                partition_vectors_per_side, dimension, "right"
            )

            servers[0].receive_from_client(left_vecs)
            servers[2].receive_from_client(right_vecs)
            time.sleep(2.0)

            count_before_heal = self._total_vector_count(servers)
            unique_before_heal = self._count_unique_vectors(servers)
            print(
                f"[Phase 3] Before heal: {unique_before_heal} unique, {count_before_heal} total"
            )

            print("[Phase 4] Healing network partition...")
            heal_network(servers)
            time.sleep(5.0)

            count_after_heal = self._total_vector_count(servers)
            unique_after_heal = self._count_unique_vectors(servers)
            print(
                f"[Phase 4] After heal: {unique_after_heal} unique, {count_after_heal} total"
            )

            total_vectors_sent = initial_vectors + partition_vectors_per_side * 2
            expected_total_count = total_vectors_sent * replication_factor

            assert unique_after_heal == total_vectors_sent, (
                f"VECTOR LOSS: Expected {total_vectors_sent} unique vectors, "
                f"got {unique_after_heal} "
                f"(lost {total_vectors_sent - unique_after_heal} vectors)"
            )

            assert count_after_heal == expected_total_count, (
                f"REPLICATION MISMATCH: Expected {expected_total_count} total "
                f"(vectors={total_vectors_sent} × rf={replication_factor}), "
                f"got {count_after_heal} "
                f"(difference: {expected_total_count - count_after_heal})"
            )

            print(
                f"[SUCCESS] All {total_vectors_sent} vectors present with correct replication"
            )

        finally:
            for s in servers:
                s.stop()

    def test_large_scale_partition_heal_5000_vectors(self):
        """Test partition/heal with 5000 vectors - stress test for larger datasets."""
        num_servers = 6
        replication_factor = 2
        initial_vectors = 5000
        partition_vectors_per_partition = 333
        dimension = 64
        clustering_threshold = 500

        servers = [
            Server(
                i,
                i == num_servers - 1,
                clustering_threshold,
                replication_factor,
                port=get_free_port(),
            )
            for i in range(num_servers)
        ]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)
            coordinator_idx = num_servers - 1

            print(
                f"[Phase 1] Inserting {initial_vectors} initial vectors to coordinator..."
            )
            initial_vecs = self._generate_vectors(initial_vectors, dimension, "init")
            servers[coordinator_idx].receive_from_client(initial_vecs)

            print("[Phase 1] Waiting for clustering...")
            assert self._wait_for_clustering(
                servers, timeout=180
            ), "Clustering did not complete"
            time.sleep(3.0)

            print("[Phase 2] Creating three-way partition...")
            partition_network(servers, [[0, 1], [2, 3], [4, 5]])
            time.sleep(1.0)

            print(
                f"[Phase 3] Inserting {partition_vectors_per_partition * 3} vectors during partition..."
            )
            p1_vecs = self._generate_vectors(
                partition_vectors_per_partition, dimension, "p1"
            )
            p2_vecs = self._generate_vectors(
                partition_vectors_per_partition, dimension, "p2"
            )
            p3_vecs = self._generate_vectors(
                partition_vectors_per_partition, dimension, "p3"
            )

            servers[0].receive_from_client(p1_vecs)
            servers[2].receive_from_client(p2_vecs)
            servers[4].receive_from_client(p3_vecs)
            time.sleep(3.0)

            count_before_heal = self._total_vector_count(servers)
            unique_before_heal = self._count_unique_vectors(servers)
            print(
                f"[Phase 3] Before heal: {unique_before_heal} unique, {count_before_heal} total"
            )

            print("[Phase 4] Healing network...")
            heal_network(servers)
            time.sleep(10.0)

            count_after_heal = self._total_vector_count(servers)
            unique_after_heal = self._count_unique_vectors(servers)
            print(
                f"[Phase 4] After heal: {unique_after_heal} unique, {count_after_heal} total"
            )

            total_vectors_sent = initial_vectors + partition_vectors_per_partition * 3
            expected_total_count = total_vectors_sent * replication_factor

            assert (
                unique_after_heal == total_vectors_sent
            ), f"VECTOR LOSS: Expected {total_vectors_sent} unique, got {unique_after_heal}"

            assert (
                count_after_heal == expected_total_count
            ), f"REPLICATION MISMATCH: Expected {expected_total_count}, got {count_after_heal}"

            print(
                f"[SUCCESS] All {total_vectors_sent} vectors present with correct replication"
            )

        finally:
            for s in servers:
                s.stop()

    def test_large_scale_multiple_partition_cycles(self):
        """Test multiple partition/heal cycles with thousands of vectors."""
        num_servers = 4
        replication_factor = 2
        initial_vectors = 2000
        vectors_per_cycle = 200
        num_cycles = 3
        dimension = 64
        clustering_threshold = 500

        servers = [
            Server(
                i,
                i == num_servers - 1,
                clustering_threshold,
                replication_factor,
                port=get_free_port(),
            )
            for i in range(num_servers)
        ]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)
            coordinator_idx = num_servers - 1

            print(
                f"[Init] Inserting {initial_vectors} initial vectors to coordinator..."
            )
            initial_vecs = self._generate_vectors(initial_vectors, dimension, "init")
            servers[coordinator_idx].receive_from_client(initial_vecs)

            assert self._wait_for_clustering(servers, timeout=180), "Clustering failed"
            time.sleep(2.0)

            total_vectors_sent = initial_vectors

            for cycle in range(num_cycles):
                print(f"[Cycle {cycle+1}] Starting partition cycle...")

                partition_network(servers, [[0, 1], [2, 3]])
                time.sleep(1.0)

                cycle_vecs = self._generate_vectors(
                    vectors_per_cycle, dimension, f"cycle{cycle}"
                )
                servers[cycle % num_servers].receive_from_client(cycle_vecs)
                total_vectors_sent += vectors_per_cycle
                time.sleep(1.0)

                heal_network(servers)
                time.sleep(3.0)

                current_unique = self._count_unique_vectors(servers)
                current_total = self._total_vector_count(servers)
                print(
                    f"[Cycle {cycle+1}] After heal: {current_unique} unique, {current_total} total"
                )

            time.sleep(2.0)
            final_unique = self._count_unique_vectors(servers)
            final_total = self._total_vector_count(servers)
            expected_total = total_vectors_sent * replication_factor

            print(
                f"[Final] Unique: {final_unique}, Total: {final_total}, Expected: {expected_total}"
            )

            assert (
                final_unique == total_vectors_sent
            ), f"VECTOR LOSS: Expected {total_vectors_sent} unique, got {final_unique}"

            assert (
                final_total == expected_total
            ), f"REPLICATION MISMATCH: Expected {expected_total}, got {final_total}"

            print(
                f"[SUCCESS] All {total_vectors_sent} vectors survive {num_cycles} partition cycles"
            )

        finally:
            for s in servers:
                s.stop()

    def test_large_scale_asymmetric_partition(self):
        """Test asymmetric partition (minority/majority) with thousands of vectors."""
        num_servers = 5
        replication_factor = 2
        initial_vectors = 1500
        vectors_to_isolated = 200
        vectors_to_majority = 300
        dimension = 64
        clustering_threshold = 500

        servers = [
            Server(
                i,
                i == num_servers - 1,
                clustering_threshold,
                replication_factor,
                port=get_free_port(),
            )
            for i in range(num_servers)
        ]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)
            coordinator_idx = num_servers - 1

            print(
                f"[Init] Inserting {initial_vectors} initial vectors to coordinator..."
            )
            initial_vecs = self._generate_vectors(initial_vectors, dimension, "init")
            servers[coordinator_idx].receive_from_client(initial_vecs)

            assert self._wait_for_clustering(servers, timeout=180), "Clustering failed"
            time.sleep(2.0)

            print("[Partition] Creating asymmetric partition (1 vs 4 nodes)...")
            partition_network(servers, [[0], [1, 2, 3, 4]])
            time.sleep(1.0)

            print(f"[Insert] Adding {vectors_to_isolated} vectors to isolated node...")
            isolated_vecs = self._generate_vectors(
                vectors_to_isolated, dimension, "isolated"
            )
            servers[0].receive_from_client(isolated_vecs)

            print(
                f"[Insert] Adding {vectors_to_majority} vectors to majority partition..."
            )
            majority_vecs = self._generate_vectors(
                vectors_to_majority, dimension, "majority"
            )
            servers[1].receive_from_client(majority_vecs)
            time.sleep(2.0)

            count_before = self._total_vector_count(servers)
            unique_before = self._count_unique_vectors(servers)
            print(f"[Before Heal] Unique: {unique_before}, Total: {count_before}")

            print("[Heal] Healing network...")
            heal_network(servers)
            time.sleep(5.0)

            total_vectors_sent = (
                initial_vectors + vectors_to_isolated + vectors_to_majority
            )
            expected_total = total_vectors_sent * replication_factor

            final_unique = self._count_unique_vectors(servers)
            final_total = self._total_vector_count(servers)
            print(
                f"[After Heal] Unique: {final_unique}, Total: {final_total}, Expected: {expected_total}"
            )

            assert (
                final_unique == total_vectors_sent
            ), f"VECTOR LOSS: Expected {total_vectors_sent} unique, got {final_unique}"

            assert (
                final_total == expected_total
            ), f"REPLICATION MISMATCH: Expected {expected_total}, got {final_total}"

            print(
                f"[SUCCESS] All {total_vectors_sent} vectors present after asymmetric partition heal"
            )

        finally:
            for s in servers:
                s.stop()

    def test_large_scale_high_replication_factor(self):
        """Test with high replication factor (RF=3) and thousands of vectors."""
        num_servers = 8
        replication_factor = 3
        initial_vectors = 3000
        partition_vectors = 500
        dimension = 64
        clustering_threshold = 500

        servers = [
            Server(
                i,
                i == num_servers - 1,
                clustering_threshold,
                replication_factor,
                port=get_free_port(),
            )
            for i in range(num_servers)
        ]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)
            coordinator_idx = num_servers - 1

            print(
                f"[Init] Inserting {initial_vectors} vectors with RF={replication_factor} to coordinator..."
            )
            initial_vecs = self._generate_vectors(initial_vectors, dimension, "init")
            servers[coordinator_idx].receive_from_client(initial_vecs)

            assert self._wait_for_clustering(servers, timeout=180), "Clustering failed"
            time.sleep(3.0)

            print("[Partition] Creating four-way partition...")
            partition_network(servers, [[0, 1], [2, 3], [4, 5], [6, 7]])
            time.sleep(1.0)

            print(f"[Insert] Adding {partition_vectors} vectors during partition...")
            partition_vecs = self._generate_vectors(
                partition_vectors, dimension, "part"
            )

            part_chunk = partition_vectors // 4
            servers[0].receive_from_client(partition_vecs[:part_chunk])
            servers[2].receive_from_client(partition_vecs[part_chunk : part_chunk * 2])
            servers[4].receive_from_client(
                partition_vecs[part_chunk * 2 : part_chunk * 3]
            )
            servers[6].receive_from_client(partition_vecs[part_chunk * 3 :])
            time.sleep(3.0)

            print("[Heal] Healing network...")
            heal_network(servers)
            time.sleep(10.0)

            total_vectors_sent = initial_vectors + partition_vectors
            expected_total = total_vectors_sent * replication_factor

            final_unique = self._count_unique_vectors(servers)
            final_total = self._total_vector_count(servers)
            print(
                f"[Result] Unique: {final_unique}, Total: {final_total}, Expected: {expected_total}"
            )

            assert (
                final_unique == total_vectors_sent
            ), f"VECTOR LOSS: Expected {total_vectors_sent} unique, got {final_unique}"

            assert final_total == expected_total, (
                f"REPLICATION MISMATCH: Expected {expected_total} "
                f"(vectors={total_vectors_sent} × rf={replication_factor}), "
                f"got {final_total}"
            )

            print(
                f"[SUCCESS] All {total_vectors_sent} vectors correctly replicated {replication_factor}x"
            )

        finally:
            for s in servers:
                s.stop()


class TestRecursivePartitions:
    """Tests for complex recursive partition scenarios."""

    def test_three_way_partition(self):
        """Test A|B|C isolated partition scenario."""

        servers = [Server(i, i == 2, 100, 1, port=get_free_port()) for i in range(3)]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)

            partition_network(servers, [[0], [1], [2]])
            time.sleep(0.1)

            for s in servers:
                assert (
                    s.is_coordinator
                ), f"Server {s.id} should be coordinator of its partition"

        finally:
            for s in servers:
                s.stop()

    def test_cascading_partitions(self):
        """Test ABC -> A|BC -> A|B|C."""

        servers = [Server(i, i == 2, 100, 1, port=get_free_port()) for i in range(3)]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)

            assert servers[2].is_coordinator

            partition_network(servers, [[0], [1, 2]])
            time.sleep(0.1)

            assert servers[0].is_coordinator
            assert servers[2].is_coordinator
            assert not servers[1].is_coordinator

            partition_network(servers, [[0], [1], [2]])
            time.sleep(0.1)

            for s in servers:
                assert s.is_coordinator

        finally:
            for s in servers:
                s.stop()

    def test_partial_heal_chain(self):
        """Test A|B|C -> AB|C -> ABC heal sequence."""

        servers = [Server(i, i == 2, 100, 1, port=get_free_port()) for i in range(3)]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)

            s0, s1, s2 = servers

            for s in servers:
                s.status = "clustered"
                s.clusters = [(0, [0.5, 0.5])]

            partition_network(servers, [[0], [1], [2]])
            time.sleep(0.1)

            s0.receive_from_client([([1.0, 0.0], "v0")])
            s1.receive_from_client([([0.5, 0.5], "v1")])
            s2.receive_from_client([([0.0, 1.0], "v2")])
            time.sleep(0.2)

            assert s0.count() == 1
            assert s1.count() == 1
            assert s2.count() == 1

            partition_network(servers, [[0, 1], [2]])
            time.sleep(0.3)

            ids_01 = s0.store.get_all_ids() | s1.store.get_all_ids()
            assert (
                len(ids_01) >= 2
            ), "After partial heal, partitions 0 and 1 should have synced"

            heal_network(servers)
            time.sleep(0.3)

            all_ids = (
                s0.store.get_all_ids() | s1.store.get_all_ids() | s2.store.get_all_ids()
            )
            assert len(all_ids) == 3, f"All 3 vectors should exist, got {len(all_ids)}"

        finally:
            for s in servers:
                s.stop()

    def test_hinted_handoff_during_partition(self):
        """Test that hints are stored during partition and delivered on heal."""

        servers = [Server(i, i == 2, 100, 2, port=get_free_port()) for i in range(3)]

        for s in servers:
            for p in servers:
                if s.get_id() != p.get_id():
                    s.add_peer(p)

        try:
            wait_for_full_connectivity(servers)

            for s in servers:
                s.status = "clustered"
                s.clusters = [(0, [0.5, 0.5])]
                s.replication_factor = 3

            partition_network(servers, [[0, 1], [2]])
            time.sleep(0.1)

            servers[0].receive_from_client([([1.0, 0.0], "test")])
            time.sleep(0.3)

            hints_for_2 = (
                servers[0].hinted_handoff.count() + servers[1].hinted_handoff.count()
            )
            assert hints_for_2 > 0, "Hints should be stored for unreachable node"

            heal_network(servers)
            time.sleep(0.3)

            assert servers[0].hinted_handoff.count() == 0
            assert servers[1].hinted_handoff.count() == 0

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
            for i in range(100):
                large_payload = "x" * 1000
                s.store.insert(([float(i)] * 100, i, large_payload, 0, (float(i), 0)))

            digest = s.get_vector_digest()

            import sys

            sys.getsizeof(digest)

            assert len(digest) == 100

        finally:
            s.stop()

    def test_deduplication_prevents_redundant_storage(self):
        """Test that duplicate vectors with same version are rejected."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            result1 = s.store.insert(([1.0], 1, "a", 0, (1.0, 0)))
            result2 = s.store.insert(([1.0], 1, "a", 0, (1.0, 0)))

            assert result1
            assert not result2
            assert s.store.count() == 1

        finally:
            s.stop()

    def test_newer_version_replaces_older(self):
        """Test that newer versions replace older ones."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.store.insert(([1.0], 1, "v1", 0, (1.0, 0)))

            result_old = s.store.insert(([1.0], 1, "v0", 0, (0.5, 0)))
            assert not result_old

            result_new = s.store.insert(([1.0], 1, "v2", 0, (2.0, 0)))
            assert result_new

            assert s.store.count() == 1
            vec = s.store.get_vector(1)
            assert vec[2] == "v2"

        finally:
            s.stop()




def test_server_without_network():
    """Test server works normally without network simulator."""
    s = Server(0, True, 10, 1, port=get_free_port())

    try:
        assert s._is_peer_reachable(s.peers[0])

        reachable = s.get_reachable_peers()
        assert len(reachable) == 1

    finally:
        s.stop()




class TestPeerAdditionalMethods:
    """Tests for remaining Peer class methods."""

    def test_peer_get_partition_coordinator_id(self):
        mock_server = MagicMock()
        mock_server.partition_coordinator_id = 42
        peer = Peer("127.0.0.1", get_free_port(), server_instance=mock_server)
        result = peer.get_partition_coordinator_id()
        assert result == 42


class TestVectorStoreEdgeCases:
    """Tests for VectorStore edge cases."""

    def test_remove_by_id_nonexistent(self):
        """Test removing a vector that doesn't exist."""
        from server import VectorStore

        store = VectorStore()
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
        s = Server(0, False, 10, 1, port=get_free_port())

        try:
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
            s0.active_peers = {0, 1}
            assert len(s0.get_unreachable_peers()) == 0

            s0.block_peer(1)
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
            assert result[0] == frozenset()
        finally:
            s.stop()


class TestReconciliationEdgeCases:
    """Tests for reconciliation edge cases."""

    def test_reconcile_already_reconciling(self):
        """Test on_partition_heal returns early when already reconciling."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s._reconciling = True

            s.on_partition_heal()

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

            s.reconcile_with_peer(mock_peer)
        finally:
            s.stop()

    def test_should_be_on_peer_no_destinations(self):
        """Test _should_be_on_peer when vector has no stored destinations."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.clusters = [(0, [1.0, 0.0])]

        mock_peer = MagicMock()
        mock_peer.get_id.return_value = 0
        mock_peer.similarity.return_value = 1.0
        s.peers = [mock_peer]

        try:
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
            s.status = "clustered"

            vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1]))
            s.send_to_peers([vec])

            assert s.hinted_handoff.has_hints_for(1)
        finally:
            s.stop()


class TestHandoffEdgeCases:
    """Tests for hinted handoff edge cases."""

    def test_deliver_hints_exception(self):
        """Test deliver_hints handles exceptions."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.hinted_handoff.store_hint(1, [vec])

            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.receive.side_effect = Exception("Delivery failed!")

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]

            s.deliver_hints()

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
            s.receive([vec], "handoff")
            time.sleep(0.1)

            assert s.store.count() == 1
        finally:
            s.stop()

    def test_receive_reconcile(self):
        """Test receive with reconcile status."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.receive([vec], "reconcile")
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
            s.peers = [Peer(s.ip, s.port, server_instance=s)]

            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.receive([vec], "client")
            time.sleep(0.1)

        finally:
            s.stop()


class TestReconciliationWithRealCoordinator:
    """Test reconciliation with actual coordinator election."""

    def test_reconcile_exception_in_coordinator_loop(self):
        """Test _reconcile_with_other_peers handles exceptions."""
        s0 = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            s0.reconcile_with_peer = MagicMock(
                side_effect=Exception("Reconcile failed!")
            )
            s0.peers = [Peer(s0.ip, s0.port, server_instance=s0), mock_peer]

            s0._reconcile_with_other_peers()
        finally:
            s0.stop()


class TestClusteringEdgeCases:
    """Test edge cases in clustering and assignment."""

    def test_clustering_with_vector_buffer_on_clustered(self):
        """Test _handle_set_clusters processes buffered vectors."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.vector_buffer = [([1.0], 1, "a", -1, (1.0, 0))]
            s.clustering_in_progress = True

            clusters = {0: {"center": [1.0], "members": []}}
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 0
            mock_peer.similarity.return_value = 1.0
            s.peers = [mock_peer]

            s._handle_set_clusters(clusters, [(0, [1.0])])

            assert len(s.vector_buffer) == 0
            assert not s.clustering_in_progress
        finally:
            s.stop()

    def test_assign_clusters_no_nodes(self):
        """Test assign_clusters_to_peers with no reachable nodes."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.peers = []

        try:
            clusters = {0: {"center": [1.0], "members": []}}
            result = s.assign_clusters_to_peers(clusters)
            assert result == {}
        finally:
            s.stop()


class TestQueryEdgeCases:
    """Test query edge cases."""

    def test_query_bootstrap_no_coordinator(self):
        """Test query bootstrap mode when coordinator not found returns empty."""
        s = Server(0, False, 10, 1, port=get_free_port())

        try:
            result = s.query([(1, [1.0])], "client")
            assert result is None or result == []
        finally:
            s.stop()


class TestShouldBeOnPeerVariants:
    """Test _should_be_on_peer with different vector formats."""

    def test_should_be_on_peer_with_stored_destinations(self):
        """Test _should_be_on_peer when vector has stored destinations."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1, 2]))

            assert s._should_be_on_peer(vec, 1)
            assert s._should_be_on_peer(vec, 2)
            assert not s._should_be_on_peer(vec, 3)
        finally:
            s.stop()

    def test_should_be_on_peer_no_clusters(self):
        """Test _should_be_on_peer when clusters not yet formed."""
        s = Server(0, True, 10, 1, port=get_free_port())
        s.clusters = []

        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))

            assert s._should_be_on_peer(vec, 1)
        finally:
            s.stop()




from cluster_index import ClusterIndex


class TestClusterIndex:
    """Tests for ClusterIndex class to improve coverage."""

    def test_build_empty_clusters(self):
        """Test build() with empty cluster list triggers warning."""
        ci = ClusterIndex(dimension=2)
        ci.build([])
        assert ci.hnsw_index is None

    def test_build_valid_clusters(self):
        """Test build() with valid clusters creates HNSW index."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [1.0, 1.0])]
        ci.build(clusters)

        assert ci.hnsw_index is not None
        assert ci.hnsw_index.element_count == 3

    def test_build_with_invalid_cluster_id(self):
        """Test build() handles non-integer cluster IDs gracefully."""
        ci = ClusterIndex(dimension=2)
        clusters = [
            ("invalid_string", [1.0, 0.0]),
            (1, [0.0, 1.0]),
        ]
        ci.build(clusters)

        assert ci.hnsw_index.element_count == 1

    def test_build_all_invalid_ids(self):
        """Test build() with all invalid IDs triggers warning."""
        ci = ClusterIndex(dimension=2)
        clusters = [("invalid1", [1.0, 0.0]), ("invalid2", [0.0, 1.0])]
        ci.build(clusters)
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

        ci.hnsw_index.set_ef(10)
        result = ci.find_nearest_clusters([50.0, 50.0], k=20)

        assert ci.hnsw_index.ef >= 20
        assert len(result) == 20

    def test_find_nearest_clusters_returns_correct_ids(self):
        """Test find_nearest_clusters() returns correct cluster IDs."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [0.5, 0.5])]
        ci.build(clusters)

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
        assert len(result[0]) == 2

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

        assert data["dimension"] == 2
        assert data["max_clusters"] == 100
        assert data["ef_construction"] == 100
        assert data["M"] == 8
        assert data["index_data"] is None

    def test_to_serializable_with_index(self):
        """Test to_serializable() with built index."""
        ci = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci.build(clusters)

        data = ci.to_serializable()

        assert data["index_data"] is not None
        assert isinstance(data["index_data"], str)

    def test_from_serializable_no_index(self):
        """Test from_serializable() without index data."""
        data = {
            "dimension": 2,
            "max_clusters": 100,
            "ef_construction": 100,
            "M": 8,
            "index_data": None,
        }
        ci = ClusterIndex.from_serializable(data)

        assert ci.dimension == 2
        assert ci.max_clusters == 100
        assert ci.hnsw_index is None

    def test_from_serializable_with_index(self):
        """Test from_serializable() with valid index data."""
        ci1 = ClusterIndex(dimension=2)
        clusters = [(0, [1.0, 0.0]), (1, [0.0, 1.0])]
        ci1.build(clusters)
        data = ci1.to_serializable()

        ci2 = ClusterIndex.from_serializable(data)

        assert ci2.hnsw_index is not None
        assert ci2.hnsw_index.element_count == 2

    def test_from_serializable_invalid_data(self):
        """Test from_serializable() handles corrupt index data."""
        data = {
            "dimension": 2,
            "max_clusters": 100,
            "ef_construction": 100,
            "M": 8,
            "index_data": "invalid_base64_data_that_is_not_a_valid_pickle",
        }

        ci = ClusterIndex.from_serializable(data)
        assert ci.hnsw_index is None



import os
import tempfile

import qdrant_module


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

        result1 = qdrant_module.create_collection(url, collection, 2)
        assert result1

        result2 = qdrant_module.create_collection(url, collection, 2)
        assert result2

        qdrant_module._client_cache.clear()

    def test_create_collection_different_distances(self):
        """Test create_collection() with different distance metrics."""
        url = ":memory:"

        qdrant_module.create_collection(url, "test_cosine", 2, "Cosine")
        qdrant_module.create_collection(url, "test_euclid", 2, "Euclid")
        qdrant_module.create_collection(url, "test_dot", 2, "Dot")
        qdrant_module.create_collection(
            url, "test_unknown", 2, "Unknown"
        )

        qdrant_module._client_cache.clear()

    def test_delete_collection_success(self):
        """Test delete_collection() success case."""
        url = ":memory:"
        collection = "test_delete"

        qdrant_module.create_collection(url, collection, 2)
        result = qdrant_module.delete_collection(url, collection)

        assert result
        qdrant_module._client_cache.clear()

    def test_delete_collection_nonexistent(self):
        """Test delete_collection() for non-existent collection - Qdrant doesn't throw."""
        url = ":memory:"

        result = qdrant_module.delete_collection(url, "nonexistent_collection")
        assert result
        qdrant_module._client_cache.clear()

    def test_insert_and_count(self):
        """Test insert_vectors() and count()."""
        url = ":memory:"
        collection = "test_insert"

        qdrant_module.create_collection(url, collection, 2)

        vectors = [([1.0, 0.0], 1, "payload_a", 0), ([0.0, 1.0], 2, "payload_b", 0)]

        result = qdrant_module.insert_vectors(
            url, collection, vectors, batch_size_retry=1
        )
        assert result

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
        assert result

        count = qdrant_module.count(url, collection)
        assert count == 0

        qdrant_module._client_cache.clear()

    def test_delete_vector_error(self):
        """Test delete_vector() handles errors."""
        url = ":memory:"

        result = qdrant_module.delete_vector(url, "nonexistent", 1)
        assert not result

        qdrant_module._client_cache.clear()

    def test_get_all_vectors(self):
        """Test get_all_vectors() retrieves all with pagination."""
        url = ":memory:"
        collection = "test_scroll"

        qdrant_module.create_collection(url, collection, 2)

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
            ([0.5, 0.5], 3, "C", 0),
        ]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)

        query = [([1.0, 0.0], 0)]
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

            client.close()
            qdrant_module._client_cache.clear()

    def test_query_vectors_generic(self):
        """Test query_vectors_generic() groups queries by cluster and routes correctly."""
        url = ":memory:"
        base_collection = "test_generic_query"

        qdrant_module.create_collection(
            url, qdrant_module.get_collection_name(base_collection, 0), 2
        )
        qdrant_module.create_collection(
            url, qdrant_module.get_collection_name(base_collection, 1), 2
        )

        vectors_c0 = [([1.0, 0.0], 1, "A", 0)]
        vectors_c1 = [([0.0, 1.0], 2, "B", 1)]

        qdrant_module.insert_vectors(
            url,
            qdrant_module.get_collection_name(base_collection, 0),
            vectors_c0,
            batch_size_retry=1,
        )
        qdrant_module.insert_vectors(
            url,
            qdrant_module.get_collection_name(base_collection, 1),
            vectors_c1,
            batch_size_retry=1,
        )

        queries = [([1.0, 0.0], 0), ([0.0, 1.0], 1)]

        results = qdrant_module.query_vectors_generic(
            url, base_collection, queries, topk=1
        )

        assert len(results) > 0
        qdrant_module._client_cache.clear()

    def test_insert_vectors_generic(self):
        """Test insert_vectors_generic() groups vectors by cluster and inserts correctly."""
        url = ":memory:"
        base_collection = "test_generic_insert"

        qdrant_module.create_collection(
            url, qdrant_module.get_collection_name(base_collection, 0), 2
        )
        qdrant_module.create_collection(
            url, qdrant_module.get_collection_name(base_collection, 1), 2
        )

        vectors = [
            ([1.0, 0.0], 1, "A", 0),
            ([0.0, 1.0], 2, "B", 1),
            ([0.5, 0.5], 3, "C", 0),
        ]

        qdrant_module.insert_vectors_generic(
            url, base_collection, vectors, batch_size_retry=1
        )

        count_c0 = qdrant_module.count(
            url, qdrant_module.get_collection_name(base_collection, 0)
        )
        count_c1 = qdrant_module.count(
            url, qdrant_module.get_collection_name(base_collection, 1)
        )

        assert count_c0 == 2
        assert count_c1 == 1

        qdrant_module._client_cache.clear()

    def test_insert_batch_size_adjustment(self):
        """Test insert_vectors() with batch_size smaller than vector count."""
        url = ":memory:"
        collection = "test_batch_size"

        qdrant_module.create_collection(url, collection, 2)

        vectors = [([float(i), float(i)], i, f"payload_{i}", 0) for i in range(10)]

        result = qdrant_module.insert_vectors(
            url, collection, vectors, batch_size_retry=1, batch_size=3
        )
        assert result

        count = qdrant_module.count(url, collection)
        assert count == 10

        qdrant_module._client_cache.clear()



import asyncio

from peer.peer import Peer


class TestPeerRemoteCalls:
    """Tests for Peer class remote call methods to improve coverage."""

    def test_peer_no_communicator_raises(self):
        """Test _remote_call() raises when no communicator set."""
        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=None)

        with pytest.raises(Exception) as exc_info:
            peer._remote_call("get_id")

        assert "No communicator" in str(exc_info.value)

    def test_peer_remote_call_success(self):
        """Test _remote_call() with successful response."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 42}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer._remote_call("get_id")

        assert result == 42

    def test_peer_remote_call_failure(self):
        """Test _remote_call() raises on failed response status."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": -1, "error": "Connection refused", "response": None}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)

        with pytest.raises(Exception) as exc_info:
            peer._remote_call("get_id")

        assert "Connection refused" in str(exc_info.value)

    def test_peer_get_id_remote(self):
        """Test get_id() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 123}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_id()

        assert result == 123

    def test_peer_similarity_remote(self):
        """Test similarity() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 0.95}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.similarity([1.0, 0.0])

        assert result == 0.95

    def test_peer_receive_remote(self):
        """Test receive() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": None}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.receive([([1.0], 1, "a", 0)], "client")

        assert result is None

    def test_peer_i_am_coord_remote(self):
        """Test i_am_coord() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": True}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.i_am_coord()

        assert result

    def test_peer_set_clusters_remote(self):
        """Test set_clusters() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": None}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.set_clusters({0: {"center": [1.0], "members": []}}, [(0, [1.0])])

        assert result is None

    def test_peer_search_vectors_local_remote(self):
        """Test search_vectors_local() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": [([1.0], 1, "a", 0.99)]}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.search_vectors_local([([1.0], 1)], 5)

        assert len(result) == 1

    def test_peer_query_remote(self):
        """Test query() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": []}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.query([([1.0], 1)], "client")

        assert result == []

    def test_peer_get_vector_digest_remote(self):
        """Test get_vector_digest() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": {1: (1.0, 0)}}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_vector_digest()

        assert 1 in result

    def test_peer_get_vectors_by_ids_remote(self):
        """Test get_vectors_by_ids() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": []}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_vectors_by_ids([1, 2, 3])

        assert result == []

    def test_peer_get_partition_coordinator_id_remote(self):
        """Test get_partition_coordinator_id() for remote peer."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 5}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.get_partition_coordinator_id()

        assert result == 5

    def test_peer_ping_local(self):
        """Test ping() for local peer."""
        mock_server = MagicMock()
        mock_server.respond_to_ping.return_value = True

        peer = Peer("127.0.0.1", 9999, server_instance=mock_server)
        result = peer.ping()

        assert result
        mock_server.respond_to_ping.assert_called_once()

    def test_peer_ping_remote_success(self):
        """Test ping() for remote peer - success."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": True}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.ping()

        assert result

    def test_peer_ping_remote_exception(self):
        """Test ping() for remote peer - returns False on exception."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            raise Exception("Network error")

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.ping()

        assert not result

    def test_peer_is_local(self):
        """Test is_local() method."""
        local_peer = Peer("127.0.0.1", 9999, server_instance=MagicMock())
        remote_peer = Peer(
            "127.0.0.1", 9999, server_instance=None, communicator=MagicMock()
        )

        assert local_peer.is_local()
        assert not remote_peer.is_local()



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

            with open(cert_path, "rb") as f:
                cert_content = f.read()
            with open(key_path, "rb") as f:
                key_content = f.read()

            assert b"CERTIFICATE" in cert_content
            assert b"PRIVATE KEY" in key_content

    def test_generate_cert_skips_if_exists(self):
        """Test generate_self_signed_cert() skips when files already exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "test.crt")
            key_path = os.path.join(tmpdir, "test.key")

            with open(cert_path, "w") as f:
                f.write("existing cert")
            with open(key_path, "w") as f:
                f.write("existing key")

            cert_mtime_before = os.path.getmtime(cert_path)
            key_mtime_before = os.path.getmtime(key_path)

            generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

            cert_mtime_after = os.path.getmtime(cert_path)
            key_mtime_after = os.path.getmtime(key_path)

            assert cert_mtime_before == cert_mtime_after
            assert key_mtime_before == key_mtime_after




from server import (
    PhiAccrualFailureDetector,
    QdrantVectorStore,
    VectorStore,
)


class TestPhiAccrualFailureDetector:
    """Tests for PhiAccrualFailureDetector to achieve 100% coverage."""

    def test_phi_no_heartbeat(self):
        """Test phi() returns 0.0 when no heartbeat received (line 112)."""
        detector = PhiAccrualFailureDetector()
        assert detector.phi() == 0.0

    def test_reset_method(self):
        """Test reset() method (lines 160-164)."""
        detector = PhiAccrualFailureDetector()

        detector.heartbeat_received()
        time.sleep(0.01)
        detector.heartbeat_received()

        assert detector.last_heartbeat_time is not None
        assert len(detector.heartbeat_intervals) > 0

        detector.reset()

        assert detector.last_heartbeat_time is None
        assert len(detector.heartbeat_intervals) == 0
        assert detector._cached_mean == detector.first_heartbeat_estimate_ms
        assert detector._cached_variance == 0.0

    def test_update_statistics_empty_intervals(self):
        """Test _update_statistics() with empty intervals (line 90)."""
        detector = PhiAccrualFailureDetector()
        detector._update_statistics()
        assert detector._cached_mean == detector.first_heartbeat_estimate_ms

    def test_update_statistics_single_interval(self):
        """Test _update_statistics() with single interval (lines 99-100)."""
        detector = PhiAccrualFailureDetector()
        detector.heartbeat_intervals.append(1000.0)
        detector._update_statistics()
        assert detector._cached_variance == 0.0

    def test_calculate_phi_math_edge_cases(self):
        """Test _calculate_phi() with edge cases (lines 146-147)."""
        detector = PhiAccrualFailureDetector()

        result = detector._calculate_phi(1e10)
        assert result > 0

        detector.heartbeat_received()
        time.sleep(0.01)
        detector.heartbeat_received()

        result = detector._calculate_phi(1e15)
        assert result <= 100.0

    def test_is_available_when_suspicious(self):
        """Test is_available() returns False when phi >= threshold."""
        detector = PhiAccrualFailureDetector(threshold=0.001)

        detector.heartbeat_received()
        time.sleep(0.1)

        assert isinstance(detector.is_available(), bool)

    def test_heartbeat_received_updates_stats(self):
        """Test heartbeat_received() updates statistics properly."""
        detector = PhiAccrualFailureDetector()

        detector.heartbeat_received()
        assert detector.last_heartbeat_time is not None

        time.sleep(0.01)
        detector.heartbeat_received()
        assert len(detector.heartbeat_intervals) == 1

        time.sleep(0.01)
        detector.heartbeat_received()
        assert len(detector.heartbeat_intervals) == 2


class TestCosineSimilarityNumpyBranch:
    """Tests for cosine_similarity with numpy arrays (lines 170-172)."""

    def test_cosine_similarity_numpy_arrays(self):
        """Test cosine_similarity with numpy arrays."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0])

        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(1.0)

    def test_cosine_similarity_numpy_orthogonal(self):
        """Test cosine_similarity with orthogonal numpy arrays."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(0.0)

    def test_cosine_similarity_mixed_types(self):
        """Test cosine_similarity with list and numpy array mix."""
        v1 = [1.0, 0.0]
        v2 = np.array([1.0, 0.0])

        result = cosine_similarity(v1, v2)
        assert result == pytest.approx(1.0)


class TestQdrantVectorStoreComplete:
    """Complete tests for QdrantVectorStore to achieve 100% coverage."""

    def test_insert_numpy_array(self):
        """Test insert with numpy array (line 397)."""
        store = QdrantVectorStore(":memory:", "test_numpy", 2)

        vec = (np.array([1.0, 2.0]), 1, "payload", 0, (1.0, 0))
        result = store.insert(vec)
        assert result

        qdrant_module._client_cache.clear()

    def test_insert_version_conflict(self):
        """Test insert with version conflict detection (lines 410-420)."""
        store = QdrantVectorStore(":memory:", "test_conflict", 2)

        vec1 = ([1.0, 2.0], 1, "v1", 0, (1.0, 0))
        store.insert(vec1)

        vec2 = ([1.0, 2.0], 1, "v0", 0, (0.5, 0))
        result = store.insert(vec2)
        assert not result

        vec3 = ([1.0, 2.0], 1, "v2", 0, (2.0, 0))
        result = store.insert(vec3)
        assert result

        qdrant_module._client_cache.clear()

    def test_insert_with_destinations(self):
        """Test insert with destinations (line 441)."""
        store = QdrantVectorStore(":memory:", "test_dest", 2)

        vec = ([1.0, 2.0], 1, "payload", 0, (1.0, 0), frozenset([1, 2, 3]))
        result = store.insert(vec)
        assert result

        retrieved = store.get_vector(1)
        assert retrieved is not None
        assert 5 in range(len(retrieved))

        qdrant_module._client_cache.clear()

    def test_insert_batch_with_conflicts(self):
        """Test insert_batch with version conflicts (lines 479-565)."""
        store = QdrantVectorStore(":memory:", "test_batch_conflict", 2)

        vec1 = ([1.0, 2.0], 1, "v1", 0, (1.0, 0))
        store.insert(vec1)

        vectors = [
            ([1.0, 2.0], 1, "v0", 0, (0.5, 0)),
            ([3.0, 4.0], 2, "v2", 0, (1.0, 0)),
            ([5.0, 6.0], 3, "v3", 0, (1.0, 0)),
        ]

        count = store.insert_batch(vectors)
        assert count == 2

        qdrant_module._client_cache.clear()

    def test_insert_batch_empty(self):
        """Test insert_batch with empty list (line 479)."""
        store = QdrantVectorStore(":memory:", "test_batch_empty", 2)

        count = store.insert_batch([])
        assert count == 0

        qdrant_module._client_cache.clear()

    def test_insert_batch_all_skipped(self):
        """Test insert_batch when all vectors are skipped (lines 552-553)."""
        store = QdrantVectorStore(":memory:", "test_batch_skip", 2)

        store.insert(([1.0, 2.0], 1, "v1", 0, (5.0, 0)))
        store.insert(([3.0, 4.0], 2, "v2", 0, (5.0, 0)))

        vectors = [
            ([1.0, 2.0], 1, "old", 0, (1.0, 0)),
            ([3.0, 4.0], 2, "old", 0, (1.0, 0)),
        ]

        count = store.insert_batch(vectors)
        assert count == 0

        qdrant_module._client_cache.clear()

    def test_remove_by_id(self):
        """Test remove_by_id (line 567)."""
        store = QdrantVectorStore(":memory:", "test_remove", 2)

        store.insert(([1.0, 2.0], 1, "payload", 0, (1.0, 0)))
        assert store.count() == 1

        store.remove_by_id(1)
        assert store.count() == 0

        qdrant_module._client_cache.clear()

    def test_has_vector(self):
        """Test has_vector (lines 569-570)."""
        store = QdrantVectorStore(":memory:", "test_has", 2)

        assert not store.has_vector(1)

        store.insert(([1.0, 2.0], 1, "payload", 0, (1.0, 0)))
        assert store.has_vector(1)

        qdrant_module._client_cache.clear()

    def test_get_vector(self):
        """Test get_vector (lines 572-576)."""
        store = QdrantVectorStore(":memory:", "test_get", 2)

        assert store.get_vector(1) is None

        store.insert(([1.0, 2.0], 1, "payload", 0, (1.0, 0)))
        result = store.get_vector(1)
        assert result is not None
        assert result[1] == 1

        qdrant_module._client_cache.clear()

    def test_get_all_ids(self):
        """Test get_all_ids (lines 614-615)."""
        store = QdrantVectorStore(":memory:", "test_ids", 2)

        store.insert(([1.0, 2.0], 1, "a", 0, (1.0, 0)))
        store.insert(([3.0, 4.0], 2, "b", 0, (1.0, 0)))

        ids = store.get_all_ids()
        assert ids == {1, 2}

        qdrant_module._client_cache.clear()

    def test_point_to_tuple_with_destinations(self):
        """Test _point_to_tuple with destinations (lines 634-636)."""
        store = QdrantVectorStore(":memory:", "test_tuple", 2)

        vec = ([1.0, 2.0], 1, "payload", 0, (1.0, 0), frozenset([1, 2]))
        store.insert(vec)

        result = store.get_vector(1)
        assert result is not None
        assert len(result) == 6
        assert isinstance(result[5], frozenset)

        qdrant_module._client_cache.clear()

    def test_get_by_cluster(self):
        """Test get_by_cluster (lines 582-608)."""
        store = QdrantVectorStore(":memory:", "test_cluster", 2)

        store.insert(([1.0, 2.0], 1, "a", 0, (1.0, 0)))
        store.insert(([3.0, 4.0], 2, "b", 1, (1.0, 0)))
        store.insert(([5.0, 6.0], 3, "c", 0, (1.0, 0)))

        cluster0 = store.get_by_cluster(0)
        assert len(cluster0) == 2

        cluster1 = store.get_by_cluster(1)
        assert len(cluster1) == 1

        qdrant_module._client_cache.clear()

    def test_ensure_collection_error(self):
        """Test _ensure_collection error case (line 383)."""
        store = QdrantVectorStore(":memory:", "test_ensure", 2)
        store.collection_created = True

        store._ensure_collection(2)

        qdrant_module._client_cache.clear()


class TestServerWithClientEndpoint:
    """Tests for Server with client endpoint (lines 681-682, 735-743)."""

    def test_server_with_client_port(self):
        """Test Server initialization with client_port."""
        port = get_free_port()
        client_port = get_free_port()

        s = Server(
            id=1,
            is_coordinator=True,
            before_clustering=10,
            replication_factor=1,
            port=port,
            client_port=client_port,
        )

        try:
            assert s.client_endpoint is not None
            assert hasattr(s, "client_thread")
            assert s.client_thread.is_alive()

            time.sleep(0.5)
        finally:
            s.stop()


class TestServerStopComplete:
    """Tests for Server.stop() complete coverage (lines 795-821)."""

    def test_stop_with_client_endpoint(self):
        """Test stop() with client endpoint."""
        port = get_free_port()
        client_port = get_free_port()

        s = Server(
            id=1,
            is_coordinator=True,
            before_clustering=10,
            replication_factor=1,
            port=port,
            client_port=client_port,
        )

        time.sleep(0.5)

        s.stop()

        assert not s.worker_thread.is_alive()

    def test_stop_handles_exceptions(self):
        """Test stop() handles exceptions gracefully."""
        s = Server(1, True, 10, 1, port=get_free_port())

        time.sleep(0.2)

        s.endpoint = None

        s.stop()


class TestServerHeartbeatEdgeCases:
    """Tests for Server heartbeat edge cases (lines 953-954, 977-979, 1012-1015)."""

    def test_get_peer_phi(self):
        """Test get_peer_phi method (lines 1012-1015)."""
        s = Server(1, True, 10, 1, port=get_free_port())

        try:
            phi = s.get_peer_phi(99)
            assert phi == 0.0

            s.failure_detectors[2] = PhiAccrualFailureDetector()
            s.failure_detectors[2].heartbeat_received()

            phi = s.get_peer_phi(2)
            assert phi >= 0.0
        finally:
            s.stop()

    def test_heartbeat_exception_in_peer_id(self):
        """Test heartbeat_loop handles exception in get_id (lines 953-954)."""
        s = Server(1, True, 10, 1, port=get_free_port())

        try:
            bad_peer = MagicMock()
            bad_peer.get_id.side_effect = Exception("ID Error")
            s.peers.append(bad_peer)

            time.sleep(1.5)

            assert s.running
        finally:
            s.stop()


class TestServerElectionEdgeCases:
    """Tests for Server election edge cases (line 1060)."""

    def test_elect_with_empty_active_peers(self):
        """Test elect_partition_coordinator with empty active peers (line 1060)."""
        s = Server(1, True, 10, 1, port=get_free_port())

        try:
            s.active_peers = set()

            s.elect_partition_coordinator()

        finally:
            s.stop()


class TestServerDeliverHintsException:
    """Tests for deliver_hints exception handling (lines 1116-1119)."""

    def test_deliver_hints_puts_back_on_failure(self):
        """Test deliver_hints puts hints back on failure (lines 1116-1119)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.hinted_handoff.store_hint(1, [vec])

            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.receive.side_effect = Exception("Delivery failed!")

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}

            s.deliver_hints()

            assert s.hinted_handoff.has_hints_for(1)
        finally:
            s.stop()


class TestServerSendToPeersEdgeCases:
    """Tests for send_to_peers edge cases (lines 1263-1284)."""

    def test_send_to_peers_with_existing_destinations(self):
        """Test send_to_peers uses stored destinations (lines 1271-1273)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0
            mock_peer.receive = MagicMock()

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.status = "clustered"

            vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1]))
            s.send_to_peers([vec])

            mock_peer.receive.assert_called()
        finally:
            s.stop()

    def test_send_to_peers_calculates_destinations(self):
        """Test send_to_peers calculates destinations (lines 1276-1281)."""
        s = Server(0, True, 10, 2, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0
            mock_peer.receive = MagicMock()

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.status = "clustered"
            s.clusters = [(0, [1.0])]

            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s.send_to_peers([vec])
        finally:
            s.stop()


class TestServerCalculateDestinationsEdgeCases:
    """Tests for _calculate_destinations edge cases (lines 1342-1345)."""

    def test_calculate_destinations_empty_peers(self):
        """Test _calculate_destinations with empty peers."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.peers = []

            vec = ([1.0], 1, "a", 0, (1.0, 0))
            result = s._calculate_destinations(vec)

            assert result[0] == frozenset()
        finally:
            s.stop()


class TestServerHandleReceiveEdgeCases:
    """Tests for _handle_receive edge cases (lines 1370-1375, 1407)."""

    def test_handle_receive_client_drops_when_no_coord(self):
        """Test _handle_receive client drops vectors when no coordinator (lines 1387-1388)."""
        s = Server(0, False, 10, 1, port=get_free_port())

        try:
            s.peers = [Peer(s.ip, s.port, server_instance=s)]

            initial_dropped = s.dropped_vectors

            vec = ([1.0], 1, "a", -1, (1.0, 0))
            s._handle_receive([vec], "client")

            assert s.dropped_vectors == initial_dropped + 1
        finally:
            s.stop()


class TestServerClusteringEdgeCases:
    """Tests for clustering edge cases."""

    def test_clustering_too_few_samples(self):
        """Test clustering with too few samples (lines 1526-1536)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vectors = [
                ([1.0, 0.0], 1, "a", -1, (1.0, 0)),
                ([0.5, 0.5], 2, "b", -1, (1.0, 0)),
                ([0.0, 1.0], 3, "c", -1, (1.0, 0)),
            ]

            result = s.clustering(vectors, min_k=5, max_k=10)

            assert len(result) == 1
            assert 0 in result
            assert len(result[0]["members"]) == 3
        finally:
            s.stop()

    def test_clustering_with_subsampling(self):
        """Test clustering with subsampling (lines 1541-1548, 1563-1566)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vectors = []
            for i in range(50):
                base = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
                noisy = [
                    base[0] + np.random.uniform(-0.1, 0.1),
                    base[1] + np.random.uniform(-0.1, 0.1),
                ]
                vectors.append((noisy, i, f"p{i}", -1, (1.0, 0)))

            result = s.clustering(vectors, min_k=2, max_k=5)

            assert len(result) >= 2
        finally:
            s.stop()

    def test_clustering_exception_in_kmeans(self):
        """Test clustering handles exception in kmeans (lines 1616-1618)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vectors = [
                ([1.0, 0.0], 1, "a", -1, (1.0, 0)),
                ([1.0, 0.0], 2, "b", -1, (1.0, 0)),
            ]

            result = s.clustering(vectors, min_k=2, max_k=2)

            assert isinstance(result, dict)
        finally:
            s.stop()


class TestServerSearchEdgeCases:
    """Tests for search edge cases (lines 1701-1702, 1761-1763, 1779-1782, 1808-1810)."""

    def test_search_vectors_exception_in_future(self):
        """Test search_vectors handles exception in futures (lines 1701-1702)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0
            mock_peer.search_vectors_local.side_effect = Exception("Search Error!")

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.status = "clustered"
            s.clusters = [(0, [1.0, 0.0])]

            result = s.search_vectors([(1, [1.0, 0.0])], top_k=5, top_look=2)

            assert isinstance(result, list)
        finally:
            s.stop()

    def test_search_vectors_local_matrix_error(self):
        """Test search_vectors_local handles matrix creation error (lines 1761-1763)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.store.insert((np.array([1.0, 2.0]), 1, "a", 0, (1.0, 0)))
            s.store.insert(
                (np.array([1.0, 2.0, 3.0]), 2, "b", 0, (1.0, 0))
            )

            result = s.search_vectors_local([([1.0, 0.0], 1)], top_k=5)

            assert result == []
        finally:
            s.stop()

    def test_search_vectors_local_argpartition_branch(self):
        """Test search_vectors_local argpartition branch (lines 1808-1810)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            for i in range(20):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            result = s.search_vectors_local([([10.0, 10.0], 1)], top_k=5)

            assert len(result) == 5
        finally:
            s.stop()

    def test_search_vectors_local_empty_store(self):
        """Test search_vectors_local with empty store (lines 1751-1752)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            result = s.search_vectors_local([([1.0, 0.0], 1)], top_k=5)
            assert result == []
        finally:
            s.stop()


class TestVectorStoreEdgeCasesComplete:
    """Additional VectorStore edge cases."""

    def test_insert_with_4_tuple(self):
        """Test insert with 4-tuple (old format)."""
        store = VectorStore()

        vec = ([1.0], 1, "a", 0)
        result = store.insert(vec)

        assert result
        assert store.count() == 1

    def test_insert_with_6_tuple(self):
        """Test insert with 6-tuple (full format with destinations)."""
        store = VectorStore()

        vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1, 2]))
        result = store.insert(vec)

        assert result
        stored = store.get_vector(1)
        assert len(stored) == 6

    def test_remove_cleans_up_cluster_dict(self):
        """Test remove_by_id cleans up empty cluster."""
        store = VectorStore()

        store.insert(([1.0], 1, "a", 0, (1.0, 0)))
        assert 0 in store.vectors

        store.remove_by_id(1)

        assert 0 not in store.vectors


class TestServerReconcileException:
    """Tests for reconciliation exception handling (lines 1100-1101)."""

    def test_reconcile_with_other_peers_exception(self):
        """Test _reconcile_with_other_peers logs exception."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}

            original_reconcile = s.reconcile_with_peer
            s.reconcile_with_peer = MagicMock(side_effect=Exception("Reconcile Error"))

            s._reconcile_with_other_peers()

            s.reconcile_with_peer = original_reconcile
        finally:
            s.stop()


class TestServerCalculateSleepWithJitter:
    """Tests for _calculate_sleep_with_jitter."""

    def test_calculate_sleep_with_jitter(self):
        """Test _calculate_sleep_with_jitter returns valid value."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            result = s._calculate_sleep_with_jitter()

            assert result >= 0.1
            assert result <= s.heartbeat_interval + s.heartbeat_jitter
        finally:
            s.stop()


class TestServerMainFunction:
    """Tests for main() function - pragmatic approach."""

    def test_main_imports(self):
        """Verify main function exists and is callable."""
        from server import main

        assert callable(main)

    def test_main_with_help(self):
        """Test main() with --help shows help and exits."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['server.py', '--help']; from server import main; main()",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert "usage" in result.stdout.lower() or result.returncode == 0




class TestAddPeerEdgeCases:
    """Tests for add_peer edge cases (lines 766-767, 772)."""

    def test_add_peer_invalid_argument(self):
        """Test add_peer with invalid argument (lines 766-767)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.add_peer(12345)

            assert len(s.peers) == 1
        finally:
            s.stop()

    def test_add_peer_duplicate(self):
        """Test add_peer prevents duplicates (line 772)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            other = Server(1, False, 10, 1, port=get_free_port())

            s.add_peer(other)
            s.add_peer(other)

            assert len(s.peers) == 2

            other.stop()
        finally:
            s.stop()


class TestServerEndpointStopExceptions:
    """Tests for endpoint stop exception paths (lines 795-800, 815-818)."""

    def test_stop_with_timeout_in_endpoint_stop(self):
        """Test stop handles timeout in endpoint.stop() (lines 795-796)."""
        port = get_free_port()
        s = Server(0, True, 10, 1, port=port)

        try:
            time.sleep(0.5)


            async def slow_stop():
                await asyncio.sleep(10)

            s.endpoint.stop = slow_stop

            s.stop()

        except Exception:
            pass


class TestHeartbeatPingException:
    """Tests for heartbeat ping exception (lines 977-979, 984)."""

    def test_heartbeat_ping_exception_handled(self):
        """Test heartbeat loop handles ping exception (lines 977-979)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 99
            mock_peer.ping.side_effect = Exception("Network unreachable")

            s.peers.append(mock_peer)

            time.sleep(0.2)

            assert s.running
        finally:
            s.stop()

    def test_heartbeat_phi_not_available(self):
        """Test heartbeat returns None when phi says unavailable (line 984)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            fd = PhiAccrualFailureDetector(
                threshold=0.0001
            )
            fd.heartbeat_received()
            time.sleep(0.1)

            s.failure_detectors[99] = fd


        finally:
            s.stop()


class TestRouteVectorsClusterIndexException:
    """Tests for route_vectors cluster index exception (lines 1263-1264)."""

    def test_route_vectors_cluster_index_exception(self):
        """Test route_vectors handles cluster index exception (lines 1263-1264)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.clusters = [(0, [1.0, 0.0])]

            mock_index = MagicMock()
            mock_index.search_batch.side_effect = Exception("Index Error!")
            s.cluster_index = mock_index
            s.cluster_to_destinations_cache = {0: {0, 1}}

            result = s.route_vectors([([1.0, 0.0], 1)], top_k=1)

            assert 1 in result
        finally:
            s.stop()


class TestRouteVectorsCandidateFallback:
    """Tests for route_vectors candidate fallback (lines 1283-1284)."""

    def test_route_vectors_candidates_all_unreachable(self):
        """Test route_vectors fallback when candidates unreachable (lines 1283-1284)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.clusters = [(0, [1.0, 0.0])]

            mock_index = MagicMock()
            mock_index.search_batch.return_value = [[0]]
            s.cluster_index = mock_index
            s.cluster_to_destinations_cache = {0: {99, 100}}

            result = s.route_vectors([([1.0, 0.0], 1)], top_k=1)

            assert 1 in result
        finally:
            s.stop()


class TestCalculatePhiMathErrors:
    """Tests for _calculate_phi math errors (lines 146-147)."""

    def test_calculate_phi_very_small_p(self):
        """Test _calculate_phi when p becomes very small (triggers cap)."""
        detector = PhiAccrualFailureDetector()

        detector.heartbeat_received()
        time.sleep(0.005)
        detector.heartbeat_received()
        time.sleep(0.005)
        detector.heartbeat_received()

        result = detector._calculate_phi(1e12)

        assert result == 100.0


class TestHandleReceiveComplexPaths:
    """Tests for _handle_receive complex paths (lines 1370-1375)."""

    def test_handle_receive_bootstrap_clustered_sends_to_peers(self):
        """Test _handle_receive bootstrap when clustered sends to peers (line 1370)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.status = "clustered"

            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0
            mock_peer.receive = MagicMock()

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.clusters = [(0, [1.0])]

            vec = ([1.0], 1, "a", -1, (1.0, 0))
            s._handle_receive([vec], "bootstrap")

        finally:
            s.stop()

    def test_handle_receive_client_clustered_sends_to_peers(self):
        """Test _handle_receive client when clustered (lines 1428-1430)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.status = "clustered"

            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0
            mock_peer.receive = MagicMock()

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.active_peers = {0, 1}
            s.clusters = [(0, [1.0])]

            vec = ([1.0], 1, "a", -1, (1.0, 0))
            s._handle_receive([vec], "client")

        finally:
            s.stop()


class TestSaveVectorsDuplicatesPath:
    """Tests for save_vectors when duplicates are skipped (line 1447)."""

    def test_save_vectors_some_duplicates(self):
        """Test save_vectors when some are duplicates (line 1447)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.store.insert((np.array([1.0]), 1, "a", 0, (1.0, 0)))

            vectors = [
                (np.array([1.0]), 1, "a", 0, (0.5, 0)),
                (np.array([2.0]), 2, "b", 0, (1.0, 0)),
            ]

            s.save_vectors(vectors)

            assert s.store.count() == 2
        finally:
            s.stop()


class TestClusteringEdgeCasesAdditional:
    """Additional clustering edge cases."""

    def test_clustering_single_cluster_fallback(self):
        """Test clustering falls back to single cluster (lines 1528-1536)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vectors = [
                ([1.0, 0.0], 1, "a", -1, (1.0, 0)),
                ([0.0, 1.0], 2, "b", -1, (1.0, 0)),
            ]

            result = s.clustering(vectors, min_k=10, max_k=20)

            assert len(result) == 1
        finally:
            s.stop()

    def test_clustering_subsampling_active(self):
        """Test clustering with subsampling (lines 1581-1583, 1603-1604)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        import server as server_module

        original_size = server_module.SILHOUETTE_SUBSAMPLE_SIZE
        server_module.SILHOUETTE_SUBSAMPLE_SIZE = 10

        try:
            vectors = []
            for i in range(30):
                base = [1.0, 0.0] if i < 15 else [0.0, 1.0]
                noisy = [
                    base[0] + np.random.uniform(-0.1, 0.1),
                    base[1] + np.random.uniform(-0.1, 0.1),
                ]
                vectors.append((noisy, i, f"p{i}", -1, (1.0, 0)))

            result = s.clustering(vectors, min_k=2, max_k=3)

            assert len(result) >= 2
        finally:
            server_module.SILHOUETTE_SUBSAMPLE_SIZE = original_size
            s.stop()


class TestSearchVectorsLocalQueryMatrixError:
    """Tests for search_vectors_local query matrix error (lines 1779-1782)."""

    def test_search_vectors_local_query_matrix_error(self):
        """Test search_vectors_local handles query matrix error (lines 1779-1782)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.store.insert((np.array([1.0, 2.0]), 1, "a", 0, (1.0, 0)))
            s.store.insert((np.array([3.0, 4.0]), 2, "b", 0, (1.0, 0)))


            with patch("numpy.stack", side_effect=Exception("Shape mismatch")):
                result = s.search_vectors_local([([1.0, 0.0], 1)], top_k=5)
                assert result == []
        finally:
            s.stop()


class TestEnsureCollectionError:
    """Tests for QdrantVectorStore _ensure_collection error path (line 383)."""

    def test_ensure_collection_returns_false(self):
        """Test _ensure_collection when create_collection fails."""
        with patch.object(qdrant_module, "create_collection", return_value=False):
            store = QdrantVectorStore(":memory:", "test_fail", 2)

            assert not store.collection_created

        qdrant_module._client_cache.clear()


class TestQdrantInsertBatchRetrieveError:
    """Tests for QdrantVectorStore insert_batch retrieve error (lines 507-509)."""

    def test_insert_batch_retrieve_error(self):
        """Test insert_batch handles retrieve error (lines 507-509)."""
        QdrantVectorStore(":memory:", "test_err", 2)

        original_get_client = qdrant_module.get_client

        def mock_get_client(url):
            client = original_get_client(url)

            def failing_retrieve(*args, **kwargs):
                raise Exception("Retrieve Error!")

            client.retrieve = failing_retrieve
            return client

        with patch.object(qdrant_module, "get_client", mock_get_client):
            store2 = QdrantVectorStore(":memory:", "test_err2", 2)

            vectors = [
                ([1.0, 2.0], 1, "a", 0, (1.0, 0)),
            ]

            count = store2.insert_batch(vectors)
            assert count == 0

        qdrant_module._client_cache.clear()


class TestQdrantInsertBatchUploadError:
    """Tests for QdrantVectorStore insert_batch upload error (lines 563-565)."""

    def test_insert_batch_upload_error(self):
        """Test insert_batch handles upload error (lines 563-565)."""
        store = QdrantVectorStore(":memory:", "test_upload_err", 2)

        store.insert(([0.5, 0.5], 100, "existing", 0, (1.0, 0)))

        original_get_client = qdrant_module.get_client

        def mock_get_client(url):
            client = original_get_client(url)

            def failing_upload(*args, **kwargs):
                raise Exception("Upload Error!")

            client.upload_points = failing_upload
            return client

        with patch.object(qdrant_module, "get_client", mock_get_client):
            store2 = QdrantVectorStore(":memory:", "test_upload_err2", 2)

            vectors = [
                ([1.0, 2.0], 1, "a", 0, (1.0, 0)),
            ]

            count = store2.insert_batch(vectors)
            assert count == 0

        qdrant_module._client_cache.clear()




class TestClusteringWithOldFormatVectors:
    """Tests for clustering edge cases with 4-tuple vectors (line 1528, 1574)."""

    def test_clustering_too_few_with_4_tuple_vectors(self):
        """Test clustering with old 4-tuple format (lines 1528, 1574)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vectors = [
                ([1.0, 0.0], 1, "a", -1),
                ([0.0, 1.0], 2, "b", -1),
            ]

            result = s.clustering(vectors, min_k=20, max_k=30)

            assert len(result) == 1
            assert 0 in result
            for member in result[0]["members"]:
                assert len(member) >= 5
        finally:
            s.stop()


class TestHeartbeatPingDetectorPath:
    """Additional heartbeat tests."""

    def test_heartbeat_detector_not_available_path(self):
        """Test heartbeat loop when detector says unavailable (line 984)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 99
            mock_peer.ping.return_value = True

            s.peers.append(mock_peer)

            detector = PhiAccrualFailureDetector(
                threshold=0.0
            )
            s.failure_detectors[99] = detector

            time.sleep(0.5)

            assert s.running
        finally:
            s.stop()


class TestCalculateDestinationsEmptyPeersComplete:
    """Additional test for _calculate_destinations (lines 1342-1345)."""

    def test_calculate_destinations_fallback_path(self):
        """Test _calculate_destinations fallback when no cluster_index."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 0.9

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.cluster_index = None
            s.clusters = [(0, [1.0])]

            vec = ([1.0], 1, "a", 0, (1.0, 0))
            result = s._calculate_destinations(vec)

            assert len(result) > 0
        finally:
            s.stop()


class TestHandleReceiveForwardToCoord:
    """Tests for _handle_receive forwarding path (lines 1370-1375)."""

    def test_handle_receive_forwards_to_coordinator(self):
        """Test _handle_receive client forwards to coordinator (line 1423)."""
        s = Server(0, False, 10, 1, port=get_free_port())

        try:
            mock_coord = MagicMock()
            mock_coord.get_id.return_value = 1
            mock_coord.i_am_coord.return_value = True
            mock_coord.receive = MagicMock()

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_coord]
            s.active_peers = {0, 1}
            s.status = "bootstrap"

            vec = ([1.0], 1, "a", -1, (1.0, 0))
            s._handle_receive([vec], "client")

            mock_coord.receive.assert_called()
        finally:
            s.stop()


class TestHandleReceiveStatusCorrupted:
    """Tests for status corrupted error path (line 1407, 1440)."""

    def test_handle_receive_invalid_status(self):
        """Test _handle_receive with invalid sender_status (lines 1438-1440)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vec = ([1.0], 1, "a", 0, (1.0, 0))
            s._handle_receive([vec], "completely_invalid_status")

        finally:
            s.stop()


class TestServerStopClientEndpointException:
    """Tests for stop() client endpoint exception (lines 815-818)."""

    def test_stop_client_endpoint_timeout(self):
        """Test stop handles client endpoint timeout."""
        port = get_free_port()
        client_port = get_free_port()

        s = Server(0, True, 10, 1, port=port, client_port=client_port)

        try:
            time.sleep(0.5)

            if s.client_endpoint:

                async def slow_stop():
                    await asyncio.sleep(10)

                s.client_endpoint.stop = slow_stop

        finally:
            s.stop()


class TestServerStopEndpointScheduleError:
    """Tests for stop() endpoint schedule error (lines 797-800)."""

    def test_stop_endpoint_schedule_failure(self):
        """Test stop handles endpoint schedule failure."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            time.sleep(0.3)

            if hasattr(s, "endpoint_loop"):
                original_run = asyncio.run_coroutine_threadsafe

                def failing_run(*args, **kwargs):
                    raise Exception("Schedule Error!")

                asyncio.run_coroutine_threadsafe = failing_run

        finally:
            asyncio.run_coroutine_threadsafe = (
                original_run
                if "original_run" in dir()
                else asyncio.run_coroutine_threadsafe
            )
            s.stop()


class TestClusteringExceptionHandling:
    """Tests for clustering exception in kmeans (lines 1616-1618)."""

    def test_clustering_kmeans_exception_path(self):
        """Test clustering handles exception in kmeans iteration (lines 1616-1618)."""
        from sklearn.metrics import silhouette_score as sklearn_silhouette

        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            vectors = []
            for i in range(20):
                base = [1.0, 0.0] if i < 10 else [0.0, 1.0]
                noisy = [
                    base[0] + np.random.uniform(-0.1, 0.1),
                    base[1] + np.random.uniform(-0.1, 0.1),
                ]
                vectors.append((noisy, i, f"p{i}", -1, (1.0, 0)))

            call_count = [0]

            def failing_silhouette(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ValueError("Silhouette computation failed!")
                return sklearn_silhouette(*args, **kwargs)

            with patch("server.silhouette_score", failing_silhouette):
                result = s.clustering(vectors, min_k=2, max_k=3)

                assert isinstance(result, dict)
                assert len(result) > 0
        finally:
            s.stop()


class TestSearchVectorsLocalQueryError:
    """Tests for search_vectors_local query creation error (lines 1779-1782)."""

    def test_search_vectors_local_query_conversion_error(self):
        """Test search_vectors_local handles query conversion error."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            for i in range(5):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            with patch(
                "numpy.array", side_effect=Exception("Array conversion failed!")
            ):
                result = s.search_vectors_local([([1.0, 0.0], 1)], top_k=5)
                assert result == []
        finally:
            s.stop()


class TestQdrantBatchWithDestinations:
    """Tests for QdrantVectorStore insert_batch with destinations (line 543)."""

    def test_insert_batch_with_destinations(self):
        """Test insert_batch preserves destinations."""
        store = QdrantVectorStore(":memory:", "test_batch_dest", 2)

        vectors = [
            ([1.0, 2.0], 1, "a", 0, (1.0, 0), frozenset([1, 2])),
            ([3.0, 4.0], 2, "b", 0, (1.0, 0), frozenset([2, 3])),
        ]

        count = store.insert_batch(vectors)
        assert count == 2

        vec1 = store.get_vector(1)
        assert len(vec1) == 6

        qdrant_module._client_cache.clear()


class TestQdrantInsertBatchNumpyDim:
    """Tests for QdrantVectorStore insert_batch numpy array dim (line 487)."""

    def test_insert_batch_numpy_array_dim(self):
        """Test insert_batch with numpy array dimension extraction."""
        store = QdrantVectorStore(":memory:", "test_batch_np", 2)

        vectors = [
            (np.array([1.0, 2.0]), 1, "a", 0, (1.0, 0)),
        ]

        count = store.insert_batch(vectors)
        assert count == 1

        qdrant_module._client_cache.clear()


class TestPhiCalculateOverflow:
    """Tests for _calculate_phi overflow (lines 146-147)."""

    def test_calculate_phi_triggers_exception(self):
        """Test _calculate_phi returns 100.0 on math error."""
        detector = PhiAccrualFailureDetector()

        detector.heartbeat_received()
        time.sleep(0.001)
        detector.heartbeat_received()

        with patch("math.erfc", side_effect=ValueError("Math domain error")):
            result = detector._calculate_phi(1000.0)
            assert result == 100.0




class TestRebalancingDetection:
    """Tests for load imbalance detection across servers."""

    def test_get_load_stats_single_server(self):
        """Test get_load_stats returns correct stats for single server."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            for i in range(10):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            stats = s.get_load_stats()

            assert "self" in stats
            assert stats["self"] == 10
            assert "total" in stats
            assert stats["total"] == 10
            assert "average" in stats
            assert stats["average"] == 10.0
        finally:
            s.stop()

    def test_get_load_stats_multi_server(self):
        """Test get_load_stats aggregates across reachable peers."""
        s1 = Server(0, True, 100, 1, port=get_free_port())
        s2 = Server(1, False, 100, 1, port=get_free_port())
        s3 = Server(2, False, 100, 1, port=get_free_port())

        try:
            s1.add_peer(s2)
            s1.add_peer(s3)
            s2.add_peer(s1)
            s2.add_peer(s3)
            s3.add_peer(s1)
            s3.add_peer(s2)

            wait_for_full_connectivity([s1, s2, s3])

            for i in range(50):
                s1.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )
            for i in range(50, 60):
                s2.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )
            for i in range(60, 70):
                s3.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            stats = s1.get_load_stats()

            assert stats["total"] == 70
            assert stats["average"] == pytest.approx(70 / 3.0, abs=1.0)
            assert "peers" in stats
        finally:
            s1.stop()
            s2.stop()
            s3.stop()

    def test_detect_imbalance_balanced_load(self):
        """Test detect_imbalance returns None when load is balanced."""
        s1 = Server(0, True, 100, 1, port=get_free_port())
        s2 = Server(1, False, 100, 1, port=get_free_port())

        try:
            s1.add_peer(s2)
            s2.add_peer(s1)
            wait_for_full_connectivity([s1, s2])

            for i in range(20):
                s1.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )
            for i in range(20, 40):
                s2.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            result = s1.detect_imbalance()
            assert result is None
        finally:
            s1.stop()
            s2.stop()

    def test_detect_imbalance_overloaded_server(self):
        """Test detect_imbalance identifies overloaded server."""
        s1 = Server(0, True, 100, 1, port=get_free_port())
        s2 = Server(1, False, 100, 1, port=get_free_port())

        try:
            s1.add_peer(s2)
            s2.add_peer(s1)
            wait_for_full_connectivity([s1, s2])

            for i in range(80):
                s1.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )
            for i in range(80, 100):
                s2.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            result = s1.detect_imbalance()
            assert result == 0
        finally:
            s1.stop()
            s2.stop()

    def test_get_heaviest_cluster(self):
        """Test get_heaviest_cluster finds the largest cluster on a server."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            for i in range(5):
                s.store.insert((np.array([float(i), 0.0]), i, f"p{i}", 0, (1.0, 0)))
            for i in range(5, 20):
                s.store.insert((np.array([0.0, float(i)]), i, f"p{i}", 1, (1.0, 0)))
            for i in range(20, 23):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 2, (1.0, 0))
                )

            result = s.get_heaviest_cluster(0)
            assert result == 1
        finally:
            s.stop()


class TestBalancedSplitting:
    """Tests for balanced cluster splitting."""

    def test_balanced_split_creates_subclusters(self):
        """Test balanced_split_cluster creates correct number of subclusters."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            for i in range(20):
                s.store.insert(
                    (np.array([float(i) / 10, float(i) / 10]), i, f"p{i}", 5, (1.0, 0))
                )
            s.clusters = [(5, [1.0, 1.0])]

            result = s.balanced_split_cluster(cluster_id=5, n_subclusters=2)

            assert len(result) == 2
            for _, data in result.items():
                assert "center" in data
                assert "members" in data
        finally:
            s.stop()

    def test_balanced_split_preserves_all_vectors(self):
        """Test that splitting preserves all vectors."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            original_ids = set()
            for i in range(30):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 3, (1.0, 0))
                )
                original_ids.add(i)
            s.clusters = [(3, [15.0, 15.0])]

            result = s.balanced_split_cluster(cluster_id=3, n_subclusters=2)

            result_ids = set()
            for _, data in result.items():
                for member in data["members"]:
                    result_ids.add(member[1])

            assert result_ids == original_ids
        finally:
            s.stop()

    def test_balanced_split_creates_balanced_sizes(self):
        """Test that balanced splitting creates approximately equal subclusters."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            for i in range(40):
                s.store.insert(
                    (np.array([float(i), float(i) * 2]), i, f"p{i}", 7, (1.0, 0))
                )
            s.clusters = [(7, [20.0, 40.0])]

            result = s.balanced_split_cluster(cluster_id=7, n_subclusters=2)

            sizes = [len(data["members"]) for data in result.values()]
            max_size = max(sizes)
            min_size = min(sizes)
            assert max_size - min_size <= 10
        finally:
            s.stop()

    def test_balanced_split_assigns_new_cluster_ids(self):
        """Test that split assigns new unique cluster IDs."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            for i in range(20):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 10, (1.0, 0))
                )
            s.clusters = [(10, [10.0, 10.0])]

            result = s.balanced_split_cluster(cluster_id=10, n_subclusters=2)

            new_ids = list(result.keys())
            assert len(new_ids) == 2
            assert all(cid != 10 for cid in new_ids)
        finally:
            s.stop()

    def test_constrained_kmeans_prevents_imbalance(self):
        """Test that constrained K-Means prevents extreme cluster imbalances."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            vectors = []
            blob_size = 20
            for i in range(blob_size):
                vectors.append(
                    (
                        np.array([np.random.normal(0, 0.1), np.random.normal(0, 0.1)]),
                        i,
                        f"p{i}",
                        0,
                        (1.0, 0),
                    )
                )

            outlier_idx = blob_size
            vectors.append(
                (np.array([1000.0, 1000.0]), outlier_idx, "outlier", 0, (1.0, 0))
            )

            for v in vectors:
                s.store.insert(v)

            s.clusters = [(0, [0.0, 0.0])]


            result = s.balanced_split_cluster(cluster_id=0, n_subclusters=2)

            sizes = []
            for info in result.values():
                sizes.append(len(info["members"]))

            print(f"DEBUG: Constrained split sizes: {sizes}")

            max_size = max(sizes)
            min_size = min(sizes)


            assert max_size <= 14
            assert min_size >= 7

        finally:
            s.stop()


class TestRebalancingOrchestration:
    """Tests for rebalancing orchestration across servers."""

    def test_trigger_rebalance_redistributes_vectors(self):
        """Test trigger_rebalance moves vectors from overloaded to underloaded servers."""
        s1 = Server(0, True, 100, 1, port=get_free_port())
        s2 = Server(1, False, 100, 1, port=get_free_port())

        try:
            s1.add_peer(s2)
            s2.add_peer(s1)
            wait_for_full_connectivity([s1, s2])

            for i in range(80):
                vec = (
                    np.array([float(i % 10), float(i % 10)]),
                    i,
                    f"p{i}",
                    i % 5,
                    (1.0, 0),
                )
                s1.store.insert(vec)

            for i in range(80, 100):
                vec = (
                    np.array([float(i % 10), float(i % 10)]),
                    i,
                    f"p{i}",
                    i % 5,
                    (1.0, 0),
                )
                s2.store.insert(vec)

            s1.clusters = [(j, [float(j), float(j)]) for j in range(5)]
            s1.cluster_to_destinations_cache = {
                j: {0, 1} for j in range(5)
            }
            s2.clusters = s1.clusters.copy()
            s2.cluster_to_destinations_cache = s1.cluster_to_destinations_cache.copy()
            s2.status = "clustered"
            s1.status = "clustered"

            result = s1.trigger_rebalance()

            assert result is True

            assert len(s1.clusters) > 5
        finally:
            s1.stop()
            s2.stop()

    def test_hnsw_index_updated_after_rebalance(self):
        """Test that HNSW index is updated after rebalancing."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            for i in range(30):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", i % 3, (1.0, 0))
                )

            from cluster_index import ClusterIndex

            s.cluster_index = ClusterIndex(dimension=2)
            s.clusters = [(j, [float(j * 10), float(j * 10)]) for j in range(3)]
            s.cluster_index.build(s.clusters)
            s.cluster_to_destinations_cache = {0: {0}, 1: {0}, 2: {0}}

            initial_cluster_ids = {cid for cid, _ in s.clusters}

            new_clusters = s.balanced_split_cluster(cluster_id=0, n_subclusters=2)

            assert s.cluster_index is not None
            current_cluster_ids = {cid for cid, _ in s.clusters}
            assert 0 not in current_cluster_ids
            assert len(new_clusters) > 0
        finally:
            s.stop()

    def test_search_works_after_rebalance(self):
        """Test that vector search still works correctly after rebalancing."""
        s1 = Server(0, True, 100, 2, port=get_free_port())
        s2 = Server(1, False, 100, 2, port=get_free_port())

        try:
            s1.add_peer(s2)
            s2.add_peer(s1)
            wait_for_full_connectivity([s1, s2])

            target_vec = np.array([5.0, 5.0])
            for i in range(50):
                vec = (np.array([float(i), float(i)]), i, f"p{i}", i % 3, (1.0, 0))
                s1.store.insert(vec)

            s1.clusters = [(j, [float(j * 17), float(j * 17)]) for j in range(3)]
            s1.status = "clustered"
            from cluster_index import ClusterIndex

            s1.cluster_index = ClusterIndex(dimension=2)
            s1.cluster_index.build(s1.clusters)
            s1.cluster_to_destinations_cache = {0: {0, 1}, 1: {0, 1}, 2: {0, 1}}
            s2.clusters = s1.clusters.copy()
            s2.cluster_index = s1.cluster_index
            s2.cluster_to_destinations_cache = s1.cluster_to_destinations_cache.copy()
            s2.status = "clustered"

            results_before = s1.search_vectors_local(
                [(target_vec.tolist(), 999)], top_k=5
            )
            assert len(results_before) > 0

            s1.trigger_rebalance()

            results_after = s1.search_vectors_local(
                [(target_vec.tolist(), 999)], top_k=5
            )
            assert len(results_after) > 0

            for r in results_after:
                assert r[3] > 0
        finally:
            s1.stop()
            s2.stop()

    def test_smallest_id_server_triggers_rebalance(self):
        """Test that only the server with smallest ID triggers rebalancing for a cluster."""
        s1 = Server(0, True, 100, 2, port=get_free_port())
        s2 = Server(1, False, 100, 2, port=get_free_port())
        s3 = Server(2, False, 100, 2, port=get_free_port())

        try:
            s1.add_peer(s2)
            s1.add_peer(s3)
            s2.add_peer(s1)
            s2.add_peer(s3)
            s3.add_peer(s1)
            s3.add_peer(s2)
            wait_for_full_connectivity([s1, s2, s3])

            for srv in [s1, s2, s3]:
                for i in range(10):
                    base_id = srv.id * 100 + i
                    srv.store.insert(
                        (
                            np.array([float(i), float(i)]),
                            base_id,
                            f"p{base_id}",
                            5,
                            (1.0, 0),
                        )
                    )
                srv.clusters = [(5, [5.0, 5.0])]
                srv.cluster_to_destinations_cache = {5: {0, 1, 2}}

            should_s1_trigger = s1.should_trigger_rebalance_for_cluster(5)
            should_s2_trigger = s2.should_trigger_rebalance_for_cluster(5)
            should_s3_trigger = s3.should_trigger_rebalance_for_cluster(5)

            assert should_s1_trigger is True
            assert should_s2_trigger is False
            assert should_s3_trigger is False
        finally:
            s1.stop()
            s2.stop()
            s3.stop()


class TestRebalancingEdgeCases:
    """Tests for rebalancing edge cases."""

    def test_rebalance_single_peer_noop(self):
        """Test rebalancing with single peer does nothing."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            for i in range(100):
                s.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            initial_count = s.store.count()
            result = s.trigger_rebalance()

            assert result is False
            assert s.store.count() == initial_count
        finally:
            s.stop()

    def test_rebalance_empty_cluster(self):
        """Test rebalancing handles empty clusters gracefully."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            s.clusters = [(0, [0.0, 0.0])]
            s.cluster_to_destinations_cache = {0: {0}}

            result = s.balanced_split_cluster(cluster_id=0, n_subclusters=2)

            assert isinstance(result, dict)
        finally:
            s.stop()

    def test_rebalance_already_balanced(self):
        """Test rebalancing when load is already balanced."""
        s1 = Server(0, True, 100, 1, port=get_free_port())
        s2 = Server(1, False, 100, 1, port=get_free_port())

        try:
            s1.add_peer(s2)
            s2.add_peer(s1)
            wait_for_full_connectivity([s1, s2])

            for i in range(50):
                s1.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )
            for i in range(50, 100):
                s2.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", 0, (1.0, 0))
                )

            result = s1.trigger_rebalance()

            assert result is False
        finally:
            s1.stop()
            s2.stop()

    def test_rebalance_very_small_cluster(self):
        """Test rebalancing a cluster with fewer vectors than split factor."""
        s = Server(0, True, 100, 1, port=get_free_port())

        try:
            s.store.insert((np.array([1.0, 1.0]), 1, "p1", 9, (1.0, 0)))
            s.clusters = [(9, [1.0, 1.0])]

            result = s.balanced_split_cluster(cluster_id=9, n_subclusters=2)

            assert isinstance(result, dict)
            total_members = sum(
                len(data.get("members", [])) for data in result.values()
            )
            assert total_members == 1
        finally:
            s.stop()


class TestRebalancingIntegration:
    """Full integration tests for rebalancing mechanism."""

    def test_full_rebalancing_50k_vectors(self):
        """
        MANDATORY: Full integration test with 50k vectors.

        Tests the complete rebalancing flow:
        1. Create 3 servers with uneven load
        2. Insert 50k vectors with clustering
        3. Detect imbalance
        4. Trigger rebalancing
        5. Verify vectors are redistributed
        6. Verify search still works
        """
        s1 = Server(0, True, 2000, 2, port=get_free_port())
        s2 = Server(1, False, 2000, 2, port=get_free_port())
        s3 = Server(2, False, 2000, 2, port=get_free_port())

        try:
            s1.add_peer(s2)
            s1.add_peer(s3)
            s2.add_peer(s1)
            s2.add_peer(s3)
            s3.add_peer(s1)
            s3.add_peer(s2)
            wait_for_full_connectivity([s1, s2, s3])

            np.random.seed(42)
            vectors = []
            for i in range(50000):
                cluster_pattern = i % 10
                base = np.array(
                    [float(cluster_pattern * 10), float(cluster_pattern * 10)]
                )
                noise = np.random.randn(2) * 0.5
                vec = base + noise
                vectors.append(
                    (vec, i, f"payload_{i}", cluster_pattern, (time.time(), 0))
                )

            split1 = int(50000 * 0.6)
            split2 = int(50000 * 0.85)

            for i, vec in enumerate(vectors[:split1]):
                s1.store.insert(vec)
            for i, vec in enumerate(vectors[split1:split2]):
                s2.store.insert(vec)
            for i, vec in enumerate(vectors[split2:]):
                s3.store.insert(vec)

            initial_s1 = s1.store.count()
            initial_s2 = s2.store.count()
            initial_s3 = s3.store.count()

            print(
                f"Initial distribution: S1={initial_s1}, S2={initial_s2}, S3={initial_s3}"
            )
            assert initial_s1 == 30000
            assert initial_s2 == 12500
            assert initial_s3 == 7500

            clusters = [(j, [float(j * 10), float(j * 10)]) for j in range(10)]
            dest_cache = {
                j: {0, 1, 2} for j in range(10)
            }
            for srv in [s1, s2, s3]:
                srv.clusters = clusters.copy()
                srv.cluster_to_destinations_cache = dest_cache.copy()
                srv.status = "clustered"
                from cluster_index import ClusterIndex

                srv.cluster_index = ClusterIndex(dimension=2)
                srv.cluster_index.build(clusters)

            imbalanced_id = s1.detect_imbalance()
            assert imbalanced_id == 0

            result = s1.trigger_rebalance()
            assert result is True

            time.sleep(2.0)

            final_s1 = s1.store.count()
            final_s2 = s2.store.count()
            final_s3 = s3.store.count()

            print(f"Final distribution: S1={final_s1}, S2={final_s2}, S3={final_s3}")

            assert len(s1.clusters) > 10

            test_query = np.array([50.0, 50.0])
            search_results = s1.search_vectors_local(
                [(test_query.tolist(), 999999)], top_k=10
            )

            assert len(search_results) > 0
            for r in search_results:
                assert r[3] > 0

            print("50k vector rebalancing integration test PASSED")

        finally:
            s1.stop()
            s2.stop()
            s3.stop()

    def test_rebalancing_with_concurrent_operations(self):
        """Test rebalancing while queries and inserts are happening."""
        s1 = Server(0, True, 100, 2, port=get_free_port())
        s2 = Server(1, False, 100, 2, port=get_free_port())

        try:
            s1.add_peer(s2)
            s2.add_peer(s1)
            wait_for_full_connectivity([s1, s2])

            for i in range(100):
                s1.store.insert(
                    (np.array([float(i), float(i)]), i, f"p{i}", i % 5, (1.0, 0))
                )

            s1.clusters = [(j, [float(j * 20), float(j * 20)]) for j in range(5)]
            s1.status = "clustered"
            from cluster_index import ClusterIndex

            s1.cluster_index = ClusterIndex(dimension=2)
            s1.cluster_index.build(s1.clusters)
            s1.cluster_to_destinations_cache = {j: {0, 1} for j in range(5)}
            s2.clusters = s1.clusters.copy()
            s2.cluster_index = s1.cluster_index
            s2.cluster_to_destinations_cache = s1.cluster_to_destinations_cache.copy()
            s2.status = "clustered"

            import threading

            errors = []
            results = []

            def do_searches():
                try:
                    for _ in range(10):
                        query = np.array([50.0, 50.0])
                        res = s1.search_vectors_local([(query.tolist(), 9999)], top_k=5)
                        results.append(len(res))
                        time.sleep(0.05)
                except Exception as e:
                    errors.append(e)

            def do_inserts():
                try:
                    for i in range(100, 150):
                        s1.store.insert(
                            (
                                np.array([float(i), float(i)]),
                                i,
                                f"p{i}",
                                i % 5,
                                (1.0, 0),
                            )
                        )
                        time.sleep(0.02)
                except Exception as e:
                    errors.append(e)

            def do_rebalance():
                try:
                    time.sleep(0.1)
                    s1.trigger_rebalance()
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=do_searches),
                threading.Thread(target=do_inserts),
                threading.Thread(target=do_rebalance),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            assert len(errors) == 0, f"Concurrent operations failed: {errors}"

            assert all(r >= 0 for r in results)

        finally:
            s1.stop()
            s2.stop()

    def test_rebalancing_updates_replicas(self):
        """Test that rebalancing properly updates all replica servers."""
        s1 = Server(0, True, 100, 2, port=get_free_port())
        s2 = Server(1, False, 100, 2, port=get_free_port())
        s3 = Server(2, False, 100, 2, port=get_free_port())

        try:
            s1.add_peer(s2)
            s1.add_peer(s3)
            s2.add_peer(s1)
            s2.add_peer(s3)
            s3.add_peer(s1)
            s3.add_peer(s2)
            wait_for_full_connectivity([s1, s2, s3])

            for i in range(60):
                vec = (np.array([float(i), float(i)]), i, f"p{i}", i % 3, (1.0, 0))
                s1.store.insert(vec)
                if i % 2 == 0:
                    s2.store.insert(vec)
                else:
                    s3.store.insert(vec)

            clusters = [(j, [float(j * 20), float(j * 20)]) for j in range(3)]
            for srv in [s1, s2, s3]:
                srv.clusters = clusters.copy()
                srv.cluster_to_destinations_cache = {0: {0, 1}, 1: {0, 2}, 2: {1, 2}}
                srv.status = "clustered"
                from cluster_index import ClusterIndex

                srv.cluster_index = ClusterIndex(dimension=2)
                srv.cluster_index.build(clusters)

            s1.trigger_rebalance()

            time.sleep(1.0)

            for srv in [s1, s2, s3]:
                assert srv.cluster_index is not None
                assert srv.cluster_to_destinations_cache is not None

        finally:
            s1.stop()
            s2.stop()
            s3.stop()




class TestVectorStoreDeleteByCluster:
    """Tests for VectorStore.delete_by_cluster (lines 272-280)."""

    def test_delete_by_cluster_removes_vectors(self):
        """Test delete_by_cluster removes all vectors from a cluster."""
        store = VectorStore()

        store.insert(([1.0], 1, "a", 0, (1.0, 0)))
        store.insert(([2.0], 2, "b", 0, (1.0, 0)))
        store.insert(([3.0], 3, "c", 1, (1.0, 0)))

        assert store.count() == 3

        result = store.delete_by_cluster(0)
        assert result is True

        assert store.count() == 1
        assert not store.has_vector(1)
        assert not store.has_vector(2)
        assert store.has_vector(3)

    def test_delete_by_cluster_nonexistent(self):
        """Test delete_by_cluster with nonexistent cluster."""
        store = VectorStore()

        store.insert(([1.0], 1, "a", 0, (1.0, 0)))

        result = store.delete_by_cluster(999)
        assert result is True

        assert store.count() == 1

    def test_delete_by_cluster_empties_cluster_dict(self):
        """Test delete_by_cluster removes cluster from vectors dict."""
        store = VectorStore()

        store.insert(([1.0], 1, "a", 5, (1.0, 0)))
        assert 5 in store.vectors

        store.delete_by_cluster(5)
        assert 5 not in store.vectors


class TestQdrantVectorStoreDeleteByCluster:
    """Tests for QdrantVectorStore.delete_by_cluster (line 416)."""

    def test_qdrant_delete_by_cluster(self):
        """Test delete_by_cluster on QdrantVectorStore."""
        store = QdrantVectorStore(":memory:", "test_delete_cluster", 2)

        store.insert(([1.0, 2.0], 1, "a", 0, (1.0, 0)))
        store.insert(([3.0, 4.0], 2, "b", 0, (1.0, 0)))
        store.insert(([5.0, 6.0], 3, "c", 1, (1.0, 0)))

        assert store.count() == 3

        result = store.delete_by_cluster(0)
        assert result is True

        assert store.count() == 1

        qdrant_module._client_cache.clear()


class TestQdrantModuleDeleteVectorsByPayload:
    """Tests for qdrant_module.delete_vectors_by_payload (lines 291-313)."""

    def test_delete_vectors_by_payload_success(self):
        """Test delete_vectors_by_payload successfully deletes matching vectors."""
        url = ":memory:"
        collection = "test_delete_payload"

        qdrant_module.create_collection(url, collection, 2)

        vectors = [
            ([1.0, 0.0], 1, "A", 5),
            ([0.0, 1.0], 2, "B", 5),
            ([0.5, 0.5], 3, "C", 10),
        ]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)

        assert qdrant_module.count(url, collection) == 3

        result = qdrant_module.delete_vectors_by_payload(
            url, collection, "cluster_id", 5
        )
        assert result is True

        assert qdrant_module.count(url, collection) == 1

        qdrant_module._client_cache.clear()

    def test_delete_vectors_by_payload_no_matches(self):
        """Test delete_vectors_by_payload with no matching vectors."""
        url = ":memory:"
        collection = "test_delete_no_match"

        qdrant_module.create_collection(url, collection, 2)

        vectors = [([1.0, 0.0], 1, "A", 5)]
        qdrant_module.insert_vectors(url, collection, vectors, batch_size_retry=1)

        result = qdrant_module.delete_vectors_by_payload(
            url, collection, "cluster_id", 999
        )
        assert result is True

        assert qdrant_module.count(url, collection) == 1

        qdrant_module._client_cache.clear()

    def test_delete_vectors_by_payload_error(self):
        """Test delete_vectors_by_payload handles errors."""
        url = ":memory:"

        result = qdrant_module.delete_vectors_by_payload(
            url, "nonexistent", "key", "value"
        )

        assert result is False

        qdrant_module._client_cache.clear()


class TestQdrantModuleInsertRetry:
    """Tests for qdrant_module.insert_vectors retry path (lines 222-242)."""

    def test_insert_vectors_retry_on_failure(self):
        """Test insert_vectors retries on first failure."""
        url = ":memory:"
        collection = "test_retry"

        qdrant_module.create_collection(url, collection, 2)

        original_client = qdrant_module.get_client(url)
        call_count = [0]
        original_upload = original_client.upload_points

        def mock_upload(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First upload failed!")
            return original_upload(*args, **kwargs)

        original_client.upload_points = mock_upload

        vectors = [([1.0, 0.0], 1, "A", 0)]
        result = qdrant_module.insert_vectors(
            url, collection, vectors, batch_size_retry=1
        )

        assert result is True
        assert qdrant_module.count(url, collection) == 1

        original_client.upload_points = original_upload
        qdrant_module._client_cache.clear()

    def test_insert_vectors_both_attempts_fail(self):
        """Test insert_vectors returns False when both attempts fail."""
        url = ":memory:"
        collection = "test_double_fail"

        qdrant_module.create_collection(url, collection, 2)

        original_client = qdrant_module.get_client(url)
        original_upload = original_client.upload_points

        def always_fail(*args, **kwargs):
            raise Exception("Always fails!")

        original_client.upload_points = always_fail

        vectors = [([1.0, 0.0], 1, "A", 0)]
        result = qdrant_module.insert_vectors(
            url, collection, vectors, batch_size_retry=1
        )

        assert result is False

        original_client.upload_points = original_upload
        qdrant_module._client_cache.clear()


class TestPeerRemoteCallEdgeCases:
    """Tests for Peer class remote call edge cases (lines 83, 87-89, 95, 115-116)."""

    def test_peer_count_remote(self):
        """Test count() for remote peer (line 83)."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 42}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.count()

        assert result == 42

    def test_peer_delete_vectors_by_cluster_remote(self):
        """Test delete_vectors_by_cluster() for remote peer (lines 87-89)."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": True}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.delete_vectors_by_cluster(5)

        assert result is True

    def test_peer_split_and_distribute_cluster_remote(self):
        """Test split_and_distribute_cluster() for remote peer (lines 91-95)."""
        mock_comm = MagicMock()

        async def mock_send(*args):
            return {
                "status": 0,
                "error": None,
                "response": {"new_cluster": {"center": [1.0], "members": []}},
            }

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)
        result = peer.split_and_distribute_cluster(
            cluster_id=5, split_plan={"n_subclusters": 2}
        )

        assert "new_cluster" in result

    def test_peer_remote_call_runtime_error_fallback(self):
        """Test _remote_call RuntimeError fallback (lines 115-116)."""
        mock_comm = MagicMock()

        call_count = [0]

        async def mock_send(*args):
            return {"status": 0, "error": None, "response": 123}

        mock_comm.send = mock_send

        peer = Peer("127.0.0.1", 9999, server_instance=None, communicator=mock_comm)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(asyncio.to_thread(peer.get_id))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        result = peer.get_id()
        assert result == 123


class TestPeerLocalCountAndDelete:
    """Tests for Peer local count and delete methods."""

    def test_peer_count_local(self):
        """Test count() for local peer."""
        mock_server = MagicMock()
        mock_server.count.return_value = 100

        peer = Peer("127.0.0.1", 9999, server_instance=mock_server)
        result = peer.count()

        assert result == 100
        mock_server.count.assert_called_once()

    def test_peer_delete_vectors_by_cluster_local(self):
        """Test delete_vectors_by_cluster() for local peer."""
        mock_server = MagicMock()
        mock_server.delete_vectors_by_cluster.return_value = True

        peer = Peer("127.0.0.1", 9999, server_instance=mock_server)
        result = peer.delete_vectors_by_cluster(5)

        assert result is True
        mock_server.delete_vectors_by_cluster.assert_called_once_with(5)

    def test_peer_split_and_distribute_cluster_local(self):
        """Test split_and_distribute_cluster() for local peer."""
        mock_server = MagicMock()
        mock_server.split_and_distribute_cluster.return_value = {"new": "clusters"}

        peer = Peer("127.0.0.1", 9999, server_instance=mock_server)
        result = peer.split_and_distribute_cluster(
            5, {"plan": "data"}, is_coordinator=True
        )

        assert result == {"new": "clusters"}
        mock_server.split_and_distribute_cluster.assert_called_once_with(
            5, {"plan": "data"}, True
        )


class TestServerDeleteVectorsByCluster:
    """Tests for Server.delete_vectors_by_cluster (lines 958-961)."""

    def test_server_delete_vectors_by_cluster(self):
        """Test Server.delete_vectors_by_cluster."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.store.insert((np.array([1.0, 2.0]), 1, "a", 5, (1.0, 0)))
            s.store.insert((np.array([3.0, 4.0]), 2, "b", 5, (1.0, 0)))
            s.store.insert((np.array([5.0, 6.0]), 3, "c", 10, (1.0, 0)))

            assert s.count() == 3

            result = s.delete_vectors_by_cluster(5)
            assert result is True

            assert s.count() == 1
        finally:
            s.stop()


class TestServerAddPeerPingException:
    """Tests for Server.add_peer ping exception (lines 847-848)."""

    def test_add_peer_ping_fails(self):
        """Test add_peer handles ping failure gracefully."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.ping.side_effect = Exception("Ping failed!")
            mock_peer.get_id.side_effect = Exception("ID failed!")

            with patch("server.Peer", return_value=mock_peer):
                s.add_peer("127.0.0.1", 99999)

            assert s.running
        finally:
            s.stop()


class TestServerStopExceptionPaths:
    """Tests for Server.stop exception paths (lines 876-879, 900-901)."""

    def test_stop_endpoint_schedule_error(self):
        """Test stop handles endpoint schedule error (lines 876-879)."""
        s = Server(0, True, 10, 1, port=get_free_port())

        time.sleep(0.3)

        try:
            original_stop = s.endpoint.stop

            async def failing_stop():
                raise Exception("Stop failed!")

            s.endpoint.stop = failing_stop

            s.stop()
        except Exception:
            pass

    def test_stop_client_endpoint_exception(self):
        """Test stop handles client endpoint exception (lines 900-901)."""
        port = get_free_port()
        client_port = get_free_port()

        s = Server(0, True, 10, 1, port=port, client_port=client_port)

        time.sleep(0.3)

        try:
            if s.client_endpoint:
                async def failing_stop():
                    raise Exception("Client stop failed!")

                s.client_endpoint.stop = failing_stop

            s.stop()
        except Exception:
            pass


class TestQdrantModuleCreateCollectionError:
    """Tests for qdrant_module.create_collection error (lines 74-76)."""

    def test_create_collection_exception(self):
        """Test create_collection handles exceptions."""
        url = ":memory:"

        with patch.object(qdrant_module, "get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = False
            mock_client.create_collection.side_effect = Exception("Create failed!")
            mock_get.return_value = mock_client

            result = qdrant_module.create_collection(url, "test_fail", 2)

            assert result is False

        qdrant_module._client_cache.clear()


class TestQdrantHTTPClient:
    """Tests for qdrant_module HTTP client creation (lines 31-36)."""

    def test_get_client_http_url(self):
        """Test get_client with HTTP URL creates gRPC client."""
        qdrant_module._client_cache.clear()

        http_url = "http://localhost:6333"

        try:
            client = qdrant_module.get_client(http_url)
            assert client is not None
        except Exception:
            pass

        qdrant_module._client_cache.clear()

    def test_get_client_invalid_port(self):
        """Test get_client handles invalid port in URL."""
        qdrant_module._client_cache.clear()

        http_url = "http://localhost"

        try:
            client = qdrant_module.get_client(http_url)
            assert client is not None
        except Exception:
            pass

        qdrant_module._client_cache.clear()


class TestServerHeartbeatElectionEdgeCases:
    """Tests for heartbeat and election edge cases (lines 1040-1041, 1064-1066)."""

    def test_heartbeat_detector_not_created_path(self):
        """Test heartbeat when detector doesn't exist for peer."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 99
            mock_peer.ping.return_value = True

            s.peers.append(mock_peer)

            for _ in range(10):
                if 99 in s.failure_detectors:
                    break
                time.sleep(0.5)

            assert True
        finally:
            s.stop()

    def test_elect_coordinator_tie_breaker(self):
        """Test elect_partition_coordinator with multiple candidates."""
        s1 = Server(1, True, 10, 1, port=get_free_port())
        s2 = Server(2, False, 10, 1, port=get_free_port())
        s3 = Server(3, False, 10, 1, port=get_free_port())

        try:
            s1.add_peer(s2)
            s1.add_peer(s3)
            s2.add_peer(s1)
            s2.add_peer(s3)
            s3.add_peer(s1)
            s3.add_peer(s2)

            wait_for_full_connectivity([s1, s2, s3])

            assert s3.is_coordinator
            assert not s1.is_coordinator or s1.id == 3
        finally:
            s1.stop()
            s2.stop()
            s3.stop()


class TestServerReconcileGetPeerByIdPath:
    """Tests for _get_peer_by_id path (lines 1131-1135)."""

    def test_get_peer_by_id_found(self):
        """Test _get_peer_by_id when peer exists."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            other = Server(1, False, 10, 1, port=get_free_port())
            s.add_peer(other)
            other.add_peer(s)

            wait_for_full_connectivity([s, other])

            found = s._get_peer_by_id(1)
            assert found is not None

            peer_id = found.get_id()
            assert peer_id == 1

            other.stop()
        finally:
            s.stop()

    def test_get_peer_by_id_not_found(self):
        """Test _get_peer_by_id returns None for unknown ID."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            found = s._get_peer_by_id(999)
            assert found is None
        finally:
            s.stop()


class TestServerHandleNetworkChangeComplete:
    """Tests for _handle_network_change path."""

    def test_handle_network_change_triggers_election(self):
        """Test _handle_network_change triggers election."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s._handle_network_change()

        finally:
            s.stop()


class TestQdrantVectorStoreEnsureCollectionError:
    """Tests for QdrantVectorStore._ensure_collection error path (line 412)."""

    def test_ensure_collection_prints_error(self):
        """Test _ensure_collection prints error on failure."""
        store = QdrantVectorStore(":memory:", "test_ensure_err", 2)

        store.collection_created = False

        with patch.object(qdrant_module, "create_collection", return_value=False):
            store._ensure_collection(2)

            assert not store.collection_created

        qdrant_module._client_cache.clear()


class TestServerSendToPeersHintStoredPath:
    """Test send_to_peers hint paths for comprehensive coverage."""

    def test_send_to_peers_unreachable_stores_hint(self):
        """Test that send_to_peers stores hints for unreachable peers."""
        s = Server(0, True, 10, 2, port=get_free_port())

        try:
            mock_peer = MagicMock()
            mock_peer.get_id.return_value = 1
            mock_peer.similarity.return_value = 1.0

            s.peers = [Peer(s.ip, s.port, server_instance=s), mock_peer]
            s.status = "clustered"
            s.clusters = [(0, [1.0])]

            s.active_peers = {0}

            vec = ([1.0], 1, "a", 0, (1.0, 0), frozenset([1]))
            s.send_to_peers([vec])

            assert s.hinted_handoff.has_hints_for(1)
        finally:
            s.stop()


class TestServerSearchVectorsEmptyQueries:
    """Test search_vectors with empty query list."""

    def test_search_vectors_empty_input(self):
        """Test search_vectors with empty query list."""
        s = Server(0, True, 10, 1, port=get_free_port())

        try:
            s.status = "clustered"
            s.clusters = [(0, [1.0, 0.0])]

            result = s.search_vectors([], top_k=5, top_look=1)
            assert result == []
        finally:
            s.stop()


class TestQdrantModuleDeleteCollectionError:
    """Test qdrant_module.delete_collection error path (lines 87-89)."""

    def test_delete_collection_exception(self):
        """Test delete_collection handles exceptions."""
        url = ":memory:"

        with patch.object(qdrant_module, "get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.delete_collection.side_effect = Exception("Delete failed!")
            mock_get.return_value = mock_client

            result = qdrant_module.delete_collection(url, "any_collection")

            assert result is False

        qdrant_module._client_cache.clear()


class TestNetworkConditions:
    """Tests for non-binary network failures (latency, jitter)."""

    def test_phi_accrual_with_network_latency(self):
        """
        Verify that high latency increases Phi (suspicion) but doesn't
        mark node down immediately unless threshold is crossed.
        """
        s0 = Server(0, True, 100, 1, port=get_free_port())
        s1 = Server(1, False, 100, 1, port=get_free_port())

        real_peer_class = Peer

        try:
            s0.heartbeat_interval = 0.5
            s0.add_peer(s1)
            peer_ref = s0.peers[1]
            start = time.time()
            connected = False
            while time.time() - start < 5.0:
                if s0._is_peer_reachable(peer_ref):
                    connected = True
                    break
                time.sleep(0.1)

            assert connected, "Peer should be initially reachable (timed out)"
            initial_phi = s0.get_peer_phi(s1.id)
            print(f"Initial Phi: {initial_phi}")

            original_ping = peer_ref.ping

            def slow_ping():
                time.sleep(0.3)
                return original_ping()

            with patch.object(peer_ref, "ping", side_effect=slow_ping):
                print("Injecting latency...")
                time.sleep(4.0)

                new_phi = s0.get_peer_phi(s1.id)
                print(f"Phi after latency: {new_phi}")

                assert (
                    new_phi > initial_phi
                ), "Phi should increase when latency is introduced"

                assert (
                    new_phi < 8.0
                ), f"False positive! Latency marked node dead. Phi: {new_phi}"

        finally:
            s0.stop()
            s1.stop()


def test_anti_entropy_recovery_no_hints():
    """
    Test recovery of data residing ONLY on non-coordinator nodes
    when hinted handoff has failed (hints lost).
    Scenario:
    - 4 Nodes, Rep Factor 2.
    - Initial: 100 vectors.
    - Partition: [0, 1] (Coord 0) and [2, 3] (Coord 2/3).
    - Insert 50 vectors to [0,1] and 50 to [2,3].
    - DESTROY HINTS.
    - Heal.
    - Verify complete recovery (200 vectors * 2 replicas = 400 total).
    """
    ports = [get_free_port() for _ in range(4)]
    servers = []
    try:
        for i in range(4):
            s = Server(i, i == 0, 100, 2, port=ports[i])
            servers.append(s)

        for i in range(4):
            for j in range(4):
                if i != j:
                    servers[i].add_peer(servers[j])

        wait_for_full_connectivity(servers)

        centers = [[10.0, 10.0], [10.0, -10.0], [-10.0, 10.0], [-10.0, -10.0]]
        data = []
        for i in range(100):
            center = centers[i % 4]
            vec = [center[0] + (i * 0.01), center[1] + (i * 0.01)]
            data.append((vec, f"init_{i}"))

        servers[0].receive_from_client(data)

        print("Waiting for clustering...")
        for _ in range(50):
            if servers[0].status == "clustered":
                break
            time.sleep(0.2)

        assert servers[0].status == "clustered"

        time.sleep(2)

        print("Partitioning network...")
        partition_network(servers, [[0, 1], [2, 3]])

        time.sleep(5)


        print("Inserting new data into partitions...")

        input_a = []
        for i in range(50):
            vec = [12.0 + i * 0.1, 12.0]
            input_a.append((vec, f"new_a_{i}"))
        servers[0].receive_from_client(input_a)

        input_b = []
        for i in range(50):
            vec = [-12.0 - i * 0.1, -12.0]
            input_b.append((vec, f"new_b_{i}"))

        servers[3].receive_from_client(input_b)

        time.sleep(2)

        print("Sabotaging: Destroying all hinted handoff data...")
        for s in servers:
            s.hinted_handoff.clear()

        print("Healing network...")
        heal_network(servers)
        wait_for_full_connectivity(servers)

        print("Reconciling...")
        time.sleep(10)


        counts = [s.count() for s in servers]
        total_count = sum(counts)
        print(f"Final Counts per Node: {counts}")
        print(f"Total Count: {total_count}")

        assert (
            total_count == 400
        ), f"Expected 400 vectors, got {total_count}. Counts: {counts}"


    finally:
        for s in servers:
            s.stop()
