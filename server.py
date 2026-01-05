import itertools
import heapq
import math
import random
from collections import deque
from compound_types import *
from sklearn.cluster import KMeans
import numpy as np
import threading
import queue
import time
import concurrent.futures
import asyncio
from endpoint import Endpoint
from communicator import Communicator
from typing import Dict, Set, Optional, TYPE_CHECKING
from cluster_index import ClusterIndex
import qdrant_module
from qdrant_module import GLOBAL_LOCK
from qdrant_client import models
from client_endpoint.interface import ClientEndpoint
import argparse
import sys


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    # Optimized to handle both lists and numpy arrays without redundant conversion
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        dot_product = np.dot(v1, v2)
        norm_a = np.linalg.norm(v1)
        norm_b = np.linalg.norm(v2)
    else:
        a = np.array(v1)
        b = np.array(v2)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class HintedHandoff:
    """
    Store hints for unreachable nodes. Deliver when connectivity restored.
    
    Implements the hinted handoff pattern for partition tolerance:
    when a target node is unreachable, vectors are stored locally and
    delivered when the node becomes reachable again.
    """
    
    def __init__(self):
        self.hints: Dict[int, List] = {}  # target_id -> List[VectorComplete]
        self.lock = threading.RLock()
    
    def store_hint(self, target_id: int, vectors: List):
        """Store vectors to deliver to target when reachable."""
        with self.lock:
            if target_id not in self.hints:
                self.hints[target_id] = []
            self.hints[target_id].extend(vectors)
    
    def get_hints_for(self, target_id: int) -> List:
        """Get and clear hints for target."""
        with self.lock:
            hints = self.hints.pop(target_id, [])
            return hints
    
    def peek_hints_for(self, target_id: int) -> List:
        """Get hints for target without clearing."""
        with self.lock:
            return list(self.hints.get(target_id, []))
    
    def has_hints_for(self, target_id: int) -> bool:
        """Check if there are pending hints."""
        with self.lock:
            return target_id in self.hints and len(self.hints[target_id]) > 0
    
    def get_all_targets(self) -> List[int]:
        """Get all target IDs with pending hints."""
        with self.lock:
            return [tid for tid, hints in self.hints.items() if hints]
    
    def count(self) -> int:
        """Get total number of stored hints."""
        with self.lock:
            return sum(len(hints) for hints in self.hints.values())
    
    def clear(self):
        """Clear all hints."""
        with self.lock:
            self.hints.clear()


    # ============ Peer & Server Implementation ============


from peer.peer import Peer


class VectorStore:
    """
    Thread-safe vector storage with version tracking and deduplication.

    Stores vectors organized by cluster ID, with tracking of vector IDs
    to prevent duplicates and support version-based conflict resolution.
    """
    
    def __init__(self):
        self.vectors = {}  # map cluster_id -> list of vectors
        self.vector_ids: Dict[int, Tuple] = {}  # vector_id -> vector tuple (for dedup)
        self.lock = threading.RLock()

    def insert(self, vector) -> bool:
        """
        Insert a vector, handling duplicates via version comparison.
        
        Vector format: (Vector, VectorId, VectorPayload, cluster_id, version)
        where version is (timestamp, node_id) or None for legacy format.
        
        Returns True if inserted, False if rejected (older version exists).
        """
        with self.lock:
            vec_id = vector[1]
            cluster_id = vector[3]
            
            # Get version (handle both old 4-tuple and new 5-tuple format)
            version = vector[4] if len(vector) > 4 else (0, 0)
            
            if vec_id in self.vector_ids:
                # Check version, keep newer
                existing = self.vector_ids[vec_id]
                existing_version = existing[4] if len(existing) > 4 else (0, 0)
                
                if version <= existing_version:
                    return False  # Existing is newer or equal
                
                # Remove old version
                self._remove_by_id_internal(vec_id)
            
            # Insert new vector
            # OPTIMIZATION: Convert to numpy array immediately upon insertion
            # Handle variable tuple length (legacy vs new)
            v_data = vector[0]
            if not isinstance(v_data, np.ndarray):
                v_data = np.array(v_data, dtype=np.float32)
            
            # Reconstruct vector tuple with numpy array
            if len(vector) == 6:
                vector = (v_data, vector[1], vector[2], vector[3], vector[4], vector[5])
            elif len(vector) == 5:
                # Add None for missing destinations if needed, or just keep as is
                vector = (v_data, vector[1], vector[2], vector[3], vector[4])
            else:
                vector = (v_data, vector[1], vector[2], vector[3])

            self.vector_ids[vec_id] = vector
            if cluster_id not in self.vectors:
                self.vectors[cluster_id] = []
            self.vectors[cluster_id].append(vector)
            return True
    
    def _remove_by_id_internal(self, vec_id: int):
        """Remove vector by ID (internal, assumes lock held)."""
        if vec_id not in self.vector_ids:
            return
        
        old_vec = self.vector_ids.pop(vec_id)
        old_cluster = old_vec[3]
        
        if old_cluster in self.vectors:
            self.vectors[old_cluster] = [
                v for v in self.vectors[old_cluster] if v[1] != vec_id
            ]
            if not self.vectors[old_cluster]:
                del self.vectors[old_cluster]
    
    def remove_by_id(self, vec_id: int):
        """Remove vector by ID."""
        with self.lock:
            self._remove_by_id_internal(vec_id)
    
    def has_vector(self, vec_id: int) -> bool:
        """Check if vector exists."""
        with self.lock:
            return vec_id in self.vector_ids
    
    def get_vector(self, vec_id: int):
        """Get vector by ID."""
        with self.lock:
            return self.vector_ids.get(vec_id)

    def get_all(self):
        with self.lock:
            all_vectors = []
            for v_list in self.vectors.values():
                all_vectors.extend(v_list)
            return all_vectors

    def get_by_cluster(self, cluster_id):
        with self.lock:
            return list(self.vectors.get(cluster_id, []))
            
    def count(self):
        with self.lock:
            return len(self.vector_ids)
    
    def get_all_ids(self) -> Set[int]:
        """Get all stored vector IDs."""
        with self.lock:
            return set(self.vector_ids.keys())

    def insert_batch(self, vectors) -> int:
        """Insert multiple vectors. Returns number of successes."""
        with self.lock:
            count = 0
            for v in vectors:
                if self.insert(v):
                    count += 1
            return count


class QdrantVectorStore:
    """
    Qdrant-backed vector storage.
    """
    def __init__(self, url, collection_name, vector_dim):
        self.url = url
        self.collection_name = collection_name
        self.lock = threading.RLock()
        self.collection_created = False
        
        self._ensure_collection(vector_dim)

    def _ensure_collection(self, vector_dim):
        if not self.collection_created:
            res = qdrant_module.create_collection(self.url, self.collection_name, vector_dim)
            if res:
                 self.collection_created = True
            else:
                 print(f"Error creating collection {self.collection_name}")


    def insert(self, vector) -> bool:
        """
        Insert a vector, handling duplicates via version comparison.
        """
        # vector format: (Vector, VectorId, VectorPayload, cluster_id, version, destinations)
        with self.lock:
            # Check dim
            v_data = vector[0]
            if isinstance(v_data, list):
                dim = len(v_data)
            else:
                dim = v_data.shape[0]
                
            self._ensure_collection(dim)
            
            vec_id = vector[1]
            cluster_id = vector[3]
            version = vector[4] if len(vector) > 4 else (0, 0)
            
            # Check existing
            existing_point = qdrant_module.retrieve_vector(self.url, self.collection_name, vec_id)
            
            if existing_point:
                # payload is a dict
                payload = existing_point.payload
                # recover version from payload
                # We need to store version in payload
                # payload structure in qdrant_module insert: {"string": vector_payload, "cluster_id": cluster_id}
                # We should add "version_ts" and "version_node"
                v_ts = payload.get("version_ts", 0)
                v_node = payload.get("version_node", 0)
                existing_version = (v_ts, v_node)
                
                if version <= existing_version:
                    return False
            
            # Prepare insert
            # vector payload is stored in vector[2]
            payload_str = vector[2]
            
            # Deconstruct version
            v_ts, v_node = version
            
            # Store everything we need to reconstruct the tuple
            # We also need 'destinations' if it exists (index 5)
            destinations = vector[5] if len(vector) > 5 else None
            
            # Create special Qdrant payload
            q_payload = {
                "string": payload_str, 
                "cluster_id": cluster_id,
                "version_ts": v_ts,
                "version_node": v_node,
            }
            if destinations:
                q_payload["destinations"] = list(destinations)

            # Insert using module (we need to bypass insert_vectors wrapper because it constructs payload differently)
            # OR we modify insert_vectors to accept full payload?
            # insert_vectors takes: (vector_content, vector_id, vector_payload, cluster_id)
            # and constructs payload={"string": vector_payload, "cluster_id": cluster_id}
            # This is too restrictive.
            # I should use client directly here or update module?
            # I'll update module later if needed, but for now I can modify this class to use client directly?
            # No, better to stick to module abstractions if possible or extend module.
            # But the 'insert_vectors' function in module is very specific.
            # I will assume I can modify qdrant_module.py again to support generic payload?
            # OR I can just use insert_vectors_generic if I pack my payload into 'vector_payload' as a dict?
            # But insert_vectors creates specific dict structure.
            
            # Let's call client.upload_points directly here since I have logic.
            # Or use qdrant_module.get_client
            
            client = qdrant_module.get_client(self.url)
            
            point = models.PointStruct(
                id=vec_id,
                vector=v_data if isinstance(v_data, list) else v_data.tolist(),
                payload=q_payload
            )
            
            client.upload_points(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            return True

    def insert_batch(self, vectors) -> int:
        """
        Batch insert vectors with version checking.
        """
        if not vectors:
            return 0
            
        with self.lock, GLOBAL_LOCK:
            # 1. Ensure collection exists (check dimension of first vector)
            v0_data = vectors[0][0]
            if isinstance(v0_data, list):
                dim = len(v0_data)
            else:
                dim = v0_data.shape[0]
            self._ensure_collection(dim)
            
            # 2. Retrieve existing versions for all IDs
            ids = [v[1] for v in vectors]
            client = qdrant_module.get_client(self.url)
            from qdrant_client import models
            
            # Retrieve existing points to check versions
            # We use scroll logic or retrieve logic?
            # retrieve takes generic list of IDs.
            # qdrant_module.retrieve_vector is single.
            # client.retrieve() returns list of Record.
            try:
                existing_records = client.retrieve(
                    collection_name=self.collection_name,
                    ids=ids,
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as e:
                print(f"Error retrieving for batch check: {e}")
                return 0

            existing_map = {rec.id: rec.payload for rec in existing_records}
            
            points_to_upload = []
            
            for vector in vectors:
                vec_id = vector[1]
                version = vector[4] if len(vector) > 4 else (0, 0)
                
                # Check version conflict
                if vec_id in existing_map:
                    payload = existing_map[vec_id]
                    v_ts = payload.get("version_ts", 0)
                    v_node = payload.get("version_node", 0)
                    existing_version = (v_ts, v_node)
                    
                    if version <= existing_version:
                        continue # Skip old version
                
                # Prepare payload
                payload_str = vector[2]
                cluster_id = vector[3]
                v_ts, v_node = version
                destinations = vector[5] if len(vector) > 5 else None
                v_data = vector[0]
                
                q_payload = {
                    "string": payload_str, 
                    "cluster_id": cluster_id,
                    "version_ts": v_ts,
                    "version_node": v_node,
                }
                if destinations:
                    q_payload["destinations"] = list(destinations)

                point = models.PointStruct(
                    id=vec_id,
                    vector=v_data if isinstance(v_data, list) else v_data.tolist(),
                    payload=q_payload
                )
                points_to_upload.append(point)
            
            if not points_to_upload:
                return 0
                
            # 3. Batch upload
            try:
                client.upload_points(
                    collection_name=self.collection_name,
                    points=points_to_upload,
                    wait=True
                )
                return len(points_to_upload)
            except Exception as e:
                print(f"Error in batch upload: {e}")
                return 0
    def remove_by_id(self, vec_id: int):
        qdrant_module.delete_vector(self.url, self.collection_name, vec_id)

    def has_vector(self, vec_id: int) -> bool:
        return qdrant_module.retrieve_vector(self.url, self.collection_name, vec_id) is not None

    def get_vector(self, vec_id: int):
        point = qdrant_module.retrieve_vector(self.url, self.collection_name, vec_id)
        if not point:
            return None
        return self._point_to_tuple(point)

    def get_all(self):
        points = qdrant_module.get_all_vectors(self.url, self.collection_name)
        return [self._point_to_tuple(p) for p in points]

    def get_by_cluster(self, cluster_id):
        # We need to filter by cluster_id.
        # qdrant_module doesn't export filter search.
        # Implement using scroll with filter
        client = qdrant_module.get_client(self.url)
        from qdrant_client import models
        
        filter_condition = models.Filter(
            must=[models.FieldCondition(key="cluster_id", match=models.MatchValue(value=cluster_id))]
        )
        
        all_points = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            all_points.extend(points)
            if offset is None:
                break
        
        return [self._point_to_tuple(p) for p in all_points]

    def count(self):
        return qdrant_module.count(self.url, self.collection_name)

    def get_all_ids(self) -> Set[int]:
        points = qdrant_module.get_all_vectors(self.url, self.collection_name)
        return {p.id for p in points}

    def _point_to_tuple(self, point):
        # Reconstruct tuple from PointStruct
        # Tuple: (Vector, VectorId, VectorPayload, cluster_id, version, destinations)
        payload = point.payload
        vec_data = point.vector
        vec_id = point.id
        
        p_str = payload.get("string")
        c_id = payload.get("cluster_id")
        v_ts = payload.get("version_ts", 0)
        v_node = payload.get("version_node", 0)
        dest = payload.get("destinations")
        
        version = (v_ts, v_node)
        
        # Tuple reconstruction
        # Handle destinations (frozenset)
        if dest:
            dest = frozenset(dest)
            return (vec_data, vec_id, p_str, c_id, version, dest)
        else:
            return (vec_data, vec_id, p_str, c_id, version)


class Server:
    def __init__(self, id, is_coordinator, before_clustering, replication_factor, port, ip="127.0.0.1", endpoint="QUIC", qdrant_url=None, vector_dim=384, client_port=None):
        self.id = id
        self.ip = ip
        self.port = port
        self.initial_coordinator = is_coordinator
        self.is_coordinator = is_coordinator
        self.before_clustering = before_clustering
        self.replication_factor = replication_factor
        
        # Initialize communicator first so it can be passed to peers
        self.communicator = Communicator(endpoint)
        
        # Local peer (self)
        self.peers = []
        self._add_local_peer()
        
        self.clusters = []
        self.status = 'bootstrap'
        self.qdrant_url = qdrant_url
        if qdrant_url:
            self.store = QdrantVectorStore(qdrant_url, f"node_{id}_vectors", vector_dim)
        else:
            self.store = VectorStore()
        self.vector_id = 0
        self.vector_buffer = []
        self.dropped_vectors = 0
        
        # Partition tolerance state
        self.partition_coordinator_id = id if is_coordinator else None
        self.hinted_handoff = HintedHandoff()

        self.endpoint = Endpoint(endpoint, self)
        
        # Start endpoint in a thread to handle async loop
        self._start_endpoint_thread(port)
        
        # Client Endpoint (Optional)
        self.client_endpoint = None
        if client_port:
            self.client_endpoint = ClientEndpoint(self)
            self._start_client_endpoint(client_port)
        
        # Network simulation (Client controlled)
        self.simulated_unreachable_peers: Set[int] = set()
        self.active_peers: Set[int] = {id} # Initially assume only self is reachable until gossip
        
        # Phi Accrual Failure Detectors (one per peer)
        # Configuration: threshold=8 (99.9999% certainty), sliding window of 500 samples
        self.failure_detectors: Dict[int, PhiAccrualFailureDetector] = {}
        self.phi_threshold = 8.0  # Industrial standard threshold
        self.phi_window_size = 500  # Sliding window size
        
        # Heartbeat configuration
        self.heartbeat_interval = 3.0  # Base interval in seconds
        self.heartbeat_jitter = 0.5  # Jitter: ±0.5 seconds (uniform random)
        
        # Track when reconciliation is in progress
        self._reconciling = False
        self._reconcile_lock = threading.Lock()
        
        # Track when clustering is in progress
        self.clustering_in_progress = False
        
        # HNSW Index
        self.cluster_index = None
        self.cluster_to_destinations_cache = None # Map cluster_id -> frozenset(peer_ids)

        
        # Concurrency control
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.running = True
        
        # Worker thread
        self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.worker_thread.start()

        # Heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def _start_endpoint_thread(self, port):
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.endpoint_loop = loop
            loop.run_until_complete(self.endpoint.start("0.0.0.0", port))
            loop.run_forever()
        
        self.endpoint_thread = threading.Thread(target=run_loop, daemon=True)
        self.endpoint_thread.start()

    def _start_client_endpoint(self, port):
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.client_loop = loop
            loop.run_until_complete(self.client_endpoint.start("0.0.0.0", port))
            loop.run_forever()
            
        self.client_thread = threading.Thread(target=run_loop, daemon=True)
        self.client_thread.start()

    def _add_local_peer(self):
        # Create a peer instance representing THIS server
        local_peer = Peer(self.ip, self.port, server_instance=self)
        self.peers.append(local_peer)

    def add_peer(self, peer_or_ip, port: int = None):
        """
        Add a peer to the server.
        Can accept:
        1. (ip, port) for remote peer.
        2. Server instance for local simulation/testing.
        """
        with self.lock:
            new_peer = None
            if isinstance(peer_or_ip, str) and port is not None:
                # Remote Peer
                new_peer = Peer(peer_or_ip, port, communicator=self.communicator)
            elif hasattr(peer_or_ip, 'ip') and hasattr(peer_or_ip, 'port'):
                # Server instance (Simulation)
                new_peer = Peer(peer_or_ip.ip, peer_or_ip.port, server_instance=peer_or_ip)
            else:
                print(f"Error: Invalid argument to add_peer: {peer_or_ip}, {port}")
                return

            # Avoid duplications
            for p in self.peers:
                if p.ip == new_peer.ip and p.port == new_peer.port:
                    return
            self.peers.append(new_peer)

    def stop(self):
        self.running = False
        self.queue.put(None)  # Sentinel to unblock queue
        self.worker_thread.join()
        
        # Stop heartbeat thread
        if hasattr(self, 'heartbeat_thread') and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1)

        # Stop endpoint thread
        if hasattr(self, 'endpoint_loop'):
            # Schedule cleanup and stop
            try:
                if self.endpoint_loop.is_running():
                    coro = self.endpoint.stop()
                    try:
                        future = asyncio.run_coroutine_threadsafe(coro, self.endpoint_loop)
                        # Wait for cleanup with timeout
                        try:
                            future.result(timeout=2)
                        except (concurrent.futures.TimeoutError, Exception) as e:
                            print(f"Endpoint stop warning/error for server {self.id}: {e}")
                    except Exception as e:
                         # Failed to schedule, close coroutine to avoid warning
                         coro.close()
                         print(f"Error scheduling endpoint stop for server {self.id}: {e}")
            except Exception as e:
                print(f"Error initiating endpoint stop for server {self.id}: {e}")
            finally:
                self.endpoint_loop.call_soon_threadsafe(self.endpoint_loop.stop)
                self.endpoint_thread.join(timeout=1)

        # Stop client endpoint
        if hasattr(self, 'client_loop'):
             try:
                if self.client_loop.is_running():
                    coro = self.client_endpoint.stop()
                    try:
                        future = asyncio.run_coroutine_threadsafe(coro, self.client_loop)
                        future.result(timeout=2)
                    except Exception as e:
                         pass
             except Exception:
                 pass
             finally:
                 self.client_loop.call_soon_threadsafe(self.client_loop.stop)
                 self.client_thread.join(timeout=1)

        
    def process_queue(self):
        while self.running:
            try:
                task = self.queue.get()
                if task is None:
                    break
                # print(f"DEBUG: Server {self.id} popping task. Method: {task[0].__name__}")
                method, args = task
                method(*args)
                self.queue.task_done()
            except Exception as e:
                print(f"Error in server {self.id}: {e}")


    
    def i_am_coord(self):
        return self.is_coordinator
    
    def coordinator(self):
        """Find the coordinator among reachable peers."""
        with self.lock:
            reachable = self.get_reachable_peers()
            for peer in reachable:
                if peer.i_am_coord():
                    return peer
        return None
    
    def get_new_vector_id(self):
        with self.lock:
            self.vector_id += 1
            to_mult = 1 if len(self.peers) == 0 else len(str(abs(len(self.peers))))
            return self.vector_id * (10 ** to_mult) + self.get_id()

    def get_status(self):
        with self.lock:
            return self.status

    def get_queue_size(self):
        return self.queue.qsize()

    def is_clustering(self):
        return self.clustering_in_progress
    
    def set_status(self, status):
        with self.lock:
            self.status = status
    
    def get_id(self):
        return self.id
    
    def count(self):
        with self.lock:
            return self.store.count()
    
    # ==================== Network Simulation & Gossip ====================
    
    def block_peer(self, peer_id: int):
        """Simulate a network partition blocking this peer."""
        with self.lock:
            self.simulated_unreachable_peers.add(peer_id)
            
    def unblock_peer(self, peer_id: int):
        """Remove simulation block for this peer."""
        with self.lock:
            self.simulated_unreachable_peers.discard(peer_id)

    def respond_to_ping(self) -> bool:
        """Called by other peers to check if I am reachable."""
        # In a real network, this would just happen. 
        # Here we simulate 'dropping packets' if the sender is blocked.
        # But wait, ping is called on the object reference.
        # The caller (sender) checks its own block list before calling, or we check here?
        # The logic: Sender checks if it CAN send. 
        # But 'ping' implies checking if the OTHER side is alive.
        # So it simply returns True (I am alive).
        # The connectivity check happens on the SENDER side based on 'simulated_unreachable_peers'.
        return True

    def _get_or_create_failure_detector(self, peer_id: int) -> PhiAccrualFailureDetector:
        """Get or create a failure detector for a peer."""
        with self.lock:
            if peer_id not in self.failure_detectors:
                self.failure_detectors[peer_id] = PhiAccrualFailureDetector(
                    threshold=self.phi_threshold,
                    max_sample_size=self.phi_window_size,
                    first_heartbeat_estimate_ms=self.heartbeat_interval * 1000
                )
            return self.failure_detectors[peer_id]

    def _calculate_sleep_with_jitter(self) -> float:
        """Calculate sleep time with jitter to prevent synchronized heartbeats."""
        # Base interval + uniform random jitter in range [-jitter, +jitter]
        jitter = random.uniform(-self.heartbeat_jitter, self.heartbeat_jitter)
        return max(0.1, self.heartbeat_interval + jitter)

    def heartbeat_loop(self):
        """
        Periodically ping peers and use Phi Accrual Failure Detector to determine availability.
        
        Uses a 3-second base interval with jitter to prevent thundering herd.
        Phi Accrual provides probabilistic failure detection based on heartbeat history.
        """
        while self.running:
            # Sleep with jitter to avoid synchronized heartbeats
            sleep_time = self._calculate_sleep_with_jitter()
            time.sleep(sleep_time)
            
            # Copy current state for comparison
            with self.lock:
                previous_active_snapshot = set(self.active_peers)
                
            peers_to_check = list(self.peers)
            current_active_snapshot = set()
            
            for peer in peers_to_check:
                try:
                    peer_id = peer.get_id()
                    
                    # Check simulated network conditions
                    is_blocked = False
                    with self.lock:
                        if peer_id in self.simulated_unreachable_peers:
                            is_blocked = True
                    
                    if is_blocked:
                        # Simulated partition: cannot reach peer
                        if peer_id in current_active_snapshot:
                            current_active_snapshot.discard(peer_id)
                            changes_detected = True
                        continue
                    
                    # Try to ping
                    is_reachable = False
                    if peer_id == self.id:
                        is_reachable = True
                    else:
                        if peer.ping():
                            is_reachable = True
                    
                    if is_reachable:
                        if peer_id not in current_active_snapshot:
                            current_active_snapshot.add(peer_id)
                            changes_detected = True
                    else:
                        # Ping failed (returned False?)
                        if peer_id in current_active_snapshot:
                            current_active_snapshot.discard(peer_id)
                            changes_detected = True
                                
                except Exception:
                    # Failed to connect or get_id failed
                    # We can't easily know WHICH peer_id failed if get_id failed, 
                    # but peer.get_id() failure means the peer object itself is pointing to something unreachable.
                    # Ideally we would remove it from active_peers if we knew the ID.
                    pass
                            

            
            # Update state if changed
            if changes_detected:
                with self.lock:
                    self.active_peers = current_active_snapshot
                # print(f"DEBUG: Server {self.id} detected network change. Active: {self.active_peers}")
                self._handle_network_change()

    def get_peer_phi(self, peer_id: int) -> float:
        """
        Get the current phi value for a peer.
        Useful for debugging and monitoring.
        
        Returns:
            Phi value (suspicion level). Higher = more likely dead.
        """
        with self.lock:
            if peer_id in self.failure_detectors:
                return self.failure_detectors[peer_id].phi()
        return 0.0

    def _handle_network_change(self):
        """React to changes in peer connectivity."""
        # 1. Elect new coordinator for this partition
        self.elect_partition_coordinator()
        
        # 2. If peers appeared (healing), trigger reconciliation
        # Note: This is a simplified check. Real systems might compare old/new sets.
        # We always attempt reconciliation on change just to be safe/consistent.
        self.on_partition_heal()

    def _is_peer_reachable(self, peer: Peer) -> bool:
        """Check if peer is considered active/reachable by heartbeat."""
        with self.lock:
            return peer.get_id() in self.active_peers
    
    def get_reachable_peers(self) -> List[Peer]:
        """Get list of peers reachable according to heartbeat."""
        with self.lock:
            return [p for p in self.peers if p.get_id() in self.active_peers]
    
    def get_unreachable_peers(self) -> List[Peer]:
        """Get list of peers NOT reachable according to heartbeat."""
        with self.lock:
            return [p for p in self.peers if p.get_id() not in self.active_peers]
    
    def _get_peer_by_id(self, peer_id: int) -> Optional[Peer]:
        """Get peer by ID."""
        with self.lock:
            for peer in self.peers:
                if peer.get_id() == peer_id:
                    return peer
        return None
    
    def elect_partition_coordinator(self):
        """
        Elect coordinator among currently active peers.
        Highest ID wins.
        """
        with self.lock:
            active = list(self.active_peers)
        
        if not active:
             # Should at least contain self
             return

        # Elect highest ID as coordinator
        new_coord_id = max(active)
        
        with self.lock:
            self.partition_coordinator_id = new_coord_id
            self.is_coordinator = (new_coord_id == self.id)
            # print(f"DEBUG: Server {self.id} election. Active: {active}, Winner: {new_coord_id}")
    
    def on_partition_heal(self):
        """Called when network topology changes (e.g. heal). Trigger reconciliation."""
        with self._reconcile_lock:
            if self._reconciling:
                return  # Already reconciling
            self._reconciling = True
        
        try:
            # 1. Deliver any stored hints
            self.deliver_hints()
            
            # 2. If I am coordinator, reconcile with other coordinators
            if self.is_coordinator:
                self._reconcile_with_other_coordinators()
        finally:
            with self._reconcile_lock:
                self._reconciling = False
    
    def _reconcile_with_other_coordinators(self):
        """Reconcile with all reachable peers to ensure data consistency."""
        # Optimization: Only reconcile with other potential coordinators (highest ID in their view)
        # But we don't know their view. So request reconciliation with reachable peers.
        # To avoid storm, maybe only reconcile with peers that have ID > self.id? 
        # Or just all. Let's stick to all reachable for robustness.
        for peer in self.get_reachable_peers():
            if peer.get_id() != self.id:
                try:
                    self.reconcile_with_peer(peer)
                except Exception as e:
                    print(f"Error reconciling with peer {peer.get_id()}: {e}")
    
    def deliver_hints(self):
        """Attempt to deliver stored hints to now-reachable nodes."""
        targets = self.hinted_handoff.get_all_targets()
        for target_id in targets:
            peer = self._get_peer_by_id(target_id)
            if peer and self._is_peer_reachable(peer):
                hints = self.hinted_handoff.get_hints_for(target_id)
                if hints:
                    try:
                        peer.receive(hints, 'handoff')
                    except Exception as e:
                        # Put hints back if delivery fails
                        self.hinted_handoff.store_hint(target_id, hints)
                        print(f"Failed to deliver hints to {target_id}: {e}")
    
    # ==================== Vector Versioning & Anti-Entropy ====================
    
    def get_vector_digest(self) -> Dict[int, Tuple[float, int]]:
        """
        Get digest of all vectors for anti-entropy sync.
        Returns dict of vector_id -> version tuple.
        Minimal data transfer: only IDs and versions.
        """
        result = {}
        for v in self.store.get_all():
            vec_id = v[1]
            version = v[4] if len(v) > 4 else (0, 0)
            result[vec_id] = version
        return result
    
    def get_vectors_by_ids(self, ids: List[int]) -> List:
        """Get specific vectors by ID for sync."""
        result = []
        for vec_id in ids:
            vec = self.store.get_vector(vec_id)
            if vec:
                result.append(vec)
        return result
    
    def reconcile_with_peer(self, peer: Peer):
        """
        Perform anti-entropy reconciliation with single peer.
        Only syncs vectors that SHOULD be on peer/self according to routing.
        This prevents over-replication beyond the intended replication factor.
        """
        try:
            # 1. Exchange digests (minimal network: just ID + version)
            my_digest = self.get_vector_digest()
            peer_digest = peer.get_vector_digest()
            
            # 2. Find vectors that should be on peer but aren't
            # Only send vectors that SHOULD be on peer according to routing
            my_ids = set(my_digest.keys())
            peer_ids = set(peer_digest.keys())
            
            to_send = []
            to_request = []
            
            # Check vectors I have that peer doesn't
            for vid in (my_ids - peer_ids):
                vec = self.store.get_vector(vid)
                if vec and self._should_be_on_peer(vec, peer.get_id()):
                    to_send.append(vid)
            
            # Check vectors peer has that I don't - request if they should be on me
            for vid in (peer_ids - my_ids):
                # Request vector from peer to check if it should be on us
                to_request.append(vid)
            
            # For common vectors, compare versions (only if routing matches)
            for vid in my_ids & peer_ids:
                if my_digest[vid] > peer_digest[vid]:
                    vec = self.store.get_vector(vid)
                    if vec and self._should_be_on_peer(vec, peer.get_id()):
                        to_send.append(vid)
            
            # 3. Send vectors that should be on peer
            if to_send:
                vectors_to_send = self.get_vectors_by_ids(to_send)
                peer.receive(vectors_to_send, 'reconcile')
            
            # 4. Request vectors from peer that we might need
            if to_request:
                requested_vectors = peer.get_vectors_by_ids(to_request)
                for vec in requested_vectors:
                    # Only store if routing says it should be on us
                    if self._should_be_on_me(vec):
                        self.store.insert(vec)
                    
        except Exception as e:
            print(f"Reconciliation with peer {peer.get_id()} failed: {e}")
    
    def _should_be_on_peer(self, vector, peer_id: int) -> bool:
        """
        Check if vector should be replicated to the given peer.
        Uses stored destinations if available, otherwise calculates.
        """
        # Check if vector has stored destinations (index 5)
        if len(vector) > 5 and vector[5]:
            return peer_id in vector[5]
        
        # No stored destinations - calculate based on routing
        if not self.clusters:
            return True  # Before clustering, accept everything
        
        # Calculate destinations
        destinations = self._calculate_destinations(vector)
        return peer_id in destinations
    
    def _should_be_on_me(self, vector) -> bool:
        """
        Check if vector should be stored on this server.
        Uses stored destinations if available, otherwise calculates.
        """
        return self._should_be_on_peer(vector, self.id)
    
    # ==================== Core Vector Operations ====================
    
    def similarity(self, vector):
        with self.lock:
            current_clusters = list(self.clusters)
        if not current_clusters:
            return 0.0
        return max([cosine_similarity(vector, cluster_center) for _, cluster_center in current_clusters])
    
    def route_vectors(self, vectors, top_k=3, use_all_peers=False):
        with self.lock:
            if use_all_peers:
                current_peers = list(self.peers)
            else:
                current_peers = self.get_reachable_peers()  # Only route to reachable peers
        
        if not current_peers:
            return {}
        
        results = {}
        for vector in vectors:
            peer_similarities = []
            for peer in current_peers:
                sim = peer.similarity(vector[0])
                peer_similarities.append((peer.get_id(), sim, vector))
            
            results[vector[1]] = sorted(peer_similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        # DEBUG: Print routing stats for first vector to see if similarities are all 0
        # if vectors:
        #     first_res = results[vectors[0][1]]
        #     print(f"DEBUG: Server {self.id} routing sample. Top sims: {[(x[0], x[1]) for x in first_res]}")
            
        return results
    
    def send_to_peers(self, vectors: ListOfVectorsComplete):
        """
        Send vectors to peers based on routing.
        Calculates and stores destinations in each vector on first routing,
        then uses stored destinations for subsequent operations to prevent over-replication.
        """
        with self.lock:
            all_peers = list(self.peers)
        
        peer_to_vec = {peer.get_id(): [] for peer in all_peers}
        
        for vector in vectors:
            # Check if vector already has destinations (index 5)
            if len(vector) > 5 and vector[5]:
                # Use stored destinations
                destinations = vector[5]
            else:
                # Calculate and attach destinations
                destinations = self._calculate_destinations(vector)
                # Create new vector with destinations attached
                if len(vector) == 5:
                    vector = (vector[0], vector[1], vector[2], vector[3], vector[4], destinations)
                elif len(vector) == 4:
                    vector = (vector[0], vector[1], vector[2], vector[3], (time.time(), self.id), destinations)
            
            # Route to intended destinations only
            for peer_id in destinations:
                if peer_id in peer_to_vec:
                    peer_to_vec[peer_id].append(vector)
        
        # Send to reachable peers, store hints for unreachable
        # print(f"DEBUG: Server {self.id} sending to peers. Distribution: {[len(v) for k,v in peer_to_vec.items() if v]}")
        for peer in all_peers:
            vecs = peer_to_vec[peer.get_id()]
            if not vecs:
                continue
            
            
            reachable = self._is_peer_reachable(peer)
            # print(f"DEBUG: Server {self.id} -> Peer {peer.get_id()} (reachable={reachable}): {len(vecs)} vectors")
            
            if reachable:
                try:
                    # print(f"DEBUG: Server {self.id} sending to {peer.get_id()} (Status {self.status})")
                    peer.receive(vecs, self.status)
                except Exception as e:
                    # On failure, store as hint
                    print(f"DEBUG: Server {self.id} exception sending to {peer.get_id()}: {e}")
                    self.hinted_handoff.store_hint(peer.get_id(), vecs)
            else:
                # Unreachable, store hint
                print(f"DEBUG: Server {self.id} cannot reach {peer.get_id()} (Active: {self.active_peers}), storing {len(vecs)} hints")
                self.hinted_handoff.store_hint(peer.get_id(), vecs)
    
    def _calculate_destinations(self, vector) -> frozenset:
        """Calculate intended destination peers for a vector based on similarity routing."""
        # Optimization: Use HNSW Index if available
        if self.cluster_index:
            # Find nearest cluster(s)
            # Route to the nearest cluster's responsible peers
            # If we want replication factor > 1 for *clusters*, we might search for top 1 cluster
            # and that cluster is already replicated to N peers.
            # OR we search for top M clusters?
            # Existing logic: "top_peers = sorted(peer_similarities ...)" routes to peers that are similar to the VECTOR.
            # But the vector should go to the peer that HOLDS the cluster it belongs to.
            # Wait, the current logic calculates similarity between VECTOR and PEER (which usually means max sim with peer's clusters).
            # Peer.similarity() does exactly that: max(cosine_sim(v, c) for c in peer.clusters).
            
            # So, technically, we want to find the nearest CLUSTER, and then see which peers have it.
            nearest_cluster_ids = self.cluster_index.find_nearest_clusters(vector[0], k=1)
            
            if nearest_cluster_ids and self.cluster_to_destinations_cache:
                best_cluster = nearest_cluster_ids[0]
                if best_cluster in self.cluster_to_destinations_cache:
                    return self.cluster_to_destinations_cache[best_cluster]
        
        # Fallback to linear search
        with self.lock:
            all_peers = list(self.peers)
        
        if not all_peers:
            return frozenset()
        
        peer_similarities = []
        for peer in all_peers:
            sim = peer.similarity(vector[0])
            peer_similarities.append((peer.get_id(), sim))
        
        # Sort by similarity and get top replication_factor
        top_peers = sorted(peer_similarities, key=lambda x: x[1], reverse=True)[:self.replication_factor]
        return frozenset(p[0] for p in top_peers)
    
    def receive(self, vectors: ListOfVectorsComplete, sender_status):
        # Enqueue the task
        # print(f"DEBUG: Server {self.id} enqueuing {len(vectors)} vectors with status {sender_status}")
        self.queue.put((self._handle_receive, (vectors, sender_status)))

    def _handle_receive(self, vectors: ListOfVectorsComplete, sender_status):
        # print(f"DEBUG: Server {self.id} handling receive, status={sender_status}, count={len(vectors)}")
        if sender_status == 'bootstrap':
            if not self.i_am_coord():
                print(f"Error: bootstrap sent to non coordinator node {self.id}")
            else:
                status = self.get_status()
                if status == 'bootstrap':
                    self.add_to_buffer(vectors)
                elif status == 'clustered':
                    self.send_to_peers(vectors)
                else:
                    print('error: status corrupted')
        elif sender_status == 'clustered':
            self.save_vectors(vectors)
        elif sender_status == 'client':
            status = self.get_status()
            # print(f"DEBUG: Server {self.id} RX from client. Status: {status}")
            if status == 'bootstrap':
                if self.i_am_coord():
                    self.add_to_buffer(vectors)
                else:
                    coord = self.coordinator()
                    if coord:
                        # print(f"DEBUG: Server {self.id} forwarding to coord {coord.get_id()}")
                        coord.receive(vectors, 'bootstrap')
                    else:
                        print(f"DEBUG: Server {self.id} could not find coordinator to forward to!")
                        self.dropped_vectors += len(vectors)
            elif status == 'clustered':
                self.send_to_peers(vectors)
            else:
                print('error: status corrupted')
        elif sender_status == 'handoff':
            # Hinted handoff delivery - save vectors directly
            self.save_vectors(vectors)
        elif sender_status == 'reconcile':
            # Anti-entropy reconciliation - save with version checking
            for v in vectors:
                self.store.insert(v)
        else:
            print('error: status corrupted')

    def save_vectors(self, vectors: ListOfVectorsComplete):
        # print(f"DEBUG: Server {self.id} saving {len(vectors)} vectors")
        # Use batch insert
        success_count = self.store.insert_batch(vectors)
        if success_count < len(vectors):
             pass # duplicates skipped defined behavior

    def receive_from_client(self, vectors: ListOfVectorsWithPayload):
        vectors_with_id = []
        for v, v_p in vectors:
            # Add version: (timestamp, node_id)
            version = (time.time(), self.id)
            vectors_with_id.append((v, self.get_new_vector_id(), v_p, -1, version))
        
        self.receive(vectors_with_id, 'client')
    
    def set_clusters(self, vectors_for_clusters, clusters: ListOfVectorsWithId):
        self.queue.put((self._handle_set_clusters, (vectors_for_clusters, clusters)))

    def _handle_set_clusters(self, vectors_for_clusters, clusters: ListOfVectorsWithId):
        with self.lock:
            self.clusters = clusters
            
            for cluster_info in self.clusters:
                cluster_id = cluster_info[0]
                if cluster_id in vectors_for_clusters:
                    # Batch insert all members
                    members = vectors_for_clusters[cluster_id]['members']
                    self.store.insert_batch(members)
            
            self.status = 'clustered'
            print(f"DEBUG: Server {self.id} transitioned to CLUSTERED status")
            
            vectors_to_send = []
            # Process any vectors that arrived during clustering
            if self.vector_buffer:
                print(f"DEBUG: Server {self.id}: Processing {len(self.vector_buffer)} buffered vectors after clustering")
                vectors_to_send = self.vector_buffer[:]
                self.vector_buffer = []
            
            
        if vectors_to_send:
            self.send_to_peers(vectors_to_send)
            
        with self.lock:
            self.clustering_in_progress = False

    def add_to_buffer(self, vectors: ListOfVectorsComplete):
        with self.lock:
            # print(f"DEBUG: Server {self.id} add_to_buffer {len(vectors)}")
            if not self.i_am_coord():
                print("error: not coordinator on add_to_buffer")
                return
            self.vector_buffer.extend(vectors)
            
            if self.status == 'bootstrap':
                if len(self.vector_buffer) >= self.before_clustering and not self.clustering_in_progress:
                    print(f"DEBUG: Server {self.id} starting CLUSTERING")
                    self.clustering_in_progress = True
                    v_b = self.vector_buffer[:]
                    self.vector_buffer = [] # Clear buffer that is being processed for clustering, new vectors will accumulate in buffer
                    
                    t = threading.Thread(target=self._run_clustering_background, args=(v_b,))
                    t.start()

    def _run_clustering_background(self, vectors):
        clusters = self.clustering(vectors)
        assignment = self.assign_clusters_to_peers(clusters)
        
        # Build cluster_id -> destinations mapping
        cluster_to_destinations = {}
        for peer_id, peer_clusters in assignment.items():
            for cluster_id, _ in peer_clusters:
                if cluster_id not in cluster_to_destinations:
                    cluster_to_destinations[cluster_id] = set()
                cluster_to_destinations[cluster_id].add(peer_id)
        
        # Add destinations to all vectors in clusters
        for cluster_id, cluster_data in clusters.items():
            destinations = frozenset(cluster_to_destinations.get(cluster_id, set()))
            updated_members = []
            for v in cluster_data['members']:
                # Add destinations (index 5)
                if len(v) >= 5:
                    v_with_dest = (v[0], v[1], v[2], v[3], v[4], destinations)
                else:
                    v_with_dest = (v[0], v[1], v[2], v[3], (time.time(), self.id), destinations)
                updated_members.append(v_with_dest)
            cluster_data['members'] = updated_members
        
        for peer in self.get_reachable_peers():
            peer.set_clusters(clusters, assignment[peer.get_id()])
            
        # Build HNSW Index
        print(f"DEBUG: Server {self.id} building HNSW index for {len(clusters)} clusters")
        # Ensure dimension is consistent. Use length of first center.
        if clusters:
            first_center = list(clusters.values())[0]['center']
            dim = len(first_center)
            self.cluster_index = ClusterIndex(dimension=dim)
            
            # shared types issue: clusters keys are ints, but we need list of (id, vector)
            cluster_list = [(k, v['center']) for k, v in clusters.items()]
            self.cluster_index.build(cluster_list)
            
            # Cache destinations
            self.cluster_to_destinations_cache = cluster_to_destinations
            print(f"DEBUG: Server {self.id} HNSW index built.")

    
    def clustering(self, vectors: ListOfVectorsComplete, num_clusters=10):
        vects = [v[0] for v in vectors]  # Extract just the vector data
        X = np.array(vects)
        n_clusters = max(1, min(num_clusters, X.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_.tolist()
        labels = kmeans.labels_
        members = [[] for _ in range(n_clusters)]
        
        for vec, lbl in zip(vectors, labels):
            # Create new tuple with updated cluster index
            # Handle both old 4-tuple and new 5-tuple format
            if len(vec) > 4:
                new_vec = (vec[0], vec[1], vec[2], int(lbl), vec[4])
            else:
                new_vec = (vec[0], vec[1], vec[2], int(lbl), (time.time(), self.id))
            members[lbl].append(new_vec)
        
        return {i: {'center': centers[i], 'members': members[i]} for i in range(n_clusters)}
    
    def assign_clusters_to_peers(self, clusters, beam_width=50):
        reachable_peers = self.get_reachable_peers()
        all_nodes = [peer.get_id() for peer in reachable_peers]
        
        if not all_nodes:
            return {}
        
        # Adjust replication factor if fewer peers than factor
        effective_rep = min(self.replication_factor, len(all_nodes))
        all_combos = list(itertools.combinations(all_nodes, effective_rep))
        
        beam = [(0.0, [], {id: 0.0 for id in all_nodes})]
        clusters_sorted = sorted(
            [(id, len(el['members']), el['center']) for id, el in clusters.items()],
            key=lambda x: x[1], reverse=True
        )
        
        for _, cluster_load, _ in clusters_sorted:
            potential_states = []
            for _, current_assignment, current_node_loads in beam:
                for combo in all_combos:
                    new_node_loads = dict(current_node_loads)
                    for node_idx in combo:
                        new_node_loads[node_idx] += cluster_load
                    new_assignment = current_assignment + [combo]
                    partial_score = sum(load**2 for _, load in new_node_loads.items())
                    potential_states.append(
                        (partial_score, new_assignment, new_node_loads)
                    )
            beam = heapq.nsmallest(beam_width, potential_states, key=lambda x: x[0])
        
        _, best_assignment, _ = beam[0]
        assignment = {node_id: [] for node_id in all_nodes}
        
        for index, nodes_tuple in enumerate(best_assignment):
            for node in nodes_tuple:
                assignment[node].append((clusters_sorted[index][0], clusters_sorted[index][2]))
        
        return assignment

    def get_all_vectors(self):
        print(f"DEBUG: get_all_vectors called on Server {self.id}")
        return self.store.get_all()
    
    def search_vectors(self, vectors: ListOfVectorsWithId, top_k=100, top_look=4):
        peers_similarity_per_vector = self.route_vectors([(v, v_id) for (v_id, v) in vectors], top_look)
        reachable_peers = self.get_reachable_peers()
        peer_to_vec = {peer.get_id(): [] for peer in reachable_peers}
        
        for _, p_data in peers_similarity_per_vector.items():
            for (p_id, _, v) in p_data:
                if p_id in peer_to_vec:
                    peer_to_vec[p_id].append(v)
        
        found_vectors = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for peer in reachable_peers:
                if peer_to_vec[peer.get_id()]:
                    futures.append(executor.submit(peer.search_vectors_local, peer_to_vec[peer.get_id()], top_k))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    found_vectors.extend(future.result())
                except Exception as e:
                    print(f"Error searching on peer: {e}")
        
        # Deduplicate results based on vector ID (index 1)
        unique_results = {}
        for res in found_vectors:
            v_id = res[1]
            if v_id not in unique_results:
                unique_results[v_id] = res
        
        # Sort by similarity (index 3) in descending order
        final_results = sorted(unique_results.values(), key=lambda x: x[3], reverse=True)
        
        return final_results[:top_k]

    def search_vectors_local(self, vectors, top_k=5):
        if self.qdrant_url:
             # Use Qdrant search
             query_vectors_only = [v[0] for v in vectors]
             # qdrant_module.query_vectors takes (url, collection, query_list, topk)
             results = qdrant_module.query_vectors(self.qdrant_url, self.store.collection_name, query_vectors_only, top_k)
             # Map Qdrant results to internal format: (Vector, VectorId, VectorPayload, Similarity)
             mapped = []
             # qdrant_module returns list of dicts: {'id', 'score', 'payload': {'string', 'vector'}}
             # EXCEPT: query_vectors handles BATCH query.
             # qdrant_module.query_vectors -> query_batch_points -> returns list of lists? 
             # Wait, qdrant_module.query_vectors docstring says: "Returns a list of lists of ScoredPoint objects."
             # BUT implementation says:
             # final_results = [] (flat list)
             # for response in results: for point in response.points: append
             # It flattens the results?? This is WRONG for batch query if we want to distinguish results per query vector.
             # But search_vectors_local is expected to return a single list of results (top k overall? or per vector?)
             # The existing implementation:
             # "found = [] ... for i in range(len(vectors)): found.append(...)"
             # It seems to flatten everything into one list of results?
             # Yes: "return found".
             # So flattening is actually desired behavior for this specific method signature in existing server.py?
             # Let's verify existing implementation logic.
             # It returns 'found' which accumulates top-k for EACH query vector.
             # So if I send 2 query vectors, I get top-k for vec1 AND top-k for vec2 in one list.
             # qdrant_module.query_vectors (as I modified) DOES flatten.
             # So it matches!
             
             for res in results:
                 v_data = res['payload']['vector']
                 p_str = res['payload']['string']
                 mapped.append((v_data, res['id'], p_str, res['score']))
             return mapped

        current_vectors_data = self.store.get_all()
        if not current_vectors_data:
            return []
            
        # Database Matrix (N, D)
        # All stored vectors are guaranteed to be numpy arrays due to insert() logic
        db_vectors = [v[0] for v in current_vectors_data]
        
        # Stack into matrix
        try:
            matrix = np.stack(db_vectors)
        except Exception as e:
            print(f"Error creating matrix from vectors: {e}")
            return []
        
        # Precompute norms for database vectors (N,)
        norms_matrix = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        norms_matrix[norms_matrix == 0] = 1e-10
        
        # Query Matrix (M, D)
        query_vectors = [v[0] for v in vectors]
        
        # Convert queries to numpy if needed
        # Queries usually come from client as lists, or internal as arrays
        try:
            if query_vectors and not isinstance(query_vectors[0], np.ndarray):
                 query_matrix = np.array(query_vectors, dtype=np.float32)
            else:
                 query_matrix = np.stack(query_vectors)
        except Exception as e:
             print(f"Error creating query matrix: {e}")
             return []
             
        # Compute norms for queries (M,)
        norms_query = np.linalg.norm(query_matrix, axis=1)
        norms_query[norms_query == 0] = 1e-10
        
        # Dot product: (M, D) @ (D, N) -> (M, N)
        # transpose matrix to (D, N)
        dists = np.dot(query_matrix, matrix.T)
        
        # Similarities: dists / (norm_q[:, None] * norm_m[None, :])
        # Broadcasting: (M, 1) * (1, N) -> (M, N)
        sims = dists / (norms_query[:, np.newaxis] * norms_matrix)
        
        found = []
        
        # For each query
        for i in range(len(vectors)):
            query_sims = sims[i] # (N,)
            
            # Top K
            # We want descending order
            if len(query_sims) <= top_k:
                top_indices = np.argsort(query_sims)[::-1]
            else:
                # argpartition puts top k at the end
                top_indices = np.argpartition(query_sims, -top_k)[-top_k:]
                # Sort the top k
                top_indices = top_indices[np.argsort(query_sims[top_indices])[::-1]]
            
            for idx in top_indices:
                 # Reconstruct result tuple: (Vector, VectorId, VectorPayload, Similarity)
                 entry = current_vectors_data[idx]
                 found.append((entry[0], entry[1], entry[2], float(query_sims[idx])))
                 
        return found

    def query(self, vectors: ListOfVectorsWithId, sender_status, top_k=100, top_look=4):
        if sender_status == 'client':
            status = self.get_status()
            if status == 'bootstrap':
                if self.i_am_coord():
                    with self.lock:
                        buffer_snap = list(self.vector_buffer)
                    return self._search_in_array(vectors, buffer_snap, top_k)
                else:
                    coord = self.coordinator()
                    if coord:
                        return coord.query(vectors, 'bootstrap')
                    return []
            elif status == 'clustered':
                return self.search_vectors(vectors, top_k, top_look)
            else:
                print('error: invalid sender status')
        elif sender_status == 'bootstrap':
            if self.i_am_coord():
                status = self.get_status()
                if status == 'bootstrap':
                    with self.lock:
                        buffer_snap = list(self.vector_buffer)
                    return self._search_in_array(vectors, buffer_snap, top_k)
                else:
                    return self.search_vectors(vectors, top_k, top_look)
            else:
                print('error: bootstrap sended to non coordinator node')
        elif sender_status == 'clustered':
            return self.search_vectors(vectors, top_k, top_look)
        else:
            print('error: invalid sender status')
            
    def _search_in_array(self, vectors, array, top_k):
        found = []
        for _, vector_to_query in vectors:
            found.extend(sorted(
                [(v, v_id, v_payload, cosine_similarity(v, vector_to_query)) 
                 for v, v_id, v_payload, *rest in array],
                key=lambda x: x[3], reverse=True
            )[:top_k])
        return found
    
    def query_from_client(self, vectors: ListOfVectors, top_k=100):
        vectors_with_qid = [(self.get_id(), v) for v in vectors]
        return self.query(vectors_with_qid, 'client', top_k=top_k)

def main():
    parser = argparse.ArgumentParser(description="Distributed Vector Store Server")
    parser.add_argument("--id", type=int, required=True, help="Server ID")
    parser.add_argument("--intra-port", type=int, required=True, help="Internal port for server-to-server communication")
    parser.add_argument("--inter-port", type=int, default=None, help="External port for client communication (optional)")
    parser.add_argument("--qdrant", type=str, default=None, help="Qdrant URL")
    parser.add_argument("--endpoint", type=str, default="QUIC", help="Endpoint type (QUIC, HTTP, GRPC)")
    parser.add_argument("--coordinator", action="store_true", help="Is this node the coordinator?")
    parser.add_argument("--peers", type=str, default=None, help="Coordinator address (ip:port) to join")
    
    # These params are hardcoded in startup.py logic or client.py, but server needs them
    # Server init: before_clustering, replication_factor
    parser.add_argument("--before-clustering", type=int, default=8192, help="Vectors before clustering")
    parser.add_argument("--replication-factor", type=int, default=3, help="Replication factor")

    args = parser.parse_args()
    
    # startup.py passes '--intra-port' but Server init takes 'port' for internal comms
    server = Server(
        id=args.id, 
        is_coordinator=args.coordinator, 
        before_clustering=args.before_clustering, 
        replication_factor=args.replication_factor, 
        port=args.intra_port, 
        endpoint=args.endpoint,
        qdrant_url=args.qdrant,
        client_port=args.inter_port
    )
    
    print(f"Server {args.id} started. Intra: {args.intra_port}, Inter: {args.inter_port}, Coord: {args.coordinator}")
    
    if args.peers and not args.coordinator:
        # Args.peers comes as "ip:port" of coordinator
        try:
            p_ip, p_port = args.peers.split(":")
            server.add_peer(p_ip, int(p_port))
            print(f"Added peer {args.peers}")
        except Exception as e:
            print(f"Error parsing peer address {args.peers}: {e}")
            
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopping...")
        server.stop()

if __name__ == "__main__":
    main()
