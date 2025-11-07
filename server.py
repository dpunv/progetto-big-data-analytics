from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import requests
import uvicorn
from dataclasses import dataclass, asdict
import uuid
import json
import numpy as np
import utils  # Only for cosine_similarity
import msgpack
from collections import defaultdict
from urllib.parse import urlparse
import socket
import threading
import time
import random
from embeddings import EmbeddingService
from meta_hnsw import MetaHNSW
import pickle
import base64
import hashlib
from node_assignment import compute_node_assignments

# Pydantic models for request/response validation
class VectorDataModel(BaseModel):
    id: str
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None

class SendVectorRequest(BaseModel):
    from_node: str
    vector_data: VectorDataModel

class SearchRequest(BaseModel):
    from_node: str
    query_vector: List[float]
    top_k: int = 5

class BroadcastRequest(BaseModel):
    vector_data: VectorDataModel

class QueryPeerRequest(BaseModel):
    peer_id: str
    query_vector: List[float]
    top_k: int = 5

class SyncRequest(BaseModel):
    vector_id: str
    target_node_id: str

class RegisterPeerRequest(BaseModel):
    peer_id: str
    peer_url: str
    node_vectors: List[List[float]] # Peer's representative vectors (multiple)

# --- NEW PYDANTIC MODELS FOR BULK ---
class SendVectorsBulkRequest(BaseModel):
    from_node: str
    vectors_data: List[VectorDataModel]
# --- END NEW PYDANTIC MODELS ---

@dataclass
class VectorData:
    """Represents a vector with metadata"""
    id: str
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None
    
    # NEW: Method to create a deep copy
    def clone(self) -> 'VectorData':
        """Create a deep copy of this VectorData instance."""
        import copy
        return VectorData(
            id=self.id,
            vector=self.vector.copy(),  # Clone the vector list
            payload=copy.deepcopy(self.payload) if self.payload else None  # Deep copy payload
        )

class QdrantNodeWrapper:
    """
    Wrapper for a Qdrant vector database node with inter-node communication capabilities.
    """

    def __init__(self, node_id: str, qdrant_host: str = "localhost", qdrant_port: int = 6335, collection_name: str = "vectors", self_url: str = "http://localhost:8000", vector_size: int = 384):
        self.node_id = node_id
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self.collection_name = collection_name
        self.peer_nodes = {}
        self.self_url = self_url
        self.vector_size = vector_size
        self.node_vectors: List[List[float]] = []
        self.peer_node_vectors: Dict[str, List[List[float]]] = {}
        self.embedding_service: Optional[object] = None 
        
        # --- NEW: per-peer health status ---
        self.peer_status: Dict[str, Dict[str, Any]] = {}
        self._health_thread: Optional[threading.Thread] = None
        self._health_thread_stop = False
        self._health_monitor_params = {
            "interval": 5,
            "failure_threshold": 3,
            "recovery_interval": 30
        }
        
        # NEW: Local Meta-HNSW instance (peer-to-peer)
        self.meta_hnsw: Optional[MetaHNSW] = None
        self._last_gossip_time = 0.0
        self.gossip_interval = 30  # seconds between gossip rounds
        self._gossip_thread: Optional[threading.Thread] = None
        self._gossip_thread_stop = False
        self.gossip_enabled = True  # NEW: Flag to enable/disable gossip
        
        # Track local cluster updates for gossip
        self.local_cluster_updates: Dict[int, Dict] = {}  # {cluster_id: {centroid, count, timestamp}}

        # NEW: Fixed coordinator logic
        self.is_fixed_coordinator = (node_id == "node1")  # Only node1 can be coordinator
        self.bootstrap_vectors_received = 0
        self.bootstrap_threshold = 10000
        self.clustering_complete = False
        self.clustering_in_progress = False
        self._clustering_lock = threading.Lock()
        
        # NEW: Configuration versioning and freeze state
        self.config_version = "v0_bootstrap"  # Current config version
        self.is_frozen = False  # Freeze state during critical operations
        self._config_lock = threading.Lock()
        
        # Track pending operations during freeze
        self._pending_inserts: List[Dict] = []
        self._pending_inserts_lock = threading.Lock()
        
        # NEW: Cleanup coordination
        self.cleanup_ready = False  # Ready for cleanup phase
        self.cleanup_complete = False  # Cleanup finished
        
    def register_peer(self, peer_id: str, peer_url: str):
        self.peer_nodes[peer_id] = peer_url
        # Initialize peer status
        if peer_id not in self.peer_status:
            self.peer_status[peer_id] = {
                "url": peer_url,
                "status": "UNKNOWN",
                "fail_count": 0,
                "last_ok": None,
                "last_check": None
            }
        else:
            self.peer_status[peer_id]["url"] = peer_url
        print(f"Node {self.node_id}: Registered peer {peer_id} at {peer_url}")

    def set_node_vectors(self, vectors: List[List[float]]):
        """
        Set multiple representative vectors for this node. Each vector is validated and normalized.
        """
        normalized = []
        for v in vectors:
            if len(v) != self.vector_size:
                raise ValueError(
                    f"Node {self.node_id}: Vector size mismatch. "
                    f"Expected {self.vector_size}, got {len(v)}"
                )
            arr = np.array(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            normalized.append(arr.tolist())
        self.node_vectors = normalized
        if self.node_vectors:
            print(f"Node {self.node_id}: Set {len(self.node_vectors)} representative vectors (first 3 dims of first): {self.node_vectors[0][:3]}...")

    def find_best_nodes(self, vector: List[float]) -> List[str]:
        """
        Finds all replica nodes for the cluster closest to the given vector.
       	"""
        if not self.node_vectors:
            print(f"Node {self.node_id}: Warning: This node has no representative vectors. Defaulting to self.")
            return [self.node_id]

        # 1. Combine all known centroids from self and peers
        all_centroids = {self.node_id: self.node_vectors}
        all_centroids.update(self.peer_node_vectors)

        # 2. Find the single closest centroid vector
        best_similarity = -2.0
        best_centroid=None

        for node_id, centroids in all_centroids.items():
            for centroid in centroids:
                sim = utils.cosine_similarity(vector, centroid)
                if sim > best_similarity:
                    best_similarity = sim
                    best_centroid = centroid
                    
        if best_centroid is None:
            print(f"Node {self.node_id}: Warning: No centroids found. Defaulting to self.")
            return [self.node_id]

        # 3. Find all nodes that host this best_centroid (replicas)
        replica_nodes = []
        # Convert to numpy array once for efficient comparison
        best_centroid_np = np.array(best_centroid, dtype=np.float32) 

        # Collect all centroids that match the best one (replicas)
        for node_id, centroids in all_centroids.items():
            for centroid in centroids:
            	# Use np.allclose for robust float vector comparison
                if np.allclose(np.array(centroid, dtype=np.float32), best_centroid_np):
                    replica_nodes.append(node_id)
                    break # This node is a replica, move to the next node
        
        if not replica_nodes:
            # Should not happen if best_centroid was found, but as a safe fallback
            print(f"Node {self.node_id}: Warning: Could not find node for best centroid. Defaulting to self.")
            return [self.node_id]
        return replica_nodes 

    def send_vector(self, target_node_id: str, vector_data: VectorData) -> bool:
        # Check peer status before attempting send
        status = self.peer_status.get(target_node_id, {}).get("status")
        if status == "DOWN":
            print(f"Node {self.node_id}: Skipping send to {target_node_id} because it is DOWN")
            return False

        if target_node_id not in self.peer_nodes:
            print(f"Node {self.node_id}: Unknown peer {target_node_id}")
            return False
        
        peer_url = self.peer_nodes[target_node_id]
        
        try:
            payload = {
                "from_node": self.node_id,
                "vector_data": {
                    "id": vector_data.id,
                    "vector": vector_data.vector,
                    "payload": vector_data.payload or {}
                }
            }
            response = requests.post(
                f"{peer_url}/receive_vector",
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                print(f"Node {self.node_id}: Successfully sent/forwarded vector {vector_data.id} to {target_node_id}")
                return True
            else:
                print(f"Node {self.node_id}: Failed to send vector to {target_node_id}. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error sending vector to {target_node_id}: {e}")
            return False

    # --- NEW METHOD FOR BULK FORWARDING ---
    def send_vectors_bulk(self, target_node_id: str, vectors_data: List[VectorData]) -> bool:
        # Check peer status before attempting send
        status = self.peer_status.get(target_node_id, {}).get("status")
        if status == "DOWN":
            print(f"Node {self.node_id}: Skipping bulk send to {target_node_id} because it is DOWN")
            return False

        if target_node_id not in self.peer_nodes:
            print(f"Node {self.node_id}: Unknown peer {target_node_id}")
            return False
        
        peer_url = self.peer_nodes[target_node_id]
        
        try:
            # Convert List[VectorData] to List[Dict] for serialization
            vectors_data_dicts = [asdict(vd) for vd in vectors_data]

            payload = {
                "from_node": self.node_id,
                "vectors_data": vectors_data_dicts
            }
            
            # NEW: Serialize with MessagePack for peer-to-peer forwarding
            binary_data = msgpack.packb(payload, use_bin_type=True)
            
            response = requests.post(
                f"{peer_url}/receive_vectors_bulk",
                data=binary_data,  # Send raw bytes
                headers={"Content-Type": "application/msgpack"},
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"Node {self.node_id}: Successfully sent/forwarded {len(vectors_data)} vectors to {target_node_id}")
                return True
            else:
                print(f"Node {self.node_id}: Failed to send bulk vectors to {target_node_id}. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error sending bulk vectors to {target_node_id}: {e}")
            return False
    # --- END NEW METHOD ---

    def receive_vector(self, from_node_id: str, vector_data: VectorData) -> bool:
        """
        Receive and store a SINGLE vector from another node (or external client).
        """
        try:
            point = {
                "id": vector_data.id,
                "vector": vector_data.vector,
                "payload": {
                    **(vector_data.payload or {}),
                    "received_from": from_node_id,
                    "storage_node": self.node_id
                }
            }
            
            response = requests.put(
                f"{self.qdrant_url}/collections/{self.collection_name}/points",
                params={"wait": "true"}, # Ensure operation completes
                json={"points": [point]},
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"Node {self.node_id}: Stored vector {vector_data.id} from {from_node_id}")
                current_count = self.count_local_vectors()
                if current_count != -1:
                    print(f"Node {self.node_id}: ✨ Local vector count: {current_count}")
                return True
            else:
                print(f"Node {self.node_id}: Failed to store vector. Status: {response.status_code} {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error storing vector: {e}")
            return False

    # --- NEW METHOD FOR BULK RECEIVING ---
    def receive_vectors_bulk(self, from_node_id: str, vectors_data: List[VectorData]) -> bool:
        """
        Receive and store a BATCH of vectors from another node (or external client).
        NOW: Checks config version and freeze state before accepting.
        """
        # NEW: Check freeze state
        with self._config_lock:
            if self.is_frozen:
                print(f"Node {self.node_id}: ⏸️  FROZEN - Queueing {len(vectors_data)} vectors from {from_node_id}")
                with self._pending_inserts_lock:
                    self._pending_inserts.append({
                        'from_node_id': from_node_id,
                        'vectors_data': vectors_data,
                        'timestamp': time.time()
                    })
                return True  # Queued successfully
        
        try:
            points = []
            for vector_data in vectors_data:
                point = {
                    "id": vector_data.id,
                    "vector": vector_data.vector,
                    "payload": {
                        **(vector_data.payload or {}),
                        "received_from": from_node_id,
                        "storage_node": self.node_id,
                        "config_version": self.config_version  # NEW: Tag with version
                    }
                }
                points.append(point)
            
            if not points:
                return True
            
            response = requests.put(
                f"{self.qdrant_url}/collections/{self.collection_name}/points",
                params={"wait": "true"},
                json={"points": points},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                print(f"Node {self.node_id}: Stored {len(points)} vectors from {from_node_id} (version: {self.config_version})")
                current_count = self.count_local_vectors()
                if current_count != -1:
                    print(f"Node {self.node_id}: ✨ Local vector count: {current_count}")
                return True
            else:
                print(f"Node {self.node_id}: Failed to store bulk vectors. Status: {response.status_code} {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error storing bulk vectors: {e}")
            return False
    # --- END NEW METHOD ---

    def broadcast_vector(self, vector_data: VectorData) -> Dict[str, bool]:
        results = {}
        for peer_id in self.peer_nodes:
            results[peer_id] = self.send_vector(peer_id, vector_data)
        print(f"Node {self.node_id}: Broadcast complete. Success: {sum(results.values())}/{len(results)}")
        return results
    
    def query_peer(self, peer_id: str, query_vector: List[float], 
                   top_k: int = 5) -> Optional[List[Dict]]:
        if peer_id not in self.peer_nodes:
            print(f"Node {self.node_id}: Unknown peer {peer_id}")
            return None
        peer_url = self.peer_nodes[peer_id]
        try:
            payload = {
                "from_node": self.node_id,
                "query_vector": query_vector,
                "top_k": top_k
            }
            response = requests.post(
                f"{peer_url}/search",
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                results = response.json()
                print(f"Node {self.node_id}: Received {len(results)} results from {peer_id}")
                return results
            else:
                print(f"Node {self.node_id}: Query failed. Status: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error querying peer: {e}")
            return None
    
    def search_local(self, query_vector: List[float], top_k: int = 5) -> Optional[List[Dict]]:
        try:
            payload = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
                "with_vector": True
            }
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                results = response.json().get("result", [])
                print(f"Node {self.node_id}: Found {len(results)} local results")
                return results
            else:
                print(f"Node {self.node_id}: Local search failed. Status: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error in local search: {e}")
            return None

    def count_local_vectors(self) -> int:
        try:
            payload = {"exact": True}
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/count",
                json=payload,
                timeout=5
            )
            if response.status_code == 200:
                count = response.json().get("result", {}).get("count", 0)
                return count
            else:
                print(f"Node {self.node_id}: Error counting vectors. Status: {response.status_code} - {response.text}")
                return -1
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error counting vectors: {e}")
            return -1
            
    def federated_search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, List[Dict]]:
        all_results = {}
        local_results = self.search_local(query_vector, top_k)
        if local_results:
            all_results[self.node_id] = local_results
        
        for peer_id in self.peer_nodes:
            peer_results = self.query_peer(peer_id, query_vector, top_k)
            if peer_results:
                all_results[peer_id] = peer_results
        
        print(f"Node {self.node_id}: Federated search complete across {len(all_results)} nodes")
        return all_results
    
    def sync_vector(self, vector_id: str, target_node_id: str) -> bool:
        try:
            response = requests.get(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/{vector_id}",
                timeout=10
            )
            if response.status_code != 200:
                print(f"Node {self.node_id}: Vector {vector_id} not found locally")
                return False
            
            point = response.json()["result"]
            vector_data = VectorData(
                id=vector_id,
                vector=point["vector"],
                payload=point.get("payload", {})
            )
            return self.send_vector(target_node_id, vector_data)
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error syncing vector: {e}")
            return False
    
    def create_collection(self, vector_size: int, distance: str = "Cosine"):
        try:
            response = requests.get(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                timeout=10
            )
            if response.status_code == 200:
                print(f"Node {self.node_id}: Collection '{self.collection_name}' already exists")
                return True
            
            payload = {
                "vectors": {
                    "size": vector_size,
                    "distance": distance
                }
            }
            response = requests.put(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                json=payload,
                timeout=10
            )
            if response.status_code in [200, 201]:
                print(f"Node {self.node_id}: Created collection '{self.collection_name}'")
                return True
            else:
                print(f"Node {self.node_id}: Failed to create collection. Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error creating collection: {e}")
            return False

    # --- NEW METHODS FOR HEALTH MONITORING ---
    def start_health_monitor(self, interval: Optional[int] = None, failure_threshold: Optional[int] = None, recovery_interval: Optional[int] = None):
        """Start a background thread that periodically pings peers and updates self.peer_status."""
        if interval is not None:
            self._health_monitor_params["interval"] = interval
        if failure_threshold is not None:
            self._health_monitor_params["failure_threshold"] = failure_threshold
        if recovery_interval is not None:
            self._health_monitor_params["recovery_interval"] = recovery_interval

        if self._health_thread and self._health_thread.is_alive():
            return

        self._health_thread_stop = False
        self._health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self._health_thread.start()
        print(f"Node {self.node_id}: Health monitor started (interval={self._health_monitor_params['interval']}s)")

    def stop_health_monitor(self):
        self._health_thread_stop = True
        if self._health_thread:
            self._health_thread.join(timeout=2)
        print(f"Node {self.node_id}: Health monitor stopped")

    def _health_monitor_loop(self):
        """Loop executed in a background thread that pings peer / (root) endpoints and updates statuses."""
        interval = self._health_monitor_params["interval"]
        failure_threshold = self._health_monitor_params["failure_threshold"]
        recovery_interval = self._health_monitor_params["recovery_interval"]

        while not self._health_thread_stop:
            now = time.time()
            for peer_id, peer_url in list(self.peer_nodes.items()):
                status_info = self.peer_status.get(peer_id, {
                    "url": peer_url, "status": "UNKNOWN", "fail_count": 0, "last_ok": None, "last_check": None
                })

                # If DOWN, ping less frequently
                last_check = status_info.get("last_check")
                if status_info["status"] == "DOWN":
                    if last_check and (now - last_check) < recovery_interval:
                        continue

                try:
                    status_info["last_check"] = now
                    resp = requests.get(f"{peer_url}/", timeout=2)
                    if resp.status_code == 200:
                        status_info["fail_count"] = 0
                        status_info["last_ok"] = now
                        # CHANGED: Only log on state transition
                        if status_info["status"] != "UP":
                            status_info["status"] = "UP"
                            print(f"Node {self.node_id}: Peer {peer_id} is UP")
                    else:
                        status_info["fail_count"] = status_info.get("fail_count", 0) + 1
                        # CHANGED: Only log when crossing threshold
                        if status_info["fail_count"] >= failure_threshold and status_info.get("status") != "DOWN":
                            status_info["status"] = "DOWN"
                            print(f"Node {self.node_id}: Peer {peer_id} marked DOWN (non-200 responses)")
                except requests.exceptions.RequestException:
                    status_info["fail_count"] = status_info.get("fail_count", 0) + 1
                    # CHANGED: Only log when crossing threshold
                    if status_info["fail_count"] >= failure_threshold and status_info.get("status") != "DOWN":
                        status_info["status"] = "DOWN"
                        print(f"Node {self.node_id}: Peer {peer_id} marked DOWN (connect failures)")

                self.peer_status[peer_id] = status_info

            # Sleep in small steps for responsive stop
            sleep_total = 0
            step = 1
            while sleep_total < interval and not self._health_thread_stop:
                time.sleep(step)
                sleep_total += step

    # --- END NEW METHODS ---

    def initialize_meta_hnsw(self, dimension: int, max_clusters: int = 500):
        """Initialize local Meta-HNSW instance."""
        print(f"Node {self.node_id}: Initializing local Meta-HNSW (dimension={dimension}, max_clusters={max_clusters})")
        self.meta_hnsw = MetaHNSW(
            dimension=dimension,
            max_clusters=max_clusters,
            ef_construction=200,
            M=16
        )

    def add_local_clusters(self, cluster_vectors: List[List[float]]):
        """Add this node's clusters to local Meta-HNSW."""
        if self.meta_hnsw is None:
            raise ValueError(f"Node {self.node_id}: Meta-HNSW not initialized")
        
        self.meta_hnsw.add_node_clusters(self.node_id, cluster_vectors)
        print(f"Node {self.node_id}: Added {len(cluster_vectors)} local clusters to Meta-HNSW")

    def receive_peer_clusters(self, peer_id: str, cluster_vectors: List[List[float]]):
        """Receive and store peer's cluster updates."""
        if self.meta_hnsw is None:
            print(f"Node {self.node_id}: WARNING - Meta-HNSW not initialized, cannot receive peer clusters")
            return False
        
        try:
            self.meta_hnsw.add_node_clusters(peer_id, cluster_vectors)
            print(f"Node {self.node_id}: Updated clusters for {peer_id} ({len(cluster_vectors)} clusters)")
            return True
        except Exception as e:
            print(f"Node {self.node_id}: Error receiving peer clusters from {peer_id}: {e}")
            return False

    def start_gossip_protocol(self):
        """Start background gossip thread."""
        if self._gossip_thread and self._gossip_thread.is_alive():
            return
        
        self._gossip_thread_stop = False
        self._gossip_thread = threading.Thread(target=self._gossip_loop, daemon=True)
        self._gossip_thread.start()
        print(f"Node {self.node_id}: Gossip protocol started (interval={self.gossip_interval}s)")

    def stop_gossip_protocol(self):
        """Stop gossip thread."""
        self._gossip_thread_stop = True
        if self._gossip_thread:
            self._gossip_thread.join(timeout=2)
        print(f"Node {self.node_id}: Gossip protocol stopped")

    def _gossip_loop(self):
        """Gossip loop: periodically share cluster updates with peers."""
        while not self._gossip_thread_stop:
            now = time.time()
            
            # NEW: Skip gossip if disabled (after clustering complete)
            if self.gossip_enabled and now - self._last_gossip_time >= self.gossip_interval:
                self._perform_gossip_round()
                self._last_gossip_time = now
            
            # Sleep in small steps for responsive stop
            sleep_total = 0
            step = 1
            while sleep_total < self.gossip_interval and not self._gossip_thread_stop:
                time.sleep(step)
                sleep_total += step

    def _perform_gossip_round(self):
        """Execute one gossip round: share local clusters with random peers."""
        if self.meta_hnsw is None or not self.peer_nodes:
            return
        
        # Get local cluster centroids
        local_cluster_ids = self.meta_hnsw.node_to_clusters.get(self.node_id, [])
        if not local_cluster_ids:
            return
        
        local_clusters = [self.meta_hnsw.cluster_centroids[cid].tolist() for cid in local_cluster_ids]
        
        # Select random subset of peers (gossip to ~50% of peers)
        num_targets = max(1, len(self.peer_nodes) // 2)
        target_peers = random.sample(list(self.peer_nodes.items()), min(num_targets, len(self.peer_nodes)))
        
        for peer_id, peer_url in target_peers:
            status = self.peer_status.get(peer_id, {}).get("status")
            if status == "DOWN":
                continue
            
            try:
                payload = {
                    "from_node": self.node_id,
                    "cluster_vectors": local_clusters
                }
                
                response = requests.post(
                    f"{peer_url}/gossip/clusters",
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    # Optionally: receive peer's clusters in response
                    peer_data = response.json()
                    peer_clusters = peer_data.get("cluster_vectors", [])
                    if peer_clusters:
                        self.receive_peer_clusters(peer_id, peer_clusters)
                        
            except requests.exceptions.RequestException as e:
                # Gossip failure is not critical, just skip
                pass

    def find_best_nodes_local(self, vector: List[float], k_nodes: int = 3) -> List[str]:
        """
        Use local Meta-HNSW to find best nodes (P2P routing).
        Falls back to centroid-based routing if Meta-HNSW unavailable.
        """
        if self.meta_hnsw is None:
            # Fallback to original centroid-based routing
            return self.find_best_nodes(vector)
        
        try:
            # Use local Meta-HNSW for routing
            query_vec = np.array(vector, dtype=np.float32)
            nearest_nodes = self.meta_hnsw.find_nearest_nodes(query_vec, k_nodes=k_nodes)
            
            # Extract node names
            node_names = [node_name for node_name, _ in nearest_nodes]
            return node_names
            
        except Exception as e:
            print(f"Node {self.node_id}: Meta-HNSW query failed: {e}, using fallback")
            return self.find_best_nodes(vector)

    def check_coordinator_trigger(self):
        """Check if THIS node (if coordinator) should trigger clustering."""
        if not self.is_fixed_coordinator:
            # Non-coordinator nodes just wait
            return
        
        with self._clustering_lock:
            if self.clustering_in_progress or self.clustering_complete:
                return
            
            if self.bootstrap_vectors_received >= self.bootstrap_threshold:
                print(f"\n{'='*60}")
                print(f"Node {self.node_id}: COORDINATOR TRIGGERING CLUSTERING")
                print(f"   Reached {self.bootstrap_vectors_received} bootstrap vectors")
                print(f"{'='*60}\n")
                
                self.clustering_in_progress = True
                
                # Spawn clustering thread
                threading.Thread(
                    target=self._perform_clustering_and_distribute,
                    daemon=False
                ).start()

    def _perform_clustering_and_distribute(self):
        """
        Coordinator-only: Perform clustering and distribute assignments to peers.
        NOW: Uses three-phase commit with handshakes for safe cleanup.
        """
        try:
            print(f"Node {self.node_id}: Starting server-side clustering...")
            
            # Generate config version ID
            config_version = f"v1_clustered_{int(time.time())}"
            
            # STEP 1: Fetch all local vectors WITH IDs
            vectors_with_ids = self._fetch_all_local_vectors_with_ids()
            if not vectors_with_ids or len(vectors_with_ids) < 1000:
                print(f"Node {self.node_id}: ERROR - Insufficient vectors")
                self.clustering_in_progress = False
                return
            
            # Separate IDs and vectors
            vector_ids = [item['id'] for item in vectors_with_ids]
            vectors = [item['vector'] for item in vectors_with_ids]
            
            print(f"Node {self.node_id}: Fetched {len(vectors)} vectors for clustering")
            
            # STEP 2: Save vectors temporarily for node_assignment.py
            temp_embeddings_file = 'temp_bootstrap_embeddings.json'
            print(f"Node {self.node_id}: Preparing data for clustering...")
            
            # Convert to format expected by node_assignment.py
            embeddings_data = [{'embedding': vec} for vec in vectors]
            with open(temp_embeddings_file, 'w') as f:
                json.dump(embeddings_data, f)
            
            # STEP 3: Run clustering from node_assignment.py (MANDATORY)
            num_nodes = len(self.peer_nodes) + 1
            
            print(f"Node {self.node_id}: Running K-means + Beam Search (FAISS-optimized)...")
            
            assignment_data = compute_node_assignments(
                embeddings_file=temp_embeddings_file,
                num_nodes=num_nodes,
                max_k_to_test=min(30, len(vectors) // 100),
                random_state=42,
                max_vectors=len(vectors),
                rep_factor=None,
                beam_width=5,
                use_simulated_annealing=False
            )
            
            # Cleanup temp file
            import os
            if os.path.exists(temp_embeddings_file):
                os.remove(temp_embeddings_file)
            
            # STEP 4: Validate clustering result
            if assignment_data is None:
                print(f"Node {self.node_id}: ❌ Clustering returned None")
                self.clustering_in_progress = False
                return
            
            if 'centroids' not in assignment_data or 'node_assignments' not in assignment_data:
                print(f"Node {self.node_id}: ❌ Invalid clustering result format: {list(assignment_data.keys())}")
                self.clustering_in_progress = False
                return
            
            centroids = assignment_data['centroids']
            node_assignments = assignment_data['node_assignments']
            
            print(f"Node {self.node_id}: ✓ Clustering complete:")
            print(f"   - {len(centroids)} clusters found")
            
            # STEP 5: Compute per-vector cluster assignments
            print(f"Node {self.node_id}: Computing per-vector cluster assignments...")
            vector_to_cluster = self._assign_vectors_to_clusters(vectors, centroids)
            
            # Create mapping: vector_id → cluster_idx
            vector_labels = {}
            for i, vector_id in enumerate(vector_ids):
                vector_labels[vector_id] = int(vector_to_cluster[i])
            
            print(f"Node {self.node_id}: Created labels for {len(vector_labels)} vectors")
            
            # DEBUG: Print sample assignments
            print(f"Node {self.node_id}: Sample vector labels:")
            for vid in list(vector_labels.keys())[:5]:
                print(f"  {vid}: cluster {vector_labels[vid]}")
            
            print(f"Node {self.node_id}: Sample node assignments:")
            for node_idx in range(min(3, len(node_assignments))):
                clusters = node_assignments.get(str(node_idx), node_assignments.get(node_idx, []))
                print(f"  node{node_idx + 1}: {len(clusters)} clusters = {clusters[:5]}...")
            
            # STEP 6: Distribute to ALL nodes (including labels)
            distribution_payload = {
                "config_version": config_version,  # NEW
                "centroids": centroids,
                "node_assignments": node_assignments,
                "vector_size": self.vector_size,
                "coordinator_id": self.node_id,
                "vector_labels": vector_labels,
                "stats": assignment_data['stats']  # Include optimization stats
            }
            
            # Send to self
            self._apply_clustering_config(distribution_payload)
            
            # Send to peers
            for peer_id, peer_url in self.peer_nodes.items():
                try:
                    response = requests.post(
                        f"{peer_url}/receive-clustering",
                        json=distribution_payload,
                        timeout=30
                    )
                    if response.status_code == 200:
                        print(f"  ✓ {peer_id}: Clustering config applied")
                    else:
                        print(f"  ✗ {peer_id}: Failed (HTTP {response.status_code})")
                except requests.exceptions.RequestException as e:
                    print(f"  ✗ {peer_id}: Error - {e}")
            
            # NEW: PHASE 1 - Distribute config (WITHOUT cleanup)
            print(f"\n{'='*60}")
            print(f"PHASE 1: DISTRIBUTING CONFIG (No cleanup yet)")
            print(f"   Version: {config_version}")
            print(f"{'='*60}")
            
            distribution_payload = {
                "config_version": config_version,
                "centroids": centroids,
                "node_assignments": node_assignments,
                "vector_size": self.vector_size,
                "coordinator_id": self.node_id,
                "vector_labels": vector_labels,
                "stats": assignment_data['stats'],
                "skip_cleanup": True  # NEW: Don't cleanup yet
            }
            
            # Apply to self (without cleanup)
            self._apply_clustering_config(distribution_payload)
            
            # Send to peers
            config_success = []
            for peer_id, peer_url in self.peer_nodes.items():
                try:
                    response = requests.post(
                        f"{peer_url}/receive-clustering",
                        json=distribution_payload,
                        timeout=30
                    )
                    if response.status_code == 200:
                        config_success.append(peer_id)
                        print(f"  ✓ {peer_id}: Config applied (cleanup deferred)")
                    else:
                        print(f"  ✗ {peer_id}: Failed (HTTP {response.status_code})")
                except requests.exceptions.RequestException as e:
                    print(f"  ✗ {peer_id}: Error - {e}")
            
            # NEW: PHASE 2 - Wait for all nodes to signal "ready for cleanup"
            print(f"\n{'='*60}")
            print(f"PHASE 2: WAITING FOR CLEANUP READINESS")
            print(f"{'='*60}")
            
            # Mark self as ready
            self.cleanup_ready = True
            
            # Poll all peers until they're ready
            max_wait = 60  # 1 minute timeout
            start_time = time.time()
            all_ready = False
            
            while time.time() - start_time < max_wait:
                ready_nodes = [self.node_id]  # Self is ready
                
                for peer_id, peer_url in self.peer_nodes.items():
                    try:
                        response = requests.get(f"{peer_url}/cleanup-status", timeout=3)
                        if response.status_code == 200:
                            status = response.json()
                            if status.get("cleanup_ready"):
                                ready_nodes.append(peer_id)
                    except:
                        pass
                
                if len(ready_nodes) == num_nodes:
                    all_ready = True
                    print(f"  ✅ All {num_nodes} nodes ready for cleanup")
                    break
                
                elapsed = int(time.time() - start_time)
                if elapsed % 5 == 0:
                    print(f"  [{elapsed}s] Ready: {len(ready_nodes)}/{num_nodes} nodes")
                
                time.sleep(1)
            
            if not all_ready:
                print(f"  ⚠️  Timeout waiting for all nodes to be ready")
                print(f"     Proceeding anyway (some cleanup may be incomplete)")
            
            # NEW: PHASE 3 - Trigger coordinated cleanup
            print(f"\n{'='*60}")
            print(f"PHASE 3: COORDINATED CLEANUP")
            print(f"{'='*60}")
            
            cleanup_payload = {
                "config_version": config_version,
                "action": "cleanup",
                "node_assignments": node_assignments,
                "vector_labels": vector_labels
            }
            
            # Cleanup self
            print(f"  🗑️  {self.node_id}: Starting cleanup...")
            self._execute_cleanup(node_assignments, vector_labels)
            self.cleanup_complete = True
            
            # Trigger cleanup on peers
            for peer_id, peer_url in self.peer_nodes.items():
                try:
                    response = requests.post(
                        f"{peer_url}/execute-cleanup",
                        json=cleanup_payload,
                        timeout=60
                    )
                    if response.status_code == 200:
                        print(f"  ✓ {peer_id}: Cleanup complete")
                    else:
                        print(f"  ✗ {peer_id}: Cleanup failed (HTTP {response.status_code})")
                except requests.exceptions.RequestException as e:
                    print(f"  ✗ {peer_id}: Error - {e}")
            
            # NEW: PHASE 4 - Verify cleanup completion
            print(f"\n{'='*60}")
            print(f"PHASE 4: VERIFYING CLEANUP COMPLETION")
            print(f"{'='*60}")
            
            time.sleep(5)  # Grace period for cleanup to finish
            
            for peer_id, peer_url in self.peer_nodes.items():
                try:
                    response = requests.get(f"{peer_url}/cleanup-status", timeout=3)
                    if response.status_code == 200:
                        status = response.json()
                        if status.get("cleanup_complete"):
                            print(f"  ✅ {peer_id}: Cleanup verified")
                        else:
                            print(f"  ⚠️  {peer_id}: Cleanup not complete")
                except:
                    print(f"  ⚠️  {peer_id}: Cannot verify")
            
            print(f"\n✅ Clustering and cleanup complete (Version: {config_version})")
            
            self.is_coordinator = False
            self.clustering_complete = True
            self.clustering_in_progress = False
            
        except Exception as e:
            print(f"Node {self.node_id}: ❌ Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            self.clustering_in_progress = False

    def _fetch_all_local_vectors_with_ids(self) -> List[Dict[str, Any]]:
        """
        Fetch all vectors WITH their IDs from local Qdrant.
        
        Returns:
            List of {'id': str, 'vector': List[float]}
        """
        try:
            offset = None
            all_vectors = []
            
            while True:
                payload = {
                    "limit": 1000,
                    "with_vector": True,
                    "with_payload": False
                }
                if offset:
                    payload["offset"] = offset
                
                response = requests.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/scroll",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"Node {self.node_id}: Error scrolling points")
                    break
                
                data = response.json()
                points = data.get("result", {}).get("points", [])
                
                if not points:
                    break
                
                for point in points:
                    vector = point.get("vector")
                    vector_id = point.get("id")
                    if vector and vector_id:
                        all_vectors.append({
                            'id': str(vector_id),
                            'vector': vector
                        })
                
                offset = data.get("result", {}).get("next_page_offset")
                if not offset:
                    break
            
            return all_vectors
            
        except Exception as e:
            print(f"Node {self.node_id}: Error fetching vectors with IDs: {e}")
            return []

    def _assign_vectors_to_clusters(self, vectors: List[List[float]], 
                                    centroids: List[List[float]]) -> List[int]:
        """
        Assign each vector to its nearest cluster (K-means assignment step).
        
        Args:
            vectors: List of vector embeddings
            centroids: List of cluster centroids
        
        Returns:
            List of cluster indices (one per vector)
        """
        vectors_np = np.array(vectors, dtype=np.float32)
        centroids_np = np.array(centroids, dtype=np.float32)
        
        # For each vector, find nearest centroid
        labels = []
        for vec in vectors_np:
            # Calculate cosine similarity with all centroids
            similarities = []
            for centroid in centroids_np:
                sim = np.dot(vec, centroid) / (
                    np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8
                )
                similarities.append(sim)
            
            # Assign to cluster with highest similarity
            best_cluster = int(np.argmax(similarities))
            labels.append(best_cluster)
        
        return labels

    def _apply_clustering_config(self, config: Dict[str, Any]):
        """
        Apply clustering configuration received from coordinator.
        NOW: Separates config application from cleanup.
        """
        config_version = config.get('config_version', 'unknown')
        centroids = config['centroids']
        node_assignments = config['node_assignments']
        vector_labels = config.get('vector_labels', {})
        skip_cleanup = config.get('skip_cleanup', False)
        
        print(f"Node {self.node_id}: Applying clustering config (Version: {config_version})")
        
        # Determine this node's index
        node_idx = int(self.node_id.replace("node", "")) - 1
        
        cluster_indices = node_assignments.get(str(node_idx), node_assignments.get(node_idx, []))
        my_centroids = [centroids[idx] for idx in cluster_indices]
        my_cluster_indices_set = set(cluster_indices)
        
        # Set node vectors
        self.set_node_vectors(my_centroids)
        
        # Update peer vectors
        num_nodes = len(self.peer_nodes) + 1
        for peer_idx in range(num_nodes):
            peer_node_idx = peer_idx
            if peer_node_idx == node_idx:
                continue
            
            peer_id = f"node{peer_node_idx + 1}"
            peer_cluster_indices = node_assignments.get(str(peer_node_idx), [])
            peer_centroids = [centroids[idx] for idx in peer_cluster_indices]
            
            if peer_id in self.peer_nodes:
                self.peer_node_vectors[peer_id] = peer_centroids
        
        # Initialize Meta-HNSW
        self.initialize_meta_hnsw(dimension=self.vector_size, max_clusters=len(centroids))
        
        for node_idx_str, cluster_indices in node_assignments.items():
            peer_node_id = f"node{int(node_idx_str) + 1}"
            cluster_vectors = [centroids[idx] for idx in cluster_indices]
            
            if peer_node_id == self.node_id:
                self.add_local_clusters(cluster_vectors)
            else:
                self.receive_peer_clusters(peer_id, cluster_vectors)
        
        self.meta_hnsw.force_rebuild()
        
        # NEW: Disable gossip since Meta-HNSW is now fully populated
        self.gossip_enabled = False
        print(f"Node {self.node_id}: 🔒 Gossip protocol disabled (Meta-HNSW finalized)")
        
        # Update config version
        with self._config_lock:
            self.config_version = config_version
        
        # NEW: Skip cleanup if instructed (Phase 1)
        if skip_cleanup:
            print(f"Node {self.node_id}: ✅ Config applied (cleanup deferred)")
            self.cleanup_ready = True
        else:
            # OLD PATH: Immediate cleanup (deprecated)
            print(f"Node {self.node_id}: 🗑️  Cleaning up bootstrap vectors...")
            deleted_count = self._cleanup_bootstrap_vectors_by_labels(
                my_cluster_indices_set,
                node_assignments,
                vector_labels
            )
            print(f"Node {self.node_id}: ✅ Config applied with cleanup")
            self.cleanup_complete = True
        
        self.clustering_complete = True

    def _execute_cleanup(self, node_assignments: Dict[str, List[int]], 
                        vector_labels: Dict[str, int]):
        """
        Execute cleanup phase (separate from config application).
        """
        node_idx = int(self.node_id.replace("node", "")) - 1
        cluster_indices = node_assignments.get(str(node_idx), node_assignments.get(node_idx, []))
        my_cluster_indices_set = set(cluster_indices)
        
        print(f"Node {self.node_id}: 🗑️  Starting coordinated cleanup...")
        deleted_count = self._cleanup_bootstrap_vectors_by_labels(
            my_cluster_indices_set,
            node_assignments,
            vector_labels
        )
        
        self.cleanup_complete = True
        print(f"Node {self.node_id}: ✅ Cleanup complete ({deleted_count} vectors deleted)")

    def _cleanup_bootstrap_vectors_by_labels(self, 
                                            my_cluster_indices: set,
                                            all_node_assignments: Dict[str, List[int]],
                                            vector_labels: Dict[str, int]) -> int:
        """
        Remove bootstrap vectors using EXPLICIT cluster labels from coordinator.
        
        LOGICA CORRETTA (FIXED):
        - Per ogni vettore, controlla il suo cluster
        - Trova QUALI nodi hanno quel cluster (dalle assignments beam search)
        - MANTIENE solo se QUESTO nodo è tra le repliche del cluster
        - ELIMINA se questo nodo NON è una replica del cluster
        
        Args:
            my_cluster_indices: Set di cluster assegnati a QUESTO nodo
            all_node_assignments: Dict completo {node_idx: [cluster_ids]} da beam search
            vector_labels: Mapping vector_id → cluster_idx
        
        RISULTATO ATTESO:
        - Bootstrap totali: 10.000 vettori × 3 repliche = 30.000
        - Routed:           10.000 vettori × 3 repliche = 30.000
        - TOTALE:           60.000 vettori
        """
        if not my_cluster_indices:
            print(f"Node {self.node_id}: No clusters assigned, skipping cleanup")
            return 0
        
        if not vector_labels:
            print(f"Node {self.node_id}: No labels provided, skipping cleanup")
            return 0
        
        try:
            # STEP 0: Build cluster → nodes mapping from beam search
            cluster_to_replica_nodes = {}
            for node_idx_str, cluster_list in all_node_assignments.items():
                node_name = f"node{int(node_idx_str) + 1}"
                for cluster_id in cluster_list:
                    if cluster_id not in cluster_to_replica_nodes:
                        cluster_to_replica_nodes[cluster_id] = []
                    cluster_to_replica_nodes[cluster_id].append(node_name)
            
            print(f"Node {self.node_id}: Using BEAM SEARCH replica assignments for cleanup...")
            print(f"  My assigned clusters: {sorted(my_cluster_indices)}")
            
            # DEBUG: Print replica info for my clusters
            print(f"  My cluster replicas (from beam search):")
            for cluster_id in sorted(my_cluster_indices):
                replicas = cluster_to_replica_nodes.get(cluster_id, [])
                print(f"    Cluster {cluster_id}: replicas on {replicas}")
            
            total_deleted = 0
            total_checked = 0
            vectors_kept = 0
            vectors_to_delete = []
            
            # STEP 1: Scroll and check each vector
            offset = None
            while True:
                payload = {
                    "limit": 1000,
                    "with_vector": False,
                    "with_payload": False
                }
                if offset:
                    payload["offset"] = offset
                
                response = requests.post(
                    f"{self.qdrant_url}/collections/{self.collection_name}/points/scroll",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"Node {self.node_id}: Error scrolling points")
                    break
                
                data = response.json()
                points = data.get("result", {}).get("points", [])
                
                if not points:
                    break
                
                # STEP 2: Check each vector using BEAM SEARCH replica logic
                for point in points:
                    total_checked += 1
                    vector_id = str(point.get("id"))
                    
                    # Look up cluster assignment from coordinator
                    assigned_cluster = vector_labels.get(vector_id)
                    
                    if assigned_cluster is None:
                        # Vector not in coordinator's dataset (routed vector, keep)
                        vectors_kept += 1
                        if total_checked <= 5:
                            print(f"  ℹ️  KEEP {vector_id[:16]}... (not in labels, likely routed)")
                        continue
                    
                    # FIXED: Check if THIS node is a replica for this cluster
                    replica_nodes = cluster_to_replica_nodes.get(assigned_cluster, [])
                    
                    if self.node_id in replica_nodes:
                        # KEEP: This node is a designated replica for this cluster
                        vectors_kept += 1
                        if total_checked <= 5:
                            print(f"  ✅ KEEP {vector_id[:16]}... (cluster {assigned_cluster}, I'm replica in {replica_nodes})")
                    else:
                        # DELETE: This node is NOT a replica for this cluster
                        vectors_to_delete.append(vector_id)
                        if total_checked <= 5:
                            print(f"  ❌ DELETE {vector_id[:16]}... (cluster {assigned_cluster}, replicas are {replica_nodes}, not me)")
                
                offset = data.get("result", {}).get("next_page_offset")
                if not offset:
                    break
            
            # STEP 3: Delete non-replica vectors
            if vectors_to_delete:
                print(f"Node {self.node_id}: Deleting {len(vectors_to_delete)} non-replica bootstrap vectors...")
                
                batch_size = 100
                for i in range(0, len(vectors_to_delete), batch_size):
                    batch = vectors_to_delete[i:i+batch_size]
                    
                    try:
                        delete_payload = {"points": batch}
                        
                        response = requests.post(
                            f"{self.qdrant_url}/collections/{self.collection_name}/points/delete",
                            json=delete_payload,
                            timeout=30
                        )
                        
                        if response.status_code in [200, 201]:
                            total_deleted += len(batch)
                        else:
                            print(f"  ✗ Failed to delete batch: {response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        print(f"  ✗ Error deleting batch: {e}")
            
            print(f"\nNode {self.node_id}: Cleanup complete (BEAM SEARCH REPLICA-AWARE):")
            print(f"  - Checked: {total_checked} vectors")
            print(f"  - Kept: {vectors_kept} (I'm designated replica)")
            print(f"  - Deleted: {total_deleted} (I'm NOT replica)")
            print(f"  ℹ️  Expected: ~1,000 kept per node (10k vectors / 10 nodes)")
            
            # Print new count
            new_count = self.count_local_vectors()
            if new_count != -1:
                print(f"Node {self.node_id}: ✨ New vector count: {new_count}")
            
            return total_deleted
            
        except Exception as e:
            print(f"Node {self.node_id}: Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _enter_freeze_state(self, config_version: str):
        """Enter freeze state to prevent concurrent modifications."""
        with self._config_lock:
            self.is_frozen = True
            print(f"Node {self.node_id}: ❄️  FROZEN (Version: {config_version})")
    
    def _exit_freeze_state(self):
        """Exit freeze state and process pending operations."""
        with self._config_lock:
            self.is_frozen = False
            print(f"Node {self.node_id}: 🔓 UNFROZEN")
        
        # Process pending inserts
        with self._pending_inserts_lock:
            if self._pending_inserts:
                print(f"Node {self.node_id}: Processing {len(self._pending_inserts)} pending inserts...")
                for pending in self._pending_inserts:
                    self.receive_vectors_bulk(
                        pending['from_node_id'],
                        pending['vectors_data']
                    )
                self._pending_inserts.clear()

# FastAPI Application
def create_app(node: QdrantNodeWrapper) -> FastAPI:
    """Create FastAPI application for a Qdrant node"""
    
    app = FastAPI(
        title=f"Qdrant Node API - {node.node_id}",
        description="Distributed Qdrant vector database node API",
        version="1.0.0"
    )
    
    # Allow CORS from the visualizer (running on localhost:8088) so the browser can
    # fetch /get-topology (and other endpoints) while developing. For production,
    # tighten this list to the real allowed origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8088", "http://127.0.0.1:8088"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def _startup_event():
        # Start per-node health monitor in background thread
        node.start_health_monitor()
        # NEW: Start gossip protocol
        node.start_gossip_protocol()
    
    @app.on_event("shutdown")
    async def _shutdown_event():
        node.stop_gossip_protocol()
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
        # CHANGED: Return minimal response (no logging by default)
        return {
            "status": "healthy",
            "node_id": node.node_id,
            "peers": list(node.peer_nodes.keys()),
            "node_vectors_set": len(node.node_vectors) > 0
        }
    
    @app.post("/receive_vector")
    async def receive_vector_endpoint(request: SendVectorRequest):
        """
        Endpoint to receive a SINGLE vector from other nodes
        """
        vector_data = VectorData(
            id=request.vector_data.id,
            vector=request.vector_data.vector,
            payload=request.vector_data.payload
        )
        success = node.receive_vector(request.from_node, vector_data)
        if success:
            return {
                "status": "success",
                "message": f"Vector {vector_data.id} stored successfully",
                "node_id": node.node_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store vector")

    # --- NEW ENDPOINT for receiving bulk forwards ---
    @app.post("/receive_vectors_bulk")
    async def receive_vectors_bulk_endpoint(request: Request):
        """
        Endpoint to receive a BATCH of vectors from other nodes.
        Now supports both JSON (legacy) and MessagePack (preferred).
        """
        content_type = request.headers.get("Content-Type", "application/json")
        
        try:
            if content_type == "application/msgpack":
                # NEW: Deserialize MessagePack
                body = await request.body()
                payload = msgpack.unpackb(body, raw=False)
                
                from_node = payload.get("from_node")
                vectors_data_raw = payload.get("vectors_data", [])
            else:
                # Legacy: JSON support (for backwards compatibility)
                payload = await request.json()
                from_node = payload.get("from_node")
                vectors_data_raw = payload.get("vectors_data", [])
            
            # Convert to VectorData dataclass
            vectors_data_list = [
                VectorData(
                    id=vd['id'],
                    vector=vd['vector'],
                    payload=vd.get('payload', {})
                ) for vd in vectors_data_raw
            ]
            
            success = node.receive_vectors_bulk(from_node, vectors_data_list)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Stored {len(vectors_data_list)} vectors",
                    "node_id": node.node_id
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to store bulk vectors")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")

    @app.post("/send_vector/{target_node_id}")
    async def send_vector_endpoint(target_node_id: str, vector_data: VectorDataModel):
        """
        Send a vector to a specific peer node
        """
        vec_data = VectorData(
            id=vector_data.id,
            vector=vector_data.vector,
            payload=vector_data.payload
        )
        success = node.send_vector(target_node_id, vec_data)
        if success:
            return {
                "status": "success",
                "message": f"Vector sent to {target_node_id}",
                "vector_id": vector_data.id
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Failed to send vector to {target_node_id}"
            )
    
    @app.post("/broadcast")
    async def broadcast_vector_endpoint(request: BroadcastRequest):
        """
        Store a vector locally AND broadcast it to all peer nodes
        """
        vector_data = VectorData(
            id=request.vector_data.id,
            vector=request.vector_data.vector,
            payload=request.vector_data.payload
        )
        
        print(f"Node {node.node_id}: Storing broadcast vector {vector_data.id} locally...")
        local_success = node.receive_vector(node.node_id, vector_data)
        if not local_success:
            print(f"Node {node.node_id}: WARNING - Failed to store broadcast vector locally.")

        broadcast_results = node.broadcast_vector(vector_data)
        return {
            "status": "success",
            "vector_id": vector_data.id,
            "local_storage": "success" if local_success else "failed",
            "broadcast_results": broadcast_results,
            "success_count": sum(broadcast_results.values()),
            "total_peers": len(broadcast_results)
        }

    @app.post("/add_vector")
    async def add_vector_endpoint(vector_data: VectorDataModel):
        """
        Add a new SINGLE vector from an external client.
        This node will find the best cluster and route the vector
        to ALL replica nodes for that cluster.
        """
        vec_data = VectorData(
            id=vector_data.id,
            vector=vector_data.vector,
            payload=vector_data.payload
        )
        
        # 1. Find all replica nodes for this vector
        replica_node_ids = node.find_best_nodes(vec_data.vector)
        
        print(f"Node {node.node_id}: Routing vector {vec_data.id} to {len(replica_node_ids)} replicas: {replica_node_ids}")

        success_nodes = []
        failed_nodes = []

        # 2. Send to all replicas
        for node_id in replica_node_ids:
            success = False
            if node_id == node.node_id:
                # Store locally
                print(f"Node {node.node_id}: Storing vector {vec_data.id} locally (replica).")
                success = node.receive_vector(
                    from_node_id="external_client_routed", 
                    vector_data=vec_data
                )
            else:
                # Forward to peer
                print(f"Node {node.node_id}: Forwarding vector {vec_data.id} to replica {node_id}.")
                success = node.send_vector(node_id, vec_data)
            
            if success:
                success_nodes.append(node_id)
            else:
                failed_nodes.append(node_id)

        # 3. Report result
        if not success_nodes:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to store vector {vector_data.id} on any replica."
            )

        return {
            "status": "success",
            "message": f"Vector {vector_data.id} routed to {len(replica_node_ids)} replicas.",
            "action": "routed_to_replicas",
            "replicas_targeted": replica_node_ids,
            "replicas_succeeded": success_nodes,
            "replicas_failed": failed_nodes
        }

    @app.post("/add_vectors_bulk")
    async def add_vectors_bulk_endpoint(request: Request, background_tasks: BackgroundTasks):
        """
        Add a new BATCH of vectors from an external client.
        Now supports both JSON (legacy) and MessagePack (preferred).
        """
        content_type = request.headers.get("Content-Type", "application/json")
        
        try:
            if content_type == "application/msgpack":
                # NEW: Deserialize MessagePack
                body = await request.body()
                vectors_raw = msgpack.unpackb(body, raw=False)
            else:
                # Legacy: JSON support
                vectors_raw = await request.json()
            
            # Convert to VectorDataModel (Pydantic validation)
            vectors = [
                VectorDataModel(
                    id=v['id'],
                    vector=v['vector'],
                    payload=v.get('payload', {})
                ) for v in vectors_raw
            ]
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
        
        # Check if these are bootstrap vectors
        is_bootstrap = any(
            v.payload and v.payload.get('source_type') == 'bootstrap' 
            for v in vectors
        )
        
        if is_bootstrap and not node.clustering_complete:
            # BOOTSTRAP MODE: Store locally
            print(f"Node {node.node_id}: Received {len(vectors)} bootstrap vectors")
            
            vectors_data = [
                VectorData(id=v.id, vector=v.vector, payload=v.payload)
                for v in vectors
            ]
            
            # Store locally in background
            background_tasks.add_task(
                node.receive_vectors_bulk,
                from_node_id="bootstrap",
                vectors_data=vectors_data
            )
            
            # Update counter and check coordinator trigger
            node.bootstrap_vectors_received += len(vectors)
            background_tasks.add_task(node.check_coordinator_trigger)
            
            return {
                "status": "bootstrap",
                "message": f"Stored {len(vectors)} bootstrap vectors",
                "bootstrap_count": node.bootstrap_vectors_received
            }
        
        # NEW: Check if clustering is complete before accepting routed vectors
        if not node.clustering_complete:
            print(f"Node {node.node_id}: ⏳ Clustering not complete yet, rejecting {len(vectors)} vectors")
            raise HTTPException(
                status_code=503,  # Service Unavailable
                detail="Node is still initializing (clustering in progress). Please retry in a few seconds."
            )
        
        else:
            # ROUTING MODE: Use smart routing (FIXED LOGIC)
            nodes_to_vectors: Dict[str, List[VectorData]] = defaultdict(list)
            
            total_vectors = len(vectors)
            total_routings = 0
            
            # FIXED: For each vector, replicate to ALL replica nodes
            for vd_model in vectors:
                vec_data = VectorData(
                    id=vd_model.id,
                    vector=vd_model.vector,
                    payload=vd_model.payload
                )
                
                # Find ALL replica nodes for THIS vector
                replica_node_ids = node.find_best_nodes(vec_data.vector)
                
                # CRITICAL FIX: Clone vec_data for EACH replica to avoid shared references
                for replica_node_id in replica_node_ids:
                    # Create independent copy for this replica
                    vec_data_clone = vec_data.clone()
                    nodes_to_vectors[replica_node_id].append(vec_data_clone)
                    total_routings += 1
            
            routing_summary = {}
            
            for target_node_id, vectors_list in nodes_to_vectors.items():
                batch_size = len(vectors_list)
                routing_summary[target_node_id] = batch_size
                
                if target_node_id == node.node_id:
                    # Store locally
                    background_tasks.add_task(
                        node.receive_vectors_bulk,
                        from_node_id="external_client_routed",
                        vectors_data=vectors_list
                    )
                else:
                    # Forward to peer
                    background_tasks.add_task(
                        node.send_vectors_bulk,
                        target_node_id=target_node_id,
                        vectors_data=vectors_list
                    )
            
            # FIXED: Return proper stats showing replication
            return {
                "status": "processing_bulk",
                "message": f"Processing {total_vectors} vectors. Total routings: {total_routings}",
                "batches": routing_summary  # This now shows actual replica counts per node
            }

    @app.post("/receive-clustering")
    async def receive_clustering_endpoint(payload: Dict[str, Any]):
        """
        Receive clustering configuration from coordinator.
        """
        coordinator_id = payload.get("coordinator_id")
        print(f"Node {node.node_id}: Received clustering config from coordinator {coordinator_id}")
        
        node._apply_clustering_config(payload)
        
        return {
            "status": "success",
            "message": "Clustering config applied",
            "node_id": node.node_id
        }

    @app.get("/clustering-status")
    async def clustering_status_endpoint():
        """Check clustering status."""
        return {
            "node_id": node.node_id,
            "is_fixed_coordinator": node.is_fixed_coordinator,
            "clustering_complete": node.clustering_complete,
            "clustering_in_progress": node.clustering_in_progress,
            "bootstrap_vectors_received": node.bootstrap_vectors_received
        }

    @app.post("/search")
    async def search_endpoint(request: SearchRequest):
        """
        Search local Qdrant database (called by peer nodes)
        """
        results = node.search_local(request.query_vector, request.top_k)
        if results is not None:
            return results
        else:
            raise HTTPException(status_code=500, detail="Search failed")
    
    @app.post("/search/local")
    async def search_local_endpoint(query_vector: List[float], top_k: int = 5):
        """
        Search only the local database
        """
        results = node.search_local(query_vector, top_k)
        if results is not None:
            return {
                "node_id": node.node_id,
                "results": results,
                "count": len(results)
            }
        else:
            raise HTTPException(status_code=500, detail="Local search failed")
    
    @app.post("/search/federated")
    async def federated_search_endpoint(query_vector: List[float], top_k: int = 5):
        """
        Search across all nodes (local + peers)
        """
        results = node.federated_search(query_vector, top_k)
        total_results = sum(len(r) for r in results.values())
        return {
            "status": "success",
            "results": results,
            "nodes_searched": len(results),
            "total_results": total_results
        }
    
    @app.post("/query_peer")
    async def query_peer_endpoint(request: QueryPeerRequest):
        """
        Query a specific peer node
        """
        results = node.query_peer(request.peer_id, request.query_vector, request.top_k)
        if results is not None:
            return {
                "status": "success",
                "peer_id": request.peer_id,
                "results": results,
                "count": len(results)
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to query peer {request.peer_id}"
            )
    
    @app.post("/sync")
    async def sync_vector_endpoint(request: SyncRequest):
        """
        Sync a specific vector to a target node
        """
        success = node.sync_vector(request.vector_id, request.target_node_id)
        if success:
            return {
                "status": "success",
                "message": f"Vector {request.vector_id} synced to {request.target_node_id}"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to sync vector {request.vector_id}"
            )
    
    @app.post("/register_peer")
    async def register_peer_endpoint(request: RegisterPeerRequest):
        """
        Register a new peer node and exchange representative vectors.
        """
        node.register_peer(request.peer_id, request.peer_url)
        # Normalize and cache peer representative vectors
        normalized = []
        try:
            for v in request.node_vectors:
                if len(v) != node.vector_size:
                    raise ValueError(f"Peer {request.peer_id}: Vector size mismatch. Expected {node.vector_size}, got {len(v)}")
                arr = np.array(v, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                normalized.append(arr.tolist())
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        node.peer_node_vectors[request.peer_id] = normalized
        print(f"Node {node.node_id}: Cached {len(normalized)} representative vectors for {request.peer_id}")

        if not node.node_vectors:
            print(f"Node {node.node_id}: ERROR: Peer registered but this node's representative vectors are not set.")
            raise HTTPException(status_code=500, detail="This node's representative vectors are not set.")

        return {
            "status": "success",
            "message": f"Peer {request.peer_id} registered",
            "total_peers": len(node.peer_nodes),
            "node_id": node.node_id,
            "node_vectors": node.node_vectors
        }
    
    @app.get("/peers")
    async def list_peers_endpoint():
        """List all registered peer nodes and their cached vector status"""
        peers_with_vectors = {
            pid: {
                "url": url,
                "vector_count": len(node.peer_node_vectors.get(pid, [])),
                "status": node.peer_status.get(pid, {}).get("status", "UNKNOWN"),
                "last_ok": node.peer_status.get(pid, {}).get("last_ok"),
                "last_check": node.peer_status.get(pid, {}).get("last_check")
            }
            for pid, url in node.peer_nodes.items()
        }
        return {
            "node_id": node.node_id,
            "peers": peers_with_vectors,
            "peer_count": len(node.peer_nodes)
        }
    
    @app.get("/count")
    async def count_endpoint():
        """
        Get the current vector count for this node
        """
        count = node.count_local_vectors()
        if count != -1:
            return {
                "node_id": node.node_id,
                "count": count
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to get count")
            
    @app.post("/set_node_vectors")
    async def set_node_vectors_endpoint(vectors: List[List[float]]):
        """
        Set multiple representative vectors for this node.
        """
        try:
            node.set_node_vectors(vectors)
            return {
                "status": "success",
                "node_id": node.node_id,
                "message": f"Set {len(node.node_vectors)} representative vectors",
                "node_vectors": node.node_vectors
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/get-embedding")
    async def get_embedding_endpoint(payload: Dict[str, Any]):
        """
        Riceve un JSON { "text": "..." } e restituisce { "embedding": [...] }.
        Tutta la logica per generare l'embedding è autocontenuta in questa funzione.
        """
        try:
            # Validazione input
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Field 'text' must be a non-empty string")

            # Use the per-node embedding service instance initialized at startup.
            svc = getattr(node, "embedding_service", None)
            if svc is None:
                # Fail fast: we do not lazily create the model here
                raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

            embedding = svc.get_embedding(text)

            return {"embedding": embedding}

        except Exception as e:
            # Error handling chiaro
            raise HTTPException(status_code=400, detail=str(e))
        
    @app.post("/search/local/string")
    async def search_local_string_endpoint(payload: Dict[str, Any], top_k: int = 5):
        """
        Search only the local database using a text string.
        This endpoint generates the embedding for the string and performs a vector search.
        """
        try:
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Field 'text' must be a non-empty string")

            # Use the per-node embedding service instance initialized at startup.
            svc = getattr(node, "embedding_service", None)
            if svc is None:
                raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

            embedding = svc.get_embedding(text)

            results = node.search_local(embedding, top_k)
            if results is not None:
                return {
                    "node_id": node.node_id,
                    "results": results,
                    "count": len(results)
                }
            else:
                raise HTTPException(status_code=500, detail="Local search failed")

        except Exception as e:
            # Error handling chiaro
            raise HTTPException(status_code=400, detail=str(e))
        
    @app.post("/search/federated/string")
    async def federated_search_string_endpoint(payload: Dict[str, Any], top_k: int = 5):
        """
        Search across all nodes (local + peers) using a text string.
        This endpoint generates the embedding for the string and performs a vector search.
        """
        try:
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Field 'text' must be a non-empty string")

            # Use the per-node embedding service instance initialized at startup.
            svc = getattr(node, "embedding_service", None)
            if svc is None:
                raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

            embedding = svc.get_embedding(text)

            results = node.federated_search(embedding, top_k)
            total_results = sum(len(r) for r in results.values())
            return {
                "status": "success",
                "results": results,
                "nodes_searched": len(results),
                "total_results": total_results
            }

        except Exception as e:
            # Error handling chiaro
            raise HTTPException(status_code=400, detail=str(e))
        
    @app.post("/add-vector/string")
    async def add_vector_string_endpoint(payload: Dict[str, Any]):
        """
        Add a new SINGLE vector using a text string.
        This node will generate the embedding, find the best cluster and route the vector
        to ALL replica nodes for that cluster.
        """
        try:
            text = payload.get("text")
            vector_id = payload.get("id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Field 'text' must be a non-empty string")
            if not isinstance(vector_id, str) or not vector_id.strip():
                raise ValueError("Field 'id' must be a non-empty string")

            # Use the per-node embedding service instance initialized at startup.
            svc = getattr(node, "embedding_service", None)
            if svc is None:
                raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

            embedding = svc.get_embedding(text)

            vec_data = VectorData(
                id=vector_id,
                vector=embedding,
                payload={"text": text}
            )

            # 1. Find all replica nodes for this vector
            replica_node_ids = node.find_best_nodes(vec_data.vector)

            print(f"Node {node.node_id}: Routing vector {vec_data.id} to {len(replica_node_ids)} replicas: {replica_node_ids}")

            success_nodes = []
            failed_nodes = []

            # 2. Send to all replicas
            for node_id in replica_node_ids:
                success = False
                if node_id == node.node_id:
                    # Store locally
                    print(f"Node {node.node_id}: Storing vector {vec_data.id} locally (replica).")
                    success = node.receive_vector(
                        from_node_id="external_client_routed", 
                        vector_data=vec_data
                    )
                else:
                    # Forward to peer
                    print(f"Node {node.node_id}: Forwarding vector {vec_data.id} to replica {node_id}.")
                    success = node.send_vector(node_id, vec_data)

                if success:
                    success_nodes.append(node_id)
                else:
                    failed_nodes.append(node_id)

            # 3. Report result
            if not success_nodes:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to store vector {vector_id} on any replica."
                )

            return {
                "status": "success",
                "message": f"Vector {vector_id} routed to {len(replica_node_ids)} replicas.",
                "vector_id": vector_id,
                "replicas_targeted": replica_node_ids,
                "replicas_succeeded": success_nodes,
                "replicas_failed": failed_nodes
            }
        except Exception as e:
            print(e)
        
    #  Visualization server endpoints

    # Generate the full network topology from this node's perspective
    # This performs a recursive traversal of all known peers.
    # It avoids revisiting nodes to prevent infinite loops.
    @app.post("/get-topology")
    async def get_topology():
        visited = set()
        topology = {}

        def _parse_url_info(url: Optional[str]):
            """Return (dns, ip, port).

            - dns: the hostname from the URL (may be None)
            - ip: resolved IP address for the hostname (or None if resolution failed)
            - port: integer port
            """
            if not url:
                return (None, None, None)
            try:
                parsed = urlparse(url)
                hostname = parsed.hostname
                port = parsed.port
                if port is None:
                    # default ports
                    port = 443 if parsed.scheme == "https" else 80
                ip = None
                try:
                    # Resolve DNS name to an IP (may raise)
                    if hostname:
                        ip = socket.gethostbyname(hostname)
                except Exception:
                    ip = None

                return (hostname, ip, port)
            except Exception:
                return (None, None, None)

        def traverse(node_id: str, node_url: Optional[str]):
            # Prevent revisiting the same node (avoid infinite loops)
            if node_id in visited:
                return
            visited.add(node_id)

            # Determine this node's dns/ip/port/address (prefer provided node_url, else use local self_url)
            use_url = node_url if node_url else (node.self_url if node_id == node.node_id else None)
            dns_name, ip_addr, port_num = _parse_url_info(use_url)

            # Prepare a structure to hold peer info
            # Allow mixed types (ip: str, port: int, address: str)
            peers_info: Dict[str, Dict[str, Optional[Any]]] = {}

            # For the local node, use the cached peer_nodes mapping
            if node_id == node.node_id:
                peers = node.peer_nodes
                # peers is pid -> url
                for pid, purl in peers.items():
                    pdns, pip, pport = _parse_url_info(purl)
                    peers_info[pid] = {"dns": pdns, "ip": pip, "port": pport}
            else:
                # For remote nodes, attempt to fetch their peer list via their /peers endpoint
                if not node_url:
                    topology[node_id] = {"dns": dns_name, "ip": ip_addr, "port": port_num, "peers": {}}
                    return
                try:
                    resp = requests.get(f"{node_url}/peers", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        remote_peers = data.get("peers", {})
                        # remote_peers: pid -> {"url": ..., ...}
                        for pid, info in remote_peers.items():
                            purl = info.get("url")
                            pdns, pip, pport = _parse_url_info(purl)
                            peers_info[pid] = {"dns": pdns, "ip": pip, "port": pport}
                    else:
                        peers_info = {}
                except requests.RequestException:
                    # If the remote call fails, record an empty peer list for that node
                    peers_info = {}

            topology[node_id] = {"dns": dns_name, "ip": ip_addr, "port": port_num, "peers": peers_info}

            # Recurse into each discovered peer (using its known URL), skipping already visited nodes
            for pid, info in peers_info.items():
                if pid not in visited:
                    # Prefer the original peer URL registered locally, else construct a URL from discovered ip/dns + port
                    peer_url = node.peer_nodes.get(pid)
                    if not peer_url:
                        pip = info.get("ip")
                        pport = info.get("port")
                        pdns = info.get("dns")
                        if pip:
                            peer_url = f"http://{pip}:{pport}"
                        elif pdns:
                            peer_url = f"http://{pdns}:{pport}"
                        
                    traverse(pid, peer_url)  # Recursive call for the peer

        # Start traversal from the local node
        traverse(node.node_id, node.self_url)

        # Convert topology dict to a list for easier JSON serialization
        topology_list = [
            {
                "node_id": node_id,
                "details": details,
                "peers": list(details["peers"].keys())
           
            }
            for node_id, details in topology.items()
        ]
        
        return {
            "node_id": node.node_id,
            "topology": topology_list
        }

    @app.post("/gossip/clusters")
    async def gossip_clusters_endpoint(payload: Dict[str, Any]):
        """
        Receive cluster updates from peers via gossip protocol.
        """
        from_node = payload.get("from_node")
        cluster_vectors = payload.get("cluster_vectors", [])
        
        if not cluster_vectors:
            return {"status": "ok", "message": "No clusters received"}
        
        success = node.receive_peer_clusters(from_node, cluster_vectors)
        
        # Send back own clusters
        local_cluster_ids = node.meta_hnsw.node_to_clusters.get(node.node_id, []) if node.meta_hnsw else []
        local_clusters = [node.meta_hnsw.cluster_centroids[cid].tolist() for cid in local_cluster_ids] if local_cluster_ids else []
        
        return {
            "status": "success" if success else "failed",
            "cluster_vectors": local_clusters
        }

    @app.post("/init-meta-hnsw")
    async def init_meta_hnsw_endpoint(payload: Dict[str, Any]):
        """
        Initialize Meta-HNSW with full cluster data from coordinator.
        Receives all centroids and node assignments.
        """
        try:
            dimension = payload.get("dimension", node.vector_size)
            max_clusters = payload.get("max_clusters", 500)
            all_centroids = payload.get("centroids", [])
            node_assignments = payload.get("node_assignments", {})
            
            # Initialize Meta-HNSW
            node.initialize_meta_hnsw(dimension=dimension, max_clusters=max_clusters)
            
            # Add all nodes' clusters
            for node_idx_str, cluster_indices in node_assignments.items():
                peer_node_id = f"node{int(node_idx_str) + 1}"
                cluster_vectors = [all_centroids[idx] for idx in cluster_indices]
                
                if peer_node_id == node.node_id:
                    # Add own clusters
                    node.add_local_clusters(cluster_vectors)
                else:
                    # Add peer clusters
                    node.receive_peer_clusters(peer_id, cluster_vectors)
            
            # Force rebuild to make index queryable
            node.meta_hnsw.force_rebuild()
            
            print(f"Node {node.node_id}: Meta-HNSW initialized with {len(all_centroids)} total clusters")
            
            return {
                "status": "success",
                "message": f"Meta-HNSW initialized with {len(all_centroids)} clusters",
                "node_id": node.node_id
            }
            
        except Exception as e:
            print(f"Node {node.node_id}: Error initializing Meta-HNSW: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search/p2p")
    async def search_p2p_endpoint(payload: Dict[str, Any]):
        """
        P2P search entry point (SERVER-SIDE ROUTING):
        1. Usa Meta-HNSW locale per trovare top-K nodi
        2. Interroga quei nodi (peer-to-peer)
        3. Aggrega risultati
        
        Payload:
            {
                "query_vector": [0.1, 0.2, ...],
                "top_k_nodes": 3,       # Quanti nodi interrogare
                "top_k_results": 5      # Quanti risultati per nodo
            }
        
        Returns:
            {
                "status": "success",
                "entry_node": "node3",
                "nodes_queried": 3,
                "target_nodes": ["node1", "node3", "node5"],
                "total_results": 15,
                "results_per_node": {
                    "node1": [{...}, {...}],
                    "node3": [{...}, {...}],
                    "node5": [{...}, {...}]
                },
                "best_match": {
                    "node": "node5",
                    "score": 0.98
                }
            }
        """
        try:
            query_vector = np.array(payload['query_vector'], dtype=np.float32)
            k_nodes = payload.get('top_k_nodes', 3)
            k_results = payload.get('top_k_results', 5)
            
            print(f"Node {node.node_id}: P2P search entry point activated")
            print(f"  Query params: top_k_nodes={k_nodes}, top_k_results={k_results}")
            
            # STEP 1: Usa Meta-HNSW LOCALE per routing intelligente
            if node.meta_hnsw is None:
                # Fallback: broadcast a tutti i peer
                print(f"  ⚠️  No Meta-HNSW available, falling back to broadcast")
                target_nodes = [node.node_id] + list(node.peer_nodes.keys())
                target_nodes = target_nodes[:k_nodes]  # Limita a k_nodes
            else:
                # Routing intelligente con Meta-HNSW locale

                print(f"  🔍 Using local Meta-HNSW for routing...")
                nearest_nodes = node.meta_hnsw.find_nearest_nodes(
                    query_vector, 
                    k_nodes=k_nodes
                )
                target_nodes = [node_name for node_name, _ in nearest_nodes]
                
                print(f"  📍 Meta-HNSW routing result: {target_nodes}")
                for node_name, distance in nearest_nodes:
                    print(f"     - {node_name}: distance={distance:.4f}")
            
            # STEP 2: Query nodi selezionati (peer-to-peer)
            all_results = {}
            best_score = -2.0
            best_node = "N/A"
            
            # Questo ciclo itera sulla lista dei nodi migliori (es. ["node1", "node5", "node3"])
            for target_node_id in target_nodes:
                
                # QUI LA CONDIZIONE CHIAVE:
                # Se l'ID del nodo target è uguale all'ID di questo server...
                if target_node_id == node.node_id:
                    # QUERY LOCALE...allora esegue una ricerca locale.
                    print(f"  🔎 Querying local database...")
                    results = node.search_local(query_vector.tolist(), k_results)
                else:
                    # QUERY REMOTA...altrimenti, interroga il peer remoto.
                    print(f"  📡 Querying peer {target_node_id}...")
                    results = node.query_peer(target_node_id, query_vector.tolist(), k_results)
                
                if results:
                    all_results[target_node_id] = results
                    
                    # Track best match
                    if results and results[0].get('score', -2) > best_score:
                        best_score = results[0]['score']
                        best_node = target_node_id
                        print(f"     ✓ New best match: {best_node} (score: {best_score:.4f})")
            
            # STEP 3: Aggrega e restituisci
            total_results = sum(len(r) for r in all_results.values())
            
            print(f"  ✅ P2P search complete: {total_results} results from {len(all_results)} nodes")
            
            return {
                "status": "success",
                "entry_node": node.node_id,
                "nodes_queried": len(all_results),
                "target_nodes": target_nodes,
                "total_results": total_results,
                "results_per_node": all_results,
                "best_match": {
                    "node": best_node,
                    "score": best_score
                },
                "routing_method": "meta_hnsw" if node.meta_hnsw else "broadcast"
            }
            
        except Exception as e:
            print(f"Node {node.node_id}: P2P search error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/debug/count-by-source")
    async def debug_count_by_source():
        """Count vectors by source_type for debugging."""
        try:
            bootstrap_count = 0
            routed_count = 0
            unknown_count = 0
            
            offset = None
            while True:
                payload = {
                    "limit": 1000,
                    "with_payload": True,
                    "with_vector": False
                }
                if offset:
                    payload["offset"] = offset
                
                response = requests.post(
                    f"{node.qdrant_url}/collections/{node.collection_name}/points/scroll",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                points = data.get("result", {}).get("points", [])
                
                if not points:
                    break
                
                for point in points:
                    source = point.get("payload", {}).get("source_type")
                    if source == "bootstrap":
                        bootstrap_count += 1
                    elif source == "routed":
                        routed_count += 1
                    else:
                        unknown_count += 1
                
                offset = data.get("result", {}).get("next_page_offset")
                if not offset:
                    break
            
            return {
                "node_id": node.node_id,
                "bootstrap": bootstrap_count,
                "routed": routed_count,
                "unknown": unknown_count,
                "total": bootstrap_count + routed_count + unknown_count
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/config/freeze")
    async def freeze_endpoint(payload: Dict[str, Any]):
        """Freeze node to prevent concurrent modifications during config change."""
        config_version = payload.get("config_version")
        node._enter_freeze_state(config_version)
        return {
            "status": "frozen",
            "node_id": node.node_id,
            "config_version": config_version
        }
    
    @app.post("/config/unfreeze")
    async def unfreeze_endpoint(payload: Dict[str, Any]):
        """Unfreeze node and process pending operations."""
        node._exit_freeze_state()
        return {
            "status": "unfrozen",
            "node_id": node.node_id,
            "pending_count": len(node._pending_inserts)
        }
    
    @app.get("/config/status")
    async def config_status_endpoint():
        """Get current config version and freeze state."""
        with node._config_lock:
            return {
                "node_id": node.node_id,
                "config_version": node.config_version,
                "is_frozen": node.is_frozen,
                "pending_inserts": len(node._pending_inserts)
            }

    @app.get("/cleanup-status")
    async def cleanup_status_endpoint():
        """Get cleanup readiness and completion status."""
        return {
            "node_id": node.node_id,
            "config_version": node.config_version,
            "cleanup_ready": node.cleanup_ready,
            "cleanup_complete": node.cleanup_complete,
            "clustering_complete": node.clustering_complete
        }
    
    @app.post("/execute-cleanup")
    async def execute_cleanup_endpoint(payload: Dict[str, Any]):
        """Execute coordinated cleanup phase."""
        config_version = payload.get("config_version")
        node_assignments = payload.get("node_assignments")
        vector_labels = payload.get("vector_labels")
        
        # Verify version match
        if node.config_version != config_version:
            raise HTTPException(
                status_code=409,
                detail=f"Version mismatch: expected {config_version}, got {node.config_version}"
            )
        
        # Execute cleanup
        node._execute_cleanup(node_assignments, vector_labels)
        
        return {
            "status": "success",
            "node_id": node.node_id,
            "config_version": config_version,
            "cleanup_complete": True
        }

    return app


def run_node(node_id: str, port: int, qdrant_host: str = "localhost", 
             qdrant_port: int = 6333, collection_name: str = "vectors",
             vector_size: int = 384):
    """
    Run a Qdrant node with FastAPI server
    """
    
    self_url = f"http://localhost:{port}"
    
    node = QdrantNodeWrapper(
        node_id, 
        qdrant_host, 
        qdrant_port, 
        collection_name,
        self_url=self_url,
        vector_size=vector_size
    )
    
    if node.create_collection(vector_size):
        print(f"Node {node_id}: Generating initial fallback node vector...")
        rand_vec = np.random.rand(vector_size).astype(np.float32)
        norm = np.linalg.norm(rand_vec)
        if norm > 0:
            rand_vec = rand_vec / norm
        
        try:
            node.set_node_vectors([rand_vec.tolist()])
        except ValueError as e:
            print(f"Node {node_id}: FATAL - Error setting initial node vectors: {e}")
            return
        
        initial_count = node.count_local_vectors()
        print(f"Node {node_id}: Initial vector count: {initial_count}")
    
    # Create a per-node embedding service instance
    try:
        node.embedding_service = EmbeddingService()
        print(f"Node {node_id}: Initialized embedding service (model: {node.embedding_service.model_name})")
    except Exception as e:
        node.embedding_service = None
        print(f"Node {node_id}: WARNING - could not initialize embedding service: {e}")
    
    app = create_app(node)
    
    print(f"\n{'='*60}")
    print(f"Starting Qdrant Node: {node_id}")
    print(f"FastAPI Server: {self_url}")
    print(f"Qdrant Backend: {node.qdrant_url}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}\n")
    
    # CHANGED: Suppress access logs for health checks
    import logging
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 4:
        node_id = sys.argv[1]
        port = int(sys.argv[2])
        db_port = int(sys.argv[3])
        run_node(node_id, port, "localhost", db_port, vector_size=384)
    else:
        print("Usage: python server.py <node_id> <port> <db_port>")
        print("Example: python server.py node1 8001 6333")
        
        print("\nStarting default node1 on port 8001 attached to Qdrant localhost:6333...")
        run_node("node1", 8001, qdrant_port=6333, vector_size=384)