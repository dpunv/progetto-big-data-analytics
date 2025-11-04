from fastapi import FastAPI, HTTPException, BackgroundTasks, Request  # NEW: Add Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import requests
import uvicorn
from dataclasses import dataclass, asdict
import uuid
import json
import numpy as np
import utils
import msgpack  # NEW: Import MessagePack
from collections import defaultdict
from urllib.parse import urlparse
import socket
import threading
import time
from embeddings import EmbeddingService

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

class QdrantNodeWrapper:
    """
    Wrapper for a Qdrant vector database node with inter-node communication capabilities.
    """

    def __init__(self, node_id: str, qdrant_host: str = "localhost", qdrant_port: int = 6335, collection_name: str = "vectors", self_url: str = "http://localhost:8000", vector_size: int = 384):
        self.node_id = node_id
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self.collection_name = collection_name
        self.peer_nodes = {}  # Dictionary of peer_node_id -> peer_url
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

        best_similarity = -2.0  # Use -2.0 for cosine similarity
        best_centroid = None

        # 2. Find the single closest centroid vector
        for node_id, centroids in all_centroids.items():
            for centroid in centroids:
                sim = utils.cosine_similarity(vector, centroid)
                if sim > best_similarity:
                    best_similarity = sim
                    best_centroid = centroid
        
        if best_centroid is None:
            # Fallback: no centroids found at all?
            print(f"Node {self.node_id}: Warning: No centroids found. Defaulting to self.")
            return [self.node_id]

        # 3. Find all nodes that host this best_centroid (replicas)
        replica_nodes = []
        # Convert to numpy array once for efficient comparison
        best_centroid_np = np.array(best_centroid, dtype=np.float32) 

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
        
        # print(f"Node {self.node_id}: Vector maps to {len(replica_nodes)} nodes: {replica_nodes} (sim: {best_similarity:.4f})") # Too noisy for bulk
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
        """
        try:
            points = []
            for vector_data in vectors_data:
                point = {
                    "id": vector_data.id,
                    "vector": vector_data.vector,
                    "payload": {
                        **(vector_data.payload or {}),
                        "received_from": from_node_id,
                        "storage_node": self.node_id
                    }
                }
                points.append(point)
            
            if not points:
                print(f"Node {self.node_id}: Received empty bulk insert from {from_node_id}.")
                return True # Technically not a failure

            response = requests.put(
                f"{self.qdrant_url}/collections/{self.collection_name}/points",
                params={"wait": "true"}, # Ensure operation completes
                json={"points": points},
                timeout=30 # Increase timeout for bulk ops
            )
            
            if response.status_code in [200, 201]:
                print(f"Node {self.node_id}: Stored {len(points)} vectors from {from_node_id}")
                
                # Print count after successful save
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
                        if status_info["status"] != "UP":
                            status_info["status"] = "UP"
                            print(f"Node {self.node_id}: Peer {peer_id} is UP")
                    else:
                        status_info["fail_count"] = status_info.get("fail_count", 0) + 1
                        if status_info["fail_count"] >= failure_threshold and status_info.get("status") != "DOWN":
                            status_info["status"] = "DOWN"
                            print(f"Node {self.node_id}: Peer {peer_id} marked DOWN (non-200 responses)")
                except requests.exceptions.RequestException:
                    status_info["fail_count"] = status_info.get("fail_count", 0) + 1
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
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
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
        
        # 1. Classify all vectors and group them by best node
        nodes_to_vectors: Dict[str, List[VectorData]] = defaultdict(list)
        
        total_vectors = len(vectors)
        total_routings = 0
        
        for vd_model in vectors:
            vec_data = VectorData(
                id=vd_model.id,
                vector=vd_model.vector,
                payload=vd_model.payload
            )
            
            # Find ALL nodes responsible for this vector
            replica_node_ids = node.find_best_nodes(vec_data.vector)
            
            for node_id in replica_node_ids:
                nodes_to_vectors[node_id].append(vec_data)
                total_routings += 1
        
        print(f"Node {node.node_id}: Received bulk of {total_vectors}. Routing to {len(nodes_to_vectors)} nodes (total routings: {total_routings}).")

        # 2. Process/forward the batches in the background
        routing_summary = {}
        
        for target_node_id, vectors_list in nodes_to_vectors.items():
            batch_size = len(vectors_list)
            routing_summary[target_node_id] = batch_size
            
            if target_node_id == node.node_id:
                # Store locally (in background)
                print(f"Node {node.node_id}: Queuing local storage of {batch_size} vectors.")
                background_tasks.add_task(
                    node.receive_vectors_bulk,
                    from_node_id="external_client_routed",
                    vectors_data=vectors_list
                )
            else:
                # Forward to peer (in background, using MessagePack)
                print(f"Node {node.node_id}: Queuing forward of {batch_size} vectors to {target_node_id}.")
                background_tasks.add_task(
                    node.send_vectors_bulk,
                    target_node_id=target_node_id,
                    vectors_data=vectors_list
                )
        
        # 3. Return immediate response (JSON is fine for small response)
        return {
            "status": "processing_bulk",
            "message": f"Processing {total_vectors} vectors. Total routings: {total_routings} across {len(nodes_to_vectors)} nodes.",
            "batches": routing_summary
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
        Add a new SINGLE vector from an external client using a text string.
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
                        elif pdns and pport:
                            peer_url = f"http://{pdns}:{pport}"
                        else:
                            peer_url = None
                    traverse(pid, peer_url)

        # Start traversal from this node (use self_url as its URL)
        traverse(node.node_id, node.self_url)
        return {"topology": topology}

    class NodeInfoRequest(BaseModel):
        node_id: Optional[str] = None
        dns: Optional[str] = None
        ip: Optional[str] = None
        port: Optional[int] = None
        visited: Optional[List[str]] = None

    @app.post("/get-node-info")
    async def get_node_info(request: NodeInfoRequest):
        """
        POST endpoint to fetch info about a node.
        - If no target (node_id/dns/ip) provided: return this node's info.
        - If target matches this node: return this node's info.
        - If target resolves to a direct URL, contact that node and return its info.
        - Otherwise forward the request to neighbors (walking the graph) and wait for first successful response.
        The 'visited' list is propagated to avoid infinite loops.
        """
        # Prepare visited set to avoid infinite loops
        visited = set(request.visited or [])
        if node.node_id not in visited:
            visited.add(node.node_id)

        # Helper to produce local info payload
        def local_info():
            return {
                "node_id": node.node_id,
                "vector_size": node.vector_size,
                "peer_count": len(node.peer_nodes),
                "node_vectors_count": len(node.node_vectors),
                "self_url": node.self_url,
                "vector_count": node.count_local_vectors()
            }

        # If no target specified, return local info
        if not (request.node_id or request.dns or request.ip):
            return local_info()

        # If the request explicitly targets this node id, return local info
        if request.node_id and request.node_id == node.node_id:
            return local_info()

        # Build a direct URL if ip/dns and port are provided
        def build_url(dns, ip, port):
            if ip:
                p = port if port is not None else 80
                return f"http://{ip}:{p}"
            if dns:
                p = port if port is not None else 80
                return f"http://{dns}:{p}"
            return None

        # Try direct contact if ip/dns provided
        direct_url = None
        if request.ip or request.dns:
            direct_url = build_url(request.dns, request.ip, request.port)

        # If node_id is known locally, prefer our registered peer URL
        if request.node_id and request.node_id in node.peer_nodes:
            direct_url = node.peer_nodes[request.node_id]

        # Prepare payload to forward (propagate visited)
        forward_payload = {
            "node_id": request.node_id,
            "dns": request.dns,
            "ip": request.ip,
            "port": request.port,
            "visited": list(visited)
        }

        # Try contacting the direct URL first (if available and not ourselves)
        if direct_url:
            # Avoid contacting ourselves again
            if direct_url != node.self_url:
                try:
                    resp = requests.post(f"{direct_url}/get-node-info", json=forward_payload, timeout=5)
                    if resp.status_code == 200:
                        return resp.json()
                except requests.RequestException:
                    # Failed to contact direct node, will try neighbors below
                    pass

        # If direct contact failed or wasn't possible, forward to neighbors
        # Iterate peers and ask them (skip visited)
        for peer_id, peer_url in node.peer_nodes.items():
            if peer_id in visited:
                continue
            # mark as visited for this hop
            new_payload = dict(forward_payload)
            new_payload["visited"] = list(visited | {peer_id})
            try:
                resp = requests.post(f"{peer_url}/get-node-info", json=new_payload, timeout=5)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                # ignore and try next peer
                continue

        # If nothing found
        raise HTTPException(status_code=404, detail="Target node not found in network")

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
        
    # NOTE: This initial random vector is overwritten by the client's `register_peers`
    # step, which calls `/set_node_vectors`.
    # Keep it as a fallback.
        print(f"Node {node_id}: Generating initial fallback node vector...")
        rand_vec = np.random.rand(vector_size).astype(np.float32)
        norm = np.linalg.norm(rand_vec)
        if norm > 0:
            rand_vec = rand_vec / norm
        
        try:
            # set initial fallback as a single-item list of representative vectors
            node.set_node_vectors([rand_vec.tolist()])
        except ValueError as e:
            print(f"Node {node_id}: FATAL - Error setting initial node vectors: {e}")
            return
        
        initial_count = node.count_local_vectors()
        print(f"Node {node_id}: Initial vector count: {initial_count}")
    # Create a per-node embedding service instance (each node owns its model)
    try:
        node.embedding_service = EmbeddingService()
        print(f"Node {node_id}: Initialized embedding service (model: {node.embedding_service.model_name})")
    except Exception as e:
        # If model libs not available, attach None and handle at request time
        node.embedding_service = None
        print(f"Node {node_id}: WARNING - could not initialize embedding service: {e}")
    
    app = create_app(node)
    
    print(f"\n{'='*60}")
    print(f"Starting Qdrant Node: {node_id}")
    print(f"FastAPI Server: {self_url}")
    print(f"Qdrant Backend: {node.qdrant_url}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 4:
        node_id = sys.argv[1]
        port = int(sys.argv[2])
        db_port = int(sys.argv[3])
        run_node(node_id, port, "localhost", db_port, vector_size=384)
    else:
        print("Usage: python script.py <node_id> <port> <db_port>")
        print("Example: python script.py node1 8001 6333")
        
        print("\nStarting default node1 on port 8001 attached to Qdrant localhost:6333...")
        run_node("node1", 8001, qdrant_port=6333, vector_size=384)