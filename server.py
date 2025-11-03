from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import requests
import uvicorn
from dataclasses import dataclass, asdict
import uuid
import json
import numpy as np
import utils
from collections import defaultdict # --- NEW IMPORT ---

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
    node_vector: List[float] # Peer's representative vector

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
    
    def __init__(self, node_id: str, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6335, 
                 collection_name: str = "vectors",
                 self_url: str = "http://localhost:8000",
                 vector_size: int = 384):
        self.node_id = node_id
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self.collection_name = collection_name
        self.peer_nodes = {}
        self.self_url = self_url
        self.vector_size = vector_size
        self.node_vector: Optional[List[float]] = None
        self.peer_node_vectors: Dict[str, List[float]] = {}
        
        # --- NEW: Local Centroid Management (MANCAVA!) ---
        self.local_centroid: Optional[np.ndarray] = None
        self.previous_centroid: Optional[np.ndarray] = None
        self.centroid_vector_sum: Optional[np.ndarray] = None
        self.centroid_vector_count: int = 0
        
        # Configuration
        self.centroid_update_interval: int = 1000
        self.centroid_drift_threshold: float = 0.05
        self.coordinator_url: Optional[str] = None
        
        self.insertions_since_update: int = 0
        # --- END NEW ---
        
    def register_peer(self, peer_id: str, peer_url: str):
        self.peer_nodes[peer_id] = peer_url
        print(f"Node {self.node_id}: Registered peer {peer_id} at {peer_url}")

    def set_node_vector(self, vector: List[float]):
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Node {self.node_id}: Vector size mismatch. "
                f"Expected {self.vector_size}, got {len(vector)}"
            )
        norm_vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(norm_vec)
        if norm > 0:
            norm_vec = norm_vec / norm
        self.node_vector = norm_vec.tolist()
        print(f"Node {self.node_id}: Set representative vector (first 3 dims): {self.node_vector[:3]}...")
    
    def find_best_node(self, vector: List[float]) -> str:
        if not self.node_vector:
            print(f"Node {self.node_id}: Warning: This node has no representative vector. Defaulting to self.")
            return self.node_id
            
        best_node_id = self.node_id
        best_similarity = utils.cosine_similarity(vector, self.node_vector)
        
        for peer_id, peer_vector in self.peer_node_vectors.items():
            sim = utils.cosine_similarity(vector, peer_vector)
            # print(f"node {peer_id} has similarity {sim}") # Too noisy for bulk
            if sim > best_similarity:
                best_similarity = sim
                best_node_id = peer_id

        # print(f"Node {self.node_id}: Best node for vector is {best_node_id} (sim: {best_similarity:.4f})") # Too noisy for bulk
        return best_node_id
    
    def send_vector(self, target_node_id: str, vector_data: VectorData) -> bool:
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
        """
        Send a BATCH of vectors to another node.
        (This is used for forwarding)
        """
        if target_node_id not in self.peer_nodes:
            print(f"Node {self.node_id}: Unknown peer {target_node_id}")
            return False
        
        peer_url = self.peer_nodes[target_node_id]
        
        try:
            # Convert List[VectorData] to List[Dict] for JSON
            vectors_data_dicts = [asdict(vd) for vd in vectors_data]

            payload = {
                "from_node": self.node_id, # Let the peer know who forwarded it
                "vectors_data": vectors_data_dicts
            }
            
            response = requests.post(
                f"{peer_url}/receive_vectors_bulk",
                json=payload,
                timeout=30 # Increased timeout for bulk ops
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
                params={"wait": "true"},
                json={"points": points},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                print(f"Node {self.node_id}: Stored {len(points)} vectors from {from_node_id}")
                
                # --- NEW: Update centroid incrementally (MANCAVA!) ---
                for vector_data in vectors_data:
                    self.update_centroid_incremental(vector_data.vector)
                # --- END NEW ---
                
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
            
            # --- NEW: Log Qdrant response details ---
            if response.status_code != 200:
                print(f"Node {self.node_id}: ❌ Qdrant returned {response.status_code}")
                print(f"  URL: {self.qdrant_url}/collections/{self.collection_name}/points/search")
                print(f"  Response body: {response.text[:500]}")  # First 500 chars
                return None
            # --- END NEW ---
            
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

    # --- NEW: Centroid Management Methods (MANCAVANO!) ---
    def initialize_centroid(self, initial_centroid: List[float]):
        """Initialize local centroid (called by coordinator during setup)."""
        centroid_array = np.array(initial_centroid, dtype=np.float32)
        
        norm = np.linalg.norm(centroid_array)
        if norm > 0:
            centroid_array = centroid_array / norm
        
        self.local_centroid = centroid_array
        self.previous_centroid = centroid_array.copy()
        self.centroid_vector_sum = np.zeros(self.vector_size, dtype=np.float32)
        self.centroid_vector_count = 0
        
        print(f"Node {self.node_id}: Initialized local centroid (first 3 dims: {centroid_array[:3]})")
    
    def update_centroid_incremental(self, vector: List[float]):
        """Update running centroid statistics with new vector."""
        if self.local_centroid is None:
            return
        
        vector_array = np.array(vector, dtype=np.float32)
        
        self.centroid_vector_sum += vector_array
        self.centroid_vector_count += 1
        self.insertions_since_update += 1
        
        if self.insertions_since_update >= self.centroid_update_interval:
            self._recalculate_and_check_drift()
    
    def _recalculate_and_check_drift(self):
        """Recalculate centroid and check for drift."""
        if self.centroid_vector_count == 0:
            return
        
        new_centroid = self.centroid_vector_sum / self.centroid_vector_count
        
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        
        cosine_sim = np.dot(self.previous_centroid, new_centroid) / (
            np.linalg.norm(self.previous_centroid) * np.linalg.norm(new_centroid)
        )
        cosine_dist = 1 - cosine_sim
        
        print(f"Node {self.node_id}: Centroid drift check - distance: {cosine_dist:.6f} "
              f"(threshold: {self.centroid_drift_threshold}, insertions: {self.insertions_since_update})")
        
        self.local_centroid = new_centroid
        
        if cosine_dist > self.centroid_drift_threshold:
            print(f"Node {self.node_id}: ⚠️  Centroid drift {cosine_dist:.6f} exceeds threshold!")
            self._notify_coordinator_centroid_change(new_centroid.tolist(), cosine_dist)
            self.previous_centroid = new_centroid.copy()
        
        self.insertions_since_update = 0
    
    def _notify_coordinator_centroid_change(self, new_centroid: List[float], drift: float):
        """Notify coordinator of centroid change."""
        if not self.coordinator_url:
            return
        
        try:
            payload = {
                "node_id": self.node_id,
                "new_centroid": new_centroid,
                "drift": drift,
                "vector_count": self.centroid_vector_count
            }
            
            response = requests.post(
                f"{self.coordinator_url}/update_centroid",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Node {self.node_id}: ✅ Coordinator notified of centroid change")
            else:
                print(f"Node {self.node_id}: ⚠️  Coordinator notification failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: ⚠️  Error notifying coordinator: {e}")
    
    def set_coordinator_url(self, url: str):
        """Set the coordinator URL for centroid update notifications."""
        self.coordinator_url = url
        print(f"Node {self.node_id}: Coordinator URL set to {url}")
    
    def get_centroid_stats(self) -> dict:
        """Get current centroid statistics."""
        return {
            "node_id": self.node_id,
            "centroid_initialized": self.local_centroid is not None,
            "vector_count": self.centroid_vector_count,
            "insertions_since_update": self.insertions_since_update,
            "update_interval": self.centroid_update_interval,
            "drift_threshold": self.centroid_drift_threshold,
            "current_centroid": self.local_centroid.tolist() if self.local_centroid is not None else None
        }
    # --- END NEW ---

# FastAPI Application
def create_app(node: QdrantNodeWrapper) -> FastAPI:
    """Create FastAPI application for a Qdrant node"""
    
    app = FastAPI(
        title=f"Qdrant Node API - {node.node_id}",
        description="Distributed Qdrant vector database node API",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "node_id": node.node_id,
            "peers": list(node.peer_nodes.keys()),
            "node_vector_set": node.node_vector is not None
        }
    
    @app.post("/receive_vector")
    async def receive_vector(request: SendVectorRequest):
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
    async def receive_vectors_bulk_endpoint(request: SendVectorsBulkRequest):
        """
        Endpoint to receive a BATCH of vectors from other nodes
        """
        # Convert from Pydantic models to VectorData dataclass
        vectors_data_list = [
            VectorData(
                id=vd.id,
                vector=vd.vector,
                payload=vd.payload
            ) for vd in request.vectors_data
        ]
        
        success = node.receive_vectors_bulk(request.from_node, vectors_data_list)
        
        if success:
            return {
                "status": "success",
                "message": f"Stored {len(vectors_data_list)} vectors",
                "node_id": node.node_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store bulk vectors")
    # --- END NEW ENDPOINT ---

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
        This node will determine the best storage node (self or peer)
        based on vector similarity and route it accordingly.
        """
        vec_data = VectorData(
            id=vector_data.id,
            vector=vector_data.vector,
            payload=vector_data.payload
        )
        
        best_node_id = node.find_best_node(vec_data.vector)
        
        if best_node_id == node.node_id:
            print(f"Node {node.node_id}: Storing vector {vec_data.id} locally (best match).")
            success = node.receive_vector(
                from_node_id="external_client_routed", 
                vector_data=vec_data
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Vector {vector_data.id} stored locally",
                    "node_id": node.node_id,
                    "action": "stored_locally"
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to store vector locally")
        
        else:
            print(f"Node {node.node_id}: Forwarding vector {vec_data.id} to {best_node_id}.")
            success = node.send_vector(best_node_id, vec_data)
            if success:
                return {
                    "status": "success",
                    "message": f"Vector {vector_data.id} forwarded to {best_node_id}",
                    "node_id": best_node_id,
                    "action": "forwarded"
                }
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to forward vector to {best_node_id}"
                )

    # --- NEW ENDPOINT for client bulk insert ---
    @app.post("/add_vectors_bulk")
    async def add_vectors_bulk_endpoint(vectors: List[VectorDataModel], background_tasks: BackgroundTasks):
        """
        Add a new BATCH of vectors from an external client.
        This node will determine the best storage node for EACH vector
        and route them in batches using background tasks.
        """
        
        # 1. Classify all vectors and group them by best node
        nodes_to_vectors: Dict[str, List[VectorData]] = defaultdict(list)
        
        for vd_model in vectors:
            vec_data = VectorData(
                id=vd_model.id,
                vector=vd_model.vector,
                payload=vd_model.payload
            )
            best_node_id = node.find_best_node(vec_data.vector)
            nodes_to_vectors[best_node_id].append(vec_data)
            
        print(f"Node {node.node_id}: Received bulk of {len(vectors)}. Routing to {len(nodes_to_vectors)} nodes.")

        # 2. Process/forward the batches in the background
        routing_summary = {}
        
        for target_node_id, vectors_list in nodes_to_vectors.items():
            batch_size = len(vectors_list)
            routing_summary[target_node_id] = batch_size
            
            if target_node_id == node.node_id:
                # This is the best node. Store locally (in background).
                print(f"Node {node.node_id}: Queuing local storage of {batch_size} vectors.")
                background_tasks.add_task(
                    node.receive_vectors_bulk,
                    from_node_id="external_client_routed",
                    vectors_data=vectors_list
                )
            
            else:
                # A peer is a better match. Forward it (in background).
                print(f"Node {node.node_id}: Queuing forward of {batch_size} vectors to {target_node_id}.")
                background_tasks.add_task(
                    node.send_vectors_bulk,
                    target_node_id=target_node_id,
                    vectors_data=vectors_list
                )
                
        # 3. Return an immediate response to the client with the routing summary
        return {
            "status": "processing_bulk",
            "message": f"Processing {len(vectors)} vectors. Forwarding to {len(nodes_to_vectors)} nodes.",
            "batches": routing_summary # This is what the client needs
        }
    # --- END NEW ENDPOINT ---
    
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
        node.peer_node_vectors[request.peer_id] = request.node_vector
        print(f"Node {node.node_id}: Cached representative vector for {request.peer_id}")
        
        if not node.node_vector:
            print(f"Node {node.node_id}: ERROR: Peer registered but this node's vector is not set.")
            raise HTTPException(status_code=500, detail="This node's vector is not set.")
        
        return {
            "status": "success",
            "message": f"Peer {request.peer_id} registered",
            "total_peers": len(node.peer_nodes),
            "node_id": node.node_id,
            "node_vector": node.node_vector
        }
    
    @app.get("/peers")
    async def list_peers():
        """
        List all registered peer nodes and their cached vector status
        """
        peers_with_vectors = {
            pid: {"url": url, "has_vector": pid in node.peer_node_vectors}
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
    
    # --- NEW ENDPOINT: Get all vectors for centroid calculation ---
    @app.get("/vectors/all")
    async def get_all_vectors_endpoint(limit: int = 10000):
        """
        Get all vectors stored in this node (for Meta-HNSW centroid calculation).
        
        Args:
            limit: Maximum number of vectors to return (default 10000)
        
        Returns:
            List of vectors [N, dimension]
        """
        try:
            payload = {
                "limit": limit,
                "with_payload": False,
                "with_vector": True
            }
            response = requests.post(
                f"{node.qdrant_url}/collections/{node.collection_name}/points/scroll",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                points = response.json().get("result", {}).get("points", [])
                vectors = [point["vector"] for point in points]
                
                return {
                    "node_id": node.node_id,
                    "count": len(vectors),
                    "vectors": vectors
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch vectors: {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error fetching vectors: {e}")
    # --- END NEW ENDPOINT ---
            
    @app.post("/set_node_vector")
    async def set_node_vector_endpoint(vector: List[float]):
        """
        Manually set or update this node's representative vector.
        """
        try:
            node.set_node_vector(vector)
            return {
                "status": "success",
                "node_id": node.node_id,
                "message": "Node vector updated successfully",
                "node_vector": node.node_vector
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # --- NEW ENDPOINTS: Centroid Management ---
    @app.post("/initialize_centroid")
    async def initialize_centroid_endpoint(centroid: List[float]):
        """Initialize this node's local centroid (called by coordinator)."""
        try:
            node.initialize_centroid(centroid)
            return {
                "status": "success",
                "node_id": node.node_id,
                "message": "Centroid initialized"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize centroid: {e}")
    
    @app.post("/set_coordinator")
    async def set_coordinator_endpoint(coordinator_url: str):
        """Set the coordinator URL for this node."""
        node.set_coordinator_url(coordinator_url)
        return {
            "status": "success",
            "node_id": node.node_id,
            "coordinator_url": coordinator_url
        }
    
    @app.get("/centroid_stats")
    async def centroid_stats_endpoint():
        """Get current centroid statistics for this node."""
        return node.get_centroid_stats()
    
    @app.post("/force_centroid_update")
    async def force_centroid_update_endpoint():
        """Force immediate centroid recalculation and drift check."""
        try:
            node._recalculate_and_check_drift()
            return {
                "status": "success",
                "node_id": node.node_id,
                "stats": node.get_centroid_stats()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update centroid: {e}")
    # --- END NEW ENDPOINTS ---

    @app.get("/debug/info")
    async def debug_info_endpoint():
        """
        Get debug information about this node's configuration.
        """
        count = node.count_local_vectors()
        return {
            "node_id": node.node_id,
            "fastapi_port": node.self_url,
            "qdrant_url": node.qdrant_url,
            "collection_name": node.collection_name,
            "vector_count": count,
            "peers": list(node.peer_nodes.keys()),
            "has_node_vector": node.node_vector is not None,
            "centroid_initialized": node.local_centroid is not None
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
        
        # NOTE: This initial random vector is now
        # immediately overwritten by the client's `register_peers`
        # step, which calls `/set_node_vector`.
        # This is fine, but we'll keep it as a fallback.
        print(f"Node {node_id}: Generating initial fallback node vector...")
        rand_vec = np.random.rand(vector_size).astype(np.float32)
        norm = np.linalg.norm(rand_vec)
        if norm > 0:
            rand_vec = rand_vec / norm
        
        try:
            node.set_node_vector(rand_vec.tolist())
        except ValueError as e:
            print(f"Node {node_id}: FATAL - Error setting initial node vector: {e}")
            return
        
        initial_count = node.count_local_vectors()
        print(f"Node {node_id}: Initial vector count: {initial_count}")
    
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