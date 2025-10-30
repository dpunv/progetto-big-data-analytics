from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import requests
import uvicorn
from dataclasses import dataclass, asdict
import uuid
import json
import numpy as np  # --- NEW IMPORT ---
import utils

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

# --- NEW PYDANTIC MODEL ---
class RegisterPeerRequest(BaseModel):
    peer_id: str
    peer_url: str
    node_vector: List[float] # Peer's representative vector
# --- END NEW PYDANTIC MODEL ---

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
    
    # --- MODIFIED __init__ ---
    def __init__(self, node_id: str, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6335, 
                 collection_name: str = "vectors",
                 self_url: str = "http://localhost:8000", # New: Node's own URL for registration
                 vector_size: int = 384):                  # New: Vector dimension
        """
        Initialize the Qdrant node wrapper.
        
        Args:
            node_id: Unique identifier for this node
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the collection to work with
            self_url: The externally reachable URL of this FastAPI app
            vector_size: The dimension of the vectors being stored
        """
        self.node_id = node_id
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self.collection_name = collection_name
        self.peer_nodes = {}  # Dictionary of peer_node_id -> peer_url
        
        # --- NEW STATE FOR SMART ROUTING ---
        self.self_url = self_url
        self.vector_size = vector_size
        self.node_vector: Optional[List[float]] = None # This node's representative vector
        self.peer_node_vectors: Dict[str, List[float]] = {} # Cache of peer rep. vectors
        # --- END NEW STATE ---
    # --- END MODIFIED __init__ ---
        
    def register_peer(self, peer_id: str, peer_url: str):
        """Register a peer node for communication"""
        # This method is now just for local storage. 
        # The logic is handled in the /register_peer endpoint
        self.peer_nodes[peer_id] = peer_url
        print(f"Node {self.node_id}: Registered peer {peer_id} at {peer_url}")

    # --- NEW METHOD ---
    def set_node_vector(self, vector: List[float]):
        """Sets or updates this node's representative vector."""
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Node {self.node_id}: Vector size mismatch. "
                f"Expected {self.vector_size}, got {len(vector)}"
            )
        
        # Normalize the vector to ensure fair similarity comparison
        norm_vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(norm_vec)
        if norm > 0:
            norm_vec = norm_vec / norm
        
        self.node_vector = norm_vec.tolist()
        print(f"Node {self.node_id}: Set representative vector (first 3 dims): {self.node_vector[:3]}...")
    # --- END NEW METHOD ---
    
    # --- NEW METHOD ---
    def find_best_node(self, vector: List[float]) -> str:
        """
        Find the node (self or peer) whose representative vector
        is most similar to the given vector.
        
        Returns:
            str: The node_id of the best node.
        """
        if not self.node_vector:
            print(f"Node {self.node_id}: Warning: This node has no representative vector. Defaulting to self.")
            return self.node_id
            
        best_node_id = self.node_id
        # Cosine similarity: higher is better (closer to 1.0)
        best_similarity = utils.cosine_similarity(vector, self.node_vector)
        
        for peer_id, peer_vector in self.peer_node_vectors.items():
            sim = utils.cosine_similarity(vector, peer_vector)
            print(f"node {peer_id} has similarity {sim}")
            if sim > best_similarity:
                best_similarity = sim
                best_node_id = peer_id

        print(f"Node {self.node_id}: Best node for vector is {best_node_id} (sim: {best_similarity:.4f})")
        return best_node_id
    # --- END NEW METHOD ---
    
    def send_vector(self, target_node_id: str, vector_data: VectorData) -> bool:
        """
        Send a vector to another node.
        (This is now used for forwarding)
        
        Args:
            target_node_id: ID of the target node
            vector_data: VectorData object containing the vector and metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if target_node_id not in self.peer_nodes:
            print(f"Node {self.node_id}: Unknown peer {target_node_id}")
            return False
        
        peer_url = self.peer_nodes[target_node_id]
        
        try:
            # Send vector data to peer node
            payload = {
                "from_node": self.node_id, # Let the peer know who forwarded it
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
    
    def receive_vector(self, from_node_id: str, vector_data: VectorData) -> bool:
        """
        Receive and store a vector from another node (or external client).
        
        Args:
            from_node_id: ID of the sending node or "external_client_routed"
            vector_data: VectorData object to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Store vector in local Qdrant instance
            point = {
                "id": vector_data.id,
                "vector": vector_data.vector,
                "payload": {
                    **(vector_data.payload or {}),
                    "received_from": from_node_id,
                    "storage_node": self.node_id # Clarify this node stored it
                }
            }
            
            response = requests.put(
                f"{self.qdrant_url}/collections/{self.collection_name}/points",
                json={"points": [point]},
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"Node {self.node_id}: Stored vector {vector_data.id} from {from_node_id}")
                
                # --- MODIFICATION: Print count after successful save ---
                current_count = self.count_local_vectors()
                if current_count != -1:
                    print(f"Node {self.node_id}: ✨ Local vector count: {current_count}")
                # --- END MODIFICATION ---
                    
                return True
            else:
                print(f"Node {self.node_id}: Failed to store vector. Status: {response.status_code} {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error storing vector: {e}")
            return False
    
    def broadcast_vector(self, vector_data: VectorData) -> Dict[str, bool]:
        """
        Broadcast a vector to all peer nodes.
        (Note: This does NOT store locally, the endpoint /broadcast handles that)
        
        Args:
            vector_data: VectorData object to broadcast
            
        Returns:
            Dict mapping peer_id to success status
        """
        results = {}
        for peer_id in self.peer_nodes:
            results[peer_id] = self.send_vector(peer_id, vector_data)
        
        print(f"Node {self.node_id}: Broadcast complete. Success: {sum(results.values())}/{len(results)}")
        return results
    
    def query_peer(self, peer_id: str, query_vector: List[float], 
                   top_k: int = 5) -> Optional[List[Dict]]:
        """
        Query a peer node for similar vectors.
        
        Args:
            peer_id: ID of the peer to query
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of similar vectors or None if failed
        """
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
        """
        Search the local Qdrant database.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of similar vectors or None if failed
        """
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

    # --- NEW METHOD ---
    def count_local_vectors(self) -> int:
        """
        Count the number of vectors in the local Qdrant collection.
        
        Returns:
            int: Number of vectors, or -1 if an error occurred.
        """
        try:
            # Use the Qdrant count endpoint
            payload = {"exact": True} # Request an exact count
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/count",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                # Response structure is {"result": {"count": ...}}
                count = response.json().get("result", {}).get("count", 0)
                return count
            else:
                print(f"Node {self.node_id}: Error counting vectors. Status: {response.status_code} - {response.text}")
                return -1
                
        except requests.exceptions.RequestException as e:
            print(f"Node {self.node_id}: Error counting vectors: {e}")
            return -1
    # --- END NEW METHOD ---
            
    def federated_search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search across all nodes (local + peers) and aggregate results.
        
        Args:
            query_vector: Query vector
            top_k: Number of results per node
            
        Returns:
            Dictionary mapping node_id to search results
        """
        all_results = {}
        
        # Search local node
        local_results = self.search_local(query_vector, top_k)
        if local_results:
            all_results[self.node_id] = local_results
        
        # Search all peer nodes
        for peer_id in self.peer_nodes:
            peer_results = self.query_peer(peer_id, query_vector, top_k)
            if peer_results:
                all_results[peer_id] = peer_results
        
        print(f"Node {self.node_id}: Federated search complete across {len(all_results)} nodes")
        return all_results
    
    def sync_vector(self, vector_id: str, target_node_id: str) -> bool:
        """
        Sync a specific vector to a target node.
        
        Args:
            vector_id: ID of the vector to sync
            target_node_id: Target node to sync to
            
        Returns:
            bool: True if successful
        """
        # Retrieve vector from local storage
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
        """
        Create the collection if it doesn't exist.
        
        Args:
            vector_size: Dimension of vectors
            distance: Distance metric (Cosine, Euclid, Dot)
        
        Returns:
            bool: True if successful or already exists
        """
        try:
            # Check if collection exists
            response = requests.get(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Node {self.node_id}: Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
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
        Endpoint to receive vectors from other nodes
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
    
    # --- MODIFIED ENDPOINT ---
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
        
        # --- Store locally first ---
        # We call receive_vector, identifying this node as the source
        print(f"Node {node.node_id}: Storing broadcast vector {vector_data.id} locally...")
        local_success = node.receive_vector(node.node_id, vector_data)
        if not local_success:
            # Log a warning but continue to broadcast
            print(f"Node {node.node_id}: WARNING - Failed to store broadcast vector locally.")

        # --- Broadcast to peers ---
        broadcast_results = node.broadcast_vector(vector_data)
        
        return {
            "status": "success",
            "vector_id": vector_data.id,
            "local_storage": "success" if local_success else "failed",
            "broadcast_results": broadcast_results,
            "success_count": sum(broadcast_results.values()),
            "total_peers": len(broadcast_results)
        }
    # --- END MODIFICATION ---

    # --- MODIFIED ENDPOINT FOR SMART ROUTING ---
    @app.post("/add_vector")
    async def add_vector_endpoint(vector_data: VectorDataModel):
        """
        Add a new vector from an external client.
        This node will determine the best storage node (self or peer)
        based on vector similarity and route it accordingly.
        """
        vec_data = VectorData(
            id=vector_data.id,
            vector=vector_data.vector,
            payload=vector_data.payload
        )
        
        # --- SMART ROUTING LOGIC ---
        best_node_id = node.find_best_node(vec_data.vector)
        
        if best_node_id == node.node_id:
            # This is the best node. Store it locally.
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
            # A peer is a better match. Forward it.
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
                # If forwarding fails, the client must retry.
                # We do NOT store it locally as it doesn't belong here.
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to forward vector to {best_node_id}"
                )
        # --- END SMART ROUTING LOGIC ---
    # --- END MODIFIED ENDPOINT ---
    
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
    
    # --- MODIFIED ENDPOINT FOR VECTOR EXCHANGE ---
    @app.post("/register_peer")
    async def register_peer_endpoint(request: RegisterPeerRequest):
        """
        Register a new peer node and exchange representative vectors.
        """
        # Store the peer's info
        node.register_peer(request.peer_id, request.peer_url)
        node.peer_node_vectors[request.peer_id] = request.node_vector
        print(f"Node {node.node_id}: Cached representative vector for {request.peer_id}")
        
        if not node.node_vector:
            # This should not happen if run_node() works correctly
            print(f"Node {node.node_id}: ERROR: Peer registered but this node's vector is not set.")
            raise HTTPException(status_code=500, detail="This node's vector is not set.")
        
        # Respond with this node's info for the peer to cache
        return {
            "status": "success",
            "message": f"Peer {request.peer_id} registered",
            "total_peers": len(node.peer_nodes),
            "node_id": node.node_id,
            "node_vector": node.node_vector # Return this node's vector
        }
    # --- END MODIFIED ENDPOINT ---
    
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
    
    # --- New endpoint to get the count directly ---
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
            
    # --- NEW ENDPOINT to manually set node vector ---
    @app.post("/set_node_vector")
    async def set_node_vector_endpoint(vector: List[float]):
        """
        Manually set or update this node's representative vector.
        Useful for assigning pre-calculated dissimilar vectors.
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
    # --- END NEW ENDPOINT ---
            
    return app


# --- MODIFIED FUNCTION ---
def run_node(node_id: str, port: int, qdrant_host: str = "localhost", 
             qdrant_port: int = 6333, collection_name: str = "vectors",
             vector_size: int = 384):  # Add vector_size parameter
    """
    Run a Qdrant node with FastAPI server
    """
    
    # --- NEW: Define self_url ---
    # In a real deployment, this would come from an env var or service discovery
    self_url = f"http://localhost:{port}"
    
    node = QdrantNodeWrapper(
        node_id, 
        qdrant_host, 
        qdrant_port, 
        collection_name,
        self_url=self_url,       # <-- Pass new arg
        vector_size=vector_size  # <-- Pass new arg
    )
    
    # Create collection if it doesn't exist
    if node.create_collection(vector_size):
        
        # --- NEW: Set initial random vector for this node ---
        # This is the "negotiation" step. Each node starts with a random,
        # normalized vector. You can override this by calling /set_node_vector
        print(f"Node {node_id}: Generating initial random node vector...")
        rand_vec = np.random.rand(vector_size).astype(np.float32)
        norm = np.linalg.norm(rand_vec)
        if norm > 0:
            rand_vec = rand_vec / norm
        
        try:
            node.set_node_vector(rand_vec.tolist())
        except ValueError as e:
            print(f"Node {node_id}: FATAL - Error setting initial node vector: {e}")
            return # Don't start the server if this fails
        # --- END NEW ---
        
        # Print initial count on startup
        initial_count = node.count_local_vectors()
        print(f"Node {node_id}: Initial vector count: {initial_count}")
    
    app = create_app(node)
    
    print(f"\n{'='*60}")
    print(f"Starting Qdrant Node: {node_id}")
    print(f"FastAPI Server: {self_url}")
    print(f"Qdrant Backend: {node.qdrant_url}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port) # Host 0.0.0.0 to be reachable
# --- END MODIFIED FUNCTION ---


if __name__ == "__main__":
    import sys
    
    # Example: python script.py node1 8001
    if len(sys.argv) >= 4:
        node_id = sys.argv[1]
        port = int(sys.argv[2])
        db_port = int(sys.argv[3])
        # Note: You must pass vector_size if you're not using the default
        run_node(node_id, port, "localhost", db_port, vector_size=384) # Defaulting to size 4 for tests
    else:
        print("Usage: python script.py <node_id> <port> <db_port>")
        print("Example: python script.py node1 8001 6333")
        
        # Default: run node1 on port 8001
        print("\nStarting default node1 on port 8001 attached to Qdrant localhost:6333...")
        # Note: Set vector_size=4 to match test_cluster.py
        run_node("node1", 8001, qdrant_port=6333, vector_size=384)