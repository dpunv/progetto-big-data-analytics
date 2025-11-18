from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import uvicorn
from typing import List, Tuple, Dict, TypedDict
from server_logic import ServerApp
import argparse
from compound_types import *
from clustering_module import MetaHNSW
import threading
from concurrent import futures
import grpc
import p2p_pb2_grpc
from grpc_handler import P2PNodeServicer

app = FastAPI()
server = None

class AddVectorsRequest(BaseModel):
    id: int
    content: ListOfVectorsWithPayload

class AddVectorsPeerRequest(BaseModel):
    id: int
    content: ListOfVectorsComplete
    type: str

class QueryVectorsRequest(BaseModel):
    id: int
    query: ListOfVectors
    topk: int

class AddPeersRequest(BaseModel):
    id: int
    peers: List[Tuple[str, str, str]] # Added string for gRPC url

"""class MetaHNSWStructure(BaseModel):
    dimension: int
    max_clusters: int
    ef_construction: int
    M: int
    index_data: str

class AssignmentStructure(BaseModel):
    my_vectors: List[VectorId]
    peers_clusters: Dict[str, ListOfVectorsWithId]
    meta_hnsw: MetaHNSWStructure
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """

class SetClustersRequest(BaseModel):
    id: int
    content: dict # To review 
    model_config = ConfigDict(arbitrary_types_allowed=True)


@app.get("/")
async def healh_check_endpoint():
    return {
        "status": "healthy",
        "node_id": server.node_id,
        "is_coordinator": server.i_am_coord(),
        "is_clustered": server.status,
    }

@app.post("/query")
async def query_endpoint(query: QueryVectorsRequest):
    if server == None:
        raise "ServerApp not created"
    results = server.query(query.query, query.topk, query.id)
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/query_peer")
async def query_peer_endpoint(query: QueryVectorsRequest):
    if server == None:
        raise "ServerApp not created"
    results = server.query_me(query.query, query.topk)
    print(results)
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/add")
async def add_vectors_endpoint(vectors: AddVectorsRequest):
    print(f"vector add endpoint start: {server.node_id}")
    if server == None:
        raise "ServerApp not created"
    server.add_vectors_client(vectors.content, vectors.id)
    print(f"vector add endpoint end")

@app.post("/receive_vectors_peer")
async def receive_vectors_peer_endpoint(vectors: AddVectorsPeerRequest):
    print("receive_vectors_peer created")
    if server == None:
        raise "ServerApp not created"
    server.add_vectors(vectors.content, vectors.type, vectors.id)
    print("vector added")

@app.post("/register_peers")
async def register_peers_endpoint(peers: AddPeersRequest):
    if server == None:
        raise "ServerApp not created"
    server.add_peers(peers.peers)

@app.post("/set_clusters")
async def set_clusters_endpoint(data: SetClustersRequest):
    if server == None:
        raise "ServerApp not created"
    data.content["meta_hnsw"] = MetaHNSW.from_serializable_dict(data.content["meta_hnsw"])
    server.set_clusters(data.content, data.id)

@app.get("/notify_clustering")
async def notify_clustering_endpoint():
    if server == None:
        raise "ServerApp not created"
    print(f'node {server.node_id} is receiving clustering notify')
    server.notify_clustering()

@app.get("/count")
async def count_endpoint():
    if server == None:
        raise "ServerApp not created"
    return server.get_count_client() # Dict[str, int] # id server: count su quel server

@app.get("/count_peer")
async def count_peer_endpoint():
    if server == None:
        raise "ServerApp not created"
    return server.get_count() # Dict[str, int] # id server: count su quel server

# Add gRPC URL argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBDN Server Node")
    parser.add_argument("--node-name", type=str, required=True, help="Name of the node")
    parser.add_argument("--node-url", type=str, required=True, help="URL of this node")
    parser.add_argument("--qdrant-url", type=str, required=True, help="URL of Qdrant instance")
    parser.add_argument("--node-grpc-url", type=str, required=True, help="gRPC URL of this node") # NEW
    parser.add_argument("--coordinator-url", type=str, required=True, help="URL of coordinator node")
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    parser.add_argument("--num-before-clustering", type=int, default=10000, help="Number of vectors before triggering clustering")
    
    args = parser.parse_args()

    # Initialize ServerApp
    server = ServerApp(args.node_name, args.node_url, args.qdrant_url, args.node_grpc_url, args.coordinator_url, args.replicas, num_vectors_before_clustering=args.num_before_clustering)

    # --- START GRPC SERVER ---
    def serve_grpc(server_app, grpc_port):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        p2p_pb2_grpc.add_P2PNodeServicer_to_server(P2PNodeServicer(server_app), grpc_server)
        grpc_server.add_insecure_port(f'[::]:{grpc_port}')
        print(f"gRPC server started on port {grpc_port}")
        grpc_server.start()
        grpc_server.wait_for_termination()

    grpc_port = args.node_grpc_url.split(":")[-1]
    
    # Run gRPC in a background thread
    grpc_thread = threading.Thread(target=serve_grpc, args=(server, grpc_port))
    grpc_thread.daemon = True
    grpc_thread.start()

    # --- START FASTAPI (Main Thread) ---
    http_port = int(args.node_url.split(":")[-1])
    uvicorn.run(app, host="0.0.0.0", port=http_port)