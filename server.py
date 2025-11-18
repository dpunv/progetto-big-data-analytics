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
import logging
import sys

app = FastAPI()
server = None
logger = logging.getLogger(__name__)

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
    peers: List[Tuple[str, str, str]]

class SetClustersRequest(BaseModel):
    id: int
    content: dict 
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
    logger.info(f"API: /query called (ReqID: {query.id})")
    if server == None:
        logger.error("ServerApp not created")
        raise Exception("ServerApp not created")
    results = server.query(query.query, query.topk, query.id)
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/query_peer")
async def query_peer_endpoint(query: QueryVectorsRequest):
    logger.info(f"API: /query_peer called")
    if server == None:
        raise Exception("ServerApp not created")
    results = server.query_me(query.query, query.topk)
    logger.debug(f"Query peer results count: {len(results) if results else 0}")
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/add")
async def add_vectors_endpoint(vectors: AddVectorsRequest):
    logger.info(f"API: /add called. ReqID: {vectors.id}, Vectors: {len(vectors.content)}")
    if server == None:
        raise Exception("ServerApp not created")
    server.add_vectors_client(vectors.content, vectors.id)
    logger.info("API: /add completed")

@app.post("/receive_vectors_peer")
async def receive_vectors_peer_endpoint(vectors: AddVectorsPeerRequest):
    logger.info("API: /receive_vectors_peer called")
    if server == None:
        raise Exception("ServerApp not created")
    server.add_vectors(vectors.content, vectors.type, vectors.id)
    logger.info("vector added via peer endpoint")

@app.post("/register_peers")
async def register_peers_endpoint(peers: AddPeersRequest):
    if server == None:
        raise Exception("ServerApp not created")
    server.add_peers(peers.peers)

@app.post("/set_clusters")
async def set_clusters_endpoint(data: SetClustersRequest):
    logger.info(f"API: /set_clusters called")
    if server == None:
        raise Exception("ServerApp not created")
    data.content["meta_hnsw"] = MetaHNSW.from_serializable_dict(data.content["meta_hnsw"])
    server.set_clusters(data.content, data.id)

@app.get("/count")
async def count_endpoint():
    if server == None:
        raise Exception("ServerApp not created")
    return server.get_count_client()

@app.get("/count_peer")
async def count_peer_endpoint():
    if server == None:
        raise Exception("ServerApp not created")
    return server.get_count()

# Add gRPC URL argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBDN Server Node")
    parser.add_argument("--node-name", type=str, required=True, help="Name of the node")
    parser.add_argument("--node-url", type=str, required=True, help="URL of this node")
    parser.add_argument("--qdrant-url", type=str, required=True, help="URL of Qdrant instance")
    parser.add_argument("--node-grpc-url", type=str, required=True, help="gRPC URL of this node")
    parser.add_argument("--coordinator-url", type=str, required=True, help="URL of coordinator node")
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    parser.add_argument("--num-before-clustering", type=int, default=10000, help="Number of vectors before triggering clustering")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")
    
    args = parser.parse_args()

    # Setup Logging
    handlers = []
    if args.log_file:
        # If log file is specified, ONLY write to file (keep terminal clean)
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
    else:
        # Default to stdout if no file provided
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s',
        handlers=handlers,
        force=True 
    )

    logger.info(f"*** SERVER STARTUP: {args.node_name} ***")
    logger.info(f"Config: URL={args.node_url}, GRPC={args.node_grpc_url}, Qdrant={args.qdrant_url}")

    # Initialize ServerApp
    server = ServerApp(args.node_name, args.node_url, args.qdrant_url, args.node_grpc_url, args.coordinator_url, args.replicas, num_vectors_before_clustering=args.num_before_clustering)

    # --- START GRPC SERVER ---
    def serve_grpc(server_app, grpc_port):
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        p2p_pb2_grpc.add_P2PNodeServicer_to_server(P2PNodeServicer(server_app), grpc_server)
        grpc_server.add_insecure_port(f'[::]:{grpc_port}')
        logger.info(f"gRPC server started on port {grpc_port}")
        grpc_server.start()
        grpc_server.wait_for_termination()

    grpc_port = args.node_grpc_url.split(":")[-1]
    
    # Run gRPC in a background thread
    grpc_thread = threading.Thread(target=serve_grpc, args=(server, grpc_port), name="GRPC-Thread")
    grpc_thread.daemon = True
    grpc_thread.start()

    # --- START FASTAPI (Main Thread) ---
    http_port = int(args.node_url.split(":")[-1])
    logger.info(f"Starting Uvicorn HTTP server on port {http_port}")
    
    # CRITICAL FIX: log_config=None prevents Uvicorn from resetting our logging configuration
    uvicorn.run(app, host="0.0.0.0", port=http_port, log_config=None)