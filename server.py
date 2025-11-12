from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Tuple, Dict
from server_logic import ServerApp
import argparse
from compound_types import *

app = FastAPI()
server = None

class AddVectorsRequest(BaseModel):
    id: int
    content: ListOfVectorsWithPayload

class AddVectorsPeerRequest(BaseModel):
    id: int
    content: ListOfVectorsComplete

class QueryVectorsRequest(BaseModel):
    id: int
    query: ListOfVectors
    topk: int

class AddPeersRequest(BaseModel):
    id: int
    peers: List[Tuple[str, str]]

class SetClustersRequest(BaseModel):
    id: int
    content: Dict['my_vectors': List[VectorId], 'peers_clusters': Dict[str: ListOfVectorsWithId], 'meta_hnsw': MetaHNSW]

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
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/add")
async def add_vectors_endpoint(vectors: AddVectorsRequest):
    if server == None:
        raise "ServerApp not created"
    server.add_vectors_client(vectors.content, vectors.id)

@app.post("/receive_vectors_peer")
async def receive_vectors_peer_endpoint(vectors: AddVectorsPeerRequest):
    if server == None:
        raise "ServerApp not created"
    server.add_vectors(vectors.content, vectors.id)

@app.post("/register_peers")
async def register_peers_endpoint(peers: AddPeersRequest):
    if server == None:
        raise "ServerApp not created"
    server.add_peers(peers.peers)

@app.post("/set_clusters")
async def set_clusters_endpoint(data: SetClustersRequest):
    if server == None:
        raise "ServerApp not created"
    server.set_clusters(data.content, data.id)

@app.get("/notify_clustering")
async def notify_clustering_endpoint():
    if server == None:
        raise "ServerApp not created"
    server.notify_clustering()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBDN Server Node")
    parser.add_argument("--node-name", type=str, required=True, help="Name of the node")
    parser.add_argument("--node-url", type=str, required=True, help="URL of this node")
    parser.add_argument("--qdrant-url", type=str, required=True, help="URL of Qdrant instance")
    parser.add_argument("--coordinator-url", type=str, required=True, help="URL of coordinator node")
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    parser.add_argument("--num-before-clustering", type=int, default=10000, help="Number of vectors before triggering clustering")
    

    args = parser.parse_args()
    #print(f'node name:    {args.node_name}')
    #print(f'node url:     {args.node_url}')
    #print(f'qdrant url:   {args.qdrant_url}')
    #print(f'is coord:     {args.coordinator_url}')
    #print(f'replicas:     {args.replicas}')
    #print(f'before clust: {args.num_before_clustering}')

    server = ServerApp(args.node_name, args.node_url, args.qdrant_url, args.coordinator_url, args.replicas, num_vectors_before_clustering=args.num_before_clustering)

    port = int(args.node_url.split(":")[1])

    uvicorn.run(app, host="0.0.0.0", port=port)