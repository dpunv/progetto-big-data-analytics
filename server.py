from click import Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from server_logic import ServerApp

app = FastAPI()


class AddVectorsRequest(BaseModel):
    id: int
    content: List[Tuple[str,List[float]]]

class QueryVectorsRequest(BaseModel):
    id: int
    query: List[List[float]]
    topk: int

class AddPeersRequest(BaseModel):
    id: int
    peers: List[Tuple[str, str]]

@app.post("/query")
async def query_endpoint(query: QueryVectorsRequest):
    results = ServerApp.query(query.query, query.topk, query.id)
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/query_peer")
async def query_peer_endpoint(query: QueryVectorsRequest):
    results = ServerApp.query_me(query.query, query.topk)
    return {
        "results": results,
        "status": "success",
        "count": len(results)
    }

@app.post("/add")
async def add_vectors_endpoint(vectors: AddVectorsRequest):
    ServerApp.add_vectors_client(vectors.content, vectors.id)

@app.post("/receive_vectors_peer")
async def receive_vectors_peer_endpoint(vectors: AddVectorsRequest):
    ServerApp.add_vectors(vectors.content)

@app.post("/register_peers")
async def register_peers_endpoint(peers: AddPeersRequest):
    ServerApp.add_peers(peers.peers)

@app.post("/set_clusters")
async def set_clusters_endpoint(data: dict):
    ServerApp.set_clusters(data)

@app.get("/notify_clustering")
async def notify_clustering_endpoint():
    ServerApp.notify_clustering()








uvicorn.run(app, host="0.0.0.0", port=8000)