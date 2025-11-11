from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from server_logic import ServerApp

app = FastAPI()


class AddVectorsRequest(BaseModel):
    id: int
    content: List[(List[float], str)]

class QueryVectorsRequest(BaseModel):
    id: int
    query: List[List[float]]
    topk: int

class AddPeersRequest(BaseModel):
    id: int
    peers: List[(str, str)]

@app.post("/query")
async def query_endpoint(query: QueryVectorsRequest):
    pass

@app.post("/add")
async def add_vectors_endpoint(vectors: AddVectorsRequest):
    pass

@app.post("/register_peers")
async def register_peers_endpoint(peers: AddPeersRequest):
    pass




uvicorn.run(app, host="0.0.0.0", port=8000)