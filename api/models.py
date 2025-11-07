"""
Pydantic models for FastAPI request/response validation.
Centralized location for all API data models.
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class VectorDataModel(BaseModel):
    """Model for a single vector with metadata."""
    id: str
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None


class SendVectorRequest(BaseModel):
    """Request model for sending a single vector to a peer."""
    from_node: str
    vector_data: VectorDataModel


class SendVectorsBulkRequest(BaseModel):
    """Request model for sending multiple vectors to a peer."""
    from_node: str
    vectors_data: List[VectorDataModel]


class SearchRequest(BaseModel):
    """Request model for searching vectors."""
    from_node: str
    query_vector: List[float]
    top_k: int = 5


class BroadcastRequest(BaseModel):
    """Request model for broadcasting a vector to all peers."""
    vector_data: VectorDataModel


class QueryPeerRequest(BaseModel):
    """Request model for querying a specific peer."""
    peer_id: str
    query_vector: List[float]
    top_k: int = 5


class SyncRequest(BaseModel):
    """Request model for syncing a vector to a target node."""
    vector_id: str
    target_node_id: str


class RegisterPeerRequest(BaseModel):
    """Request model for registering a peer with multiple representative vectors."""
    peer_id: str
    peer_url: str
    node_vectors: List[List[float]]  # Multiple representative vectors
