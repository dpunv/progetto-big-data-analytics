"""
Serialization utilities for MessagePack and JSON handling.
Provides unified interface for request/response serialization.
"""
import msgpack
from typing import Any, Dict, List
from fastapi import Request


async def deserialize_request(request: Request) -> Any:
    """
    Deserialize request body based on Content-Type header.
    Supports both MessagePack and JSON automatically.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Deserialized data (dict, list, etc.)
        
    Raises:
        ValueError: If deserialization fails
    """
    content_type = request.headers.get("Content-Type", "application/json")
    
    try:
        if content_type == "application/msgpack":
            body = await request.body()
            return msgpack.unpackb(body, raw=False)
        else:
            # Default to JSON
            return await request.json()
    except Exception as e:
        raise ValueError(f"Failed to deserialize request: {str(e)}")


def serialize_msgpack(data: Any) -> bytes:
    """
    Serialize data to MessagePack format.
    
    Args:
        data: Data to serialize (dict, list, etc.)
        
    Returns:
        Binary MessagePack data
    """
    return msgpack.packb(data, use_bin_type=True)


def prepare_bulk_payload(from_node: str, vectors_data: List[Dict]) -> bytes:
    """
    Prepare bulk vector payload with MessagePack serialization.
    
    Args:
        from_node: Source node ID
        vectors_data: List of vector data dictionaries
        
    Returns:
        Serialized MessagePack binary data
    """
    payload = {
        "from_node": from_node,
        "vectors_data": vectors_data
    }
    return serialize_msgpack(payload)
