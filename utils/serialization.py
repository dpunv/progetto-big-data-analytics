"""
Serialization utilities for FastAPI public endpoints (JSON only).
gRPC uses Protocol Buffers natively for internal P2P communication.
"""
from typing import Any
from fastapi import Request


async def deserialize_request(request: Request) -> Any:
    """
    Deserialize JSON request body for FastAPI public endpoints.
    
    NOTE: This is ONLY for public client-facing endpoints.
    Internal P2P communication uses gRPC with Protocol Buffers.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Deserialized JSON data (dict, list, etc.)
        
    Raises:
        ValueError: If deserialization fails
    """
    try:
        return await request.json()
    except Exception as e:
        raise ValueError(f"Failed to deserialize JSON request: {str(e)}")
