"""
Utility functions and helpers.
"""
from .serialization import (
    deserialize_request,
    serialize_msgpack,
    prepare_bulk_payload
)
from .vector_ops import cosine_similarity

__all__ = [
    'deserialize_request',
    'serialize_msgpack',
    'prepare_bulk_payload',
    'cosine_similarity'
]
