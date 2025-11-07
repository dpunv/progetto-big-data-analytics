"""
Utility functions and helpers.
"""
from .serialization import deserialize_request
from .vector_ops import cosine_similarity

__all__ = [
    'deserialize_request',
    'cosine_similarity'
]
