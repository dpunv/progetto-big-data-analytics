"""
FastAPI routers organized by functional domain.
"""
from .peer_router import router as peer_router
from .vector_router import router as vector_router
from .search_router import router as search_router
from .admin_router import router as admin_router

__all__ = [
    'peer_router',
    'vector_router',
    'search_router',
    'admin_router'
]
