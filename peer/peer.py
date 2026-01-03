


import asyncio
import threading
from typing import List, Dict, Tuple, Optional, Any
import time
import base64
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server import Server

class Peer:
    """
    Peer class representing a node in the network.
    Can represent the local server (self) or a remote server.
    """
    def __init__(self, ip: str, port: int, server_instance=None, communicator=None):
        self.ip = ip
        self.port = port
        self.server = server_instance  # If set, this is the local peer
        self.communicator = communicator # Used for remote communication
        
    def get_id(self) -> int:
        if self.is_local():
            return self.server.get_id()
        return self._remote_call("get_id")
    
    def similarity(self, vector) -> float:
        if self.is_local():
            return self.server.similarity(vector)
        return self._remote_call("similarity", vector)
    
    def receive(self, vectors, status):
        if self.is_local():
            return self.server.receive(vectors, status)
        return self._remote_call("receive", vectors, status)
    
    def i_am_coord(self) -> bool:
        if self.is_local():
            return self.server.i_am_coord()
        return self._remote_call("i_am_coord")
    
    def set_clusters(self, clusters, assignment):
        if self.is_local():
            return self.server.set_clusters(clusters, assignment)
        return self._remote_call("set_clusters", clusters, assignment)
        
    def search_vectors_local(self, vectors, top_k):
        if self.is_local():
            return self.server.search_vectors_local(vectors, top_k)
        return self._remote_call("search_vectors_local", vectors, top_k)
    
    def query(self, vectors, status):
        if self.is_local():
            return self.server.query(vectors, status)
        return self._remote_call("query", vectors, status)
    
    def get_vector_digest(self) -> Dict[int, Tuple[float, int]]:
        if self.is_local():
            return self.server.get_vector_digest()
        return self._remote_call("get_vector_digest")
    
    def get_vectors_by_ids(self, ids: List[int]):
        if self.is_local():
            return self.server.get_vectors_by_ids(ids)
        return self._remote_call("get_vectors_by_ids", ids)
    
    def get_partition_coordinator_id(self):
        if self.is_local():
            return self.server.partition_coordinator_id
        return self._remote_call("get_partition_coordinator_id")

    def ping(self) -> bool:
        """Check if peer is reachable."""
        if self.is_local():
            return self.server.respond_to_ping()
        try:
            return self._remote_call("respond_to_ping")
        except Exception:
            return False

    def is_local(self):
        return self.server is not None

    def _remote_call(self, method_name: str, *args):
        """
        Helper to make remote calls via communicator.
        Serializes args, sends, receives response, deserializes.
        Sync wrapper around Async communicator.
        """
        if not self.communicator:
            raise Exception(f"No communicator for remote peer {self.ip}:{self.port}")

        try:
           resp = asyncio.run(self.communicator.send(self.ip, self.port, method_name, *args))
        except RuntimeError:
           # Fallback for nested loops
           loop = asyncio.get_event_loop()
           resp = loop.run_until_complete(self.communicator.send(self.ip, self.port, method_name, *args))
        
        if resp["status"] != 0:
            raise Exception(f"Remote call {method_name} failed: {resp['error']}")
            
        return resp['response']
