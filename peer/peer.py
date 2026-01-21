import asyncio
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    pass


class Peer:
    """
    Peer class representing a node in the network.
    Can represent the local server (self) or a remote server.
    """

    def __init__(self, ip: str, port: int, server_instance=None, communicator=None):
        self.ip = ip
        self.port = port
        self.server = server_instance  # If set, this is the local peer
        self.communicator = communicator  # Used for remote communication

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

    def set_clusters(self, clusters, assignment, version=0):
        if self.is_local():
            return self.server.set_clusters(clusters, assignment, version=version)
        return self._remote_call("set_clusters", clusters, assignment, version)

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

    def get_topology_version(self) -> int:
        if self.is_local():
            return self.server.topology_version
        return self._remote_call("get_topology_version")

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

    def count(self) -> int:
        """Get vector count on this peer."""
        if self.is_local():
            return self.server.count()
        return self._remote_call("count")

    def get_cluster_vector_count(self, cluster_id: int) -> int:
        """Get vector count for a specific cluster on this peer."""
        if self.is_local():
            return self.server.get_cluster_vector_count(cluster_id)
        return self._remote_call("get_cluster_vector_count", cluster_id)

    def delete_vectors_by_cluster(self, cluster_id: int):
        """Delete all vectors belonging to a specific cluster."""
        if self.is_local():
            return self.server.delete_vectors_by_cluster(cluster_id)
        return self._remote_call("delete_vectors_by_cluster", cluster_id)

    def split_and_distribute_cluster(
        self, cluster_id: int, split_plan: dict, is_coordinator: bool = True
    ):
        """Execute distributed split mechanism."""
        if self.is_local():
            return self.server.split_and_distribute_cluster(
                cluster_id, split_plan, is_coordinator
            )
        return self._remote_call(
            "split_and_distribute_cluster", cluster_id, split_plan, is_coordinator
        )

    def get_clusters(self):
        if self.is_local():
            return self.server.get_clusters()
        return self._remote_call("get_clusters")

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
            resp = asyncio.run(
                self.communicator.send(self.ip, self.port, method_name, *args)
            )
        except RuntimeError:
            # Fallback for nested loops
            loop = asyncio.get_event_loop()
            resp = loop.run_until_complete(
                self.communicator.send(self.ip, self.port, method_name, *args)
            )

        if resp["status"] != 0:
            raise Exception(f"Remote call {method_name} failed: {resp['error']}")

        return resp["response"]
