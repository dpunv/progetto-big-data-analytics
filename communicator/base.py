from abc import ABC, abstractmethod


class BaseCommunicator(ABC):
    """Abstract base class for all network communicators.

    Provides a unified interface for sending requests to remote peers,
    regardless of the underlying protocol (HTTP, gRPC, QUIC).
    """

    @abstractmethod
    async def send(self, ip: str, port: int, query: str, *args):
        """Sends data to the specified destination.

        Args:
            ip (str): Target IP address.
            port (int): Target port.
            query (str): Endpoint name or identifier (e.g., method name).
            *args: Arguments to pass to the remote method.

        Returns:
            dict: A dictionary containing:
                - 'status' (int): 0 for success, non-zero for error.
                - 'error' (str): Error description if status != 0.
                - 'response' (Any): The return value from the remote method.
        """
        pass
