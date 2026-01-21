from abc import ABC, abstractmethod


class BaseEndpoint(ABC):
    """Abstract base class for network endpoints.

    Parameters:
        server_instance: The server instance associated with this endpoint.
    """

    def __init__(self, server_instance):
        self.server = server_instance

    @abstractmethod
    async def start(self, ip: str, port: int):
        """Starts the endpoint server to listen for incoming connections.

        Args:
            ip (str): The IP address to bind to.
            port (int): The port number to bind to.
        """
        pass
