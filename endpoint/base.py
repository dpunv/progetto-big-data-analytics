from abc import ABC, abstractmethod


class BaseEndpoint(ABC):
    """
    Abstract base class for all endpoints.
    """

    def __init__(self, server_instance):
        self.server = server_instance

    @abstractmethod
    async def start(self, ip: str, port: int):
        """
        Start the endpoint server.

        :param ip: IP address to bind to.
        :param port: Port to bind to.
        """
        pass
