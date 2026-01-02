from abc import ABC, abstractmethod

class BaseEndpoint(ABC):
    def __init__(self, server_instance):
        self.server = server_instance
        
    @abstractmethod
    async def start(self, ip: str, port: int):
        """
        Start the endpoint server.
        """
        pass
