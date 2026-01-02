from abc import ABC, abstractmethod
from typing import Callable, Any, Union

class BaseCommunicator(ABC):
    @abstractmethod
    async def send(self, ip: str, port: int, query: str, *args):
        """
        Send data to the specified ip and port.
        
        :param ip: Target IP address
        :param port: Target port
        :param query: Endpoint name or identifier
        :param args: arguments to pass to the remote method
        :return: Dict with {'status': int, 'error': str, 'response': Any}
        """
        pass
