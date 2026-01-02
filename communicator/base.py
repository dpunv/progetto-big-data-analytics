from abc import ABC, abstractmethod
from typing import Callable, Any, Union

class BaseCommunicator(ABC):
    @abstractmethod
    async def send(self, ip: str, port: int, query: str, data: Union[bytes, str]):
        """
        Send data to the specified ip and port.
        
        :param ip: Target IP address
        :param port: Target port
        :param query: Endpoint name or identifier
        :param data: Data to send (bytes or file path)
        :return: Dict with {'status': int, 'error': str, 'response': Any}
        """
        pass
