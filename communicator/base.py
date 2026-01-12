from abc import ABC, abstractmethod


class BaseCommunicator(ABC):
    """
    Abstract base class for all communicators.
    """

    @abstractmethod
    async def send(self, ip: str, port: int, query: str, *args):
        """
        Send data to the specified ip and port.

        :param ip: Target IP address
        :param port: Target port
        :param query: Endpoint name or identifier
        :param args: Arguments to pass to the remote method
        :return: Dict with {'status': int, 'error': str, 'response': Any}
        """
        pass
