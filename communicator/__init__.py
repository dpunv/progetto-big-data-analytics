from .http_comm import HTTPCommunicator
from .grpc_comm import GRPCCommunicator
from .quic_comm import QUICCommunicator
from typing import Union

class Communicator:
    def __init__(self, comm_type: str):
        """
        Initialize the Communicator with a specific protocol type.
        
        :param comm_type: "HTTP", "GRPC", "QUIC"
        """
        self.comm_type = comm_type.upper()
        if self.comm_type == "HTTP":
            self._impl = HTTPCommunicator()
        elif self.comm_type == "GRPC":
            self._impl = GRPCCommunicator()
        elif self.comm_type == "QUIC":
            self._impl = QUICCommunicator()
        else:
            raise ValueError(f"Unknown communicator type: {comm_type}")

    async def send(self, ip: str, port: int, query: str, *args):
        """
        Send data using the underlying protocol implementation.
        """
        return await self._impl.send(ip, port, query, *args)
