from .http_endpoint import HTTPEndpoint
from .grpc_endpoint import GRPCEndpoint

class Endpoint:
    def __init__(self, endpoint_type: str, server_instance):
        """
        Initialize the Endpoint with a specific type.
        
        :param endpoint_type: "HTTP"
        :param server_instance: The server instance to handle requests.
        """
        self.endpoint_type = endpoint_type.upper()
        if self.endpoint_type == "HTTP":
            self._impl = HTTPEndpoint(server_instance)
        elif self.endpoint_type == "GRPC":
            self._impl = GRPCEndpoint(server_instance)
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")

    async def start(self, ip: str, port: int):
        """
        Start the underlying endpoint implementation.
        """
        return await self._impl.start(ip, port)

    async def stop(self):
        """
        Stop the underlying endpoint implementation.
        """
        await self._impl.stop()
