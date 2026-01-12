import asyncio
import pickle
import ssl
from typing import Optional, Dict, Any, cast

from aioquic.asyncio import connect, QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived, QuicEvent
from .base import BaseCommunicator

class OneShotClientProtocol(QuicConnectionProtocol):
    """
    QUIC protocol handler for a single request-response cycle.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_data = b""
        self._response_complete = asyncio.Event()
        self._stream_id: Optional[int] = None

    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            self._response_data += event.data
            if event.end_stream:
                self._response_complete.set()

    async def query(self, query: str, args: tuple) -> Any:
        """
        Sends a query and waits for the response.

        :param query: The method/query name.
        :param args: Arguments for the query.
        :return: The response from the server.
        """
        
        payload = {'method': query, 'args': args}
        data = pickle.dumps(payload)
        
        # Get next stream ID
        self._stream_id = self._quic.get_next_available_stream_id()
        
        # Send data in chunks
        chunk_size = 1024  # Size for chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            is_last = (i + chunk_size >= len(data))
            self._quic.send_stream_data(self._stream_id, chunk, end_stream=is_last)
        self.transmit()
        
        # Wait for response
        await self._response_complete.wait()
        
        # Deserialize response
        
        try:
            return pickle.loads(self._response_data)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize response: {e}")

class QUICCommunicator(BaseCommunicator):
    """
    Communicator implementation using QUIC protocol.
    """
    def __init__(self):
        self.configuration = QuicConfiguration(
            is_client=True,
            verify_mode=ssl.CERT_NONE  # Trust self-signed for internal P2P
        )

    async def send(self, ip: str, port: int, query: str, *args):
        try:
            # Connect to server
            async with connect(
                ip, 
                port, 
                configuration=self.configuration,
                create_protocol=OneShotClientProtocol,
                wait_connected=True
            ) as protocol:
                client = cast(OneShotClientProtocol, protocol)
                
                # Perform query
                try:
                    result = await client.query(query, args)
                    return {"status": 0, "error": None, "response": result}
                except Exception as e:
                    return {"status": -1, "error": str(e), "response": None}
                    
        except Exception as e:
            # Connection failed
            return {"status": -1, "error": f"Connection failed: {e}", "response": None}
