import asyncio
import pickle
import ssl
from typing import Optional, Dict, Any, cast

from aioquic.asyncio import connect, QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived, QuicEvent
from .base import BaseCommunicator

class OneShotClientProtocol(QuicConnectionProtocol):
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
        # Create 'payload' consistent with other communicators?
        # HTTP sends: {'args': args} and puts query in URL.
        # Here we don't have URL, so we put method in payload.
        # Wait, the server expects method dispatch.
        # Let's send a dict: {'method': query, 'args': args}
        
        payload = {'method': query, 'args': args}
        data = pickle.dumps(payload)
        
        # Get next stream ID
        self._stream_id = self._quic.get_next_available_stream_id()
        
        # Send data with FIN (we don't expect to send more)
        self._quic.send_stream_data(self._stream_id, data, end_stream=True)
        self.transmit()
        
        # Wait for response
        await self._response_complete.wait()
        
        # Deserialize response
        # Expecting similar structure to HTTP which returns result directly or wrapped?
        # HTTP returns {'status': ..., 'response': ...} wrapper from send(), 
        # but the ENDPOINT returns bytes.
        
        try:
            return pickle.loads(self._response_data)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize response: {e}")

class QUICCommunicator(BaseCommunicator):
    def __init__(self):
        self.configuration = QuicConfiguration(
            is_client=True,
            verify_mode=ssl.CERT_NONE  # Trust self-signed for internal P2P
        )

    async def send(self, ip: str, port: int, query: str, *args):
        try:
            # Connect to server
            # NOTE: aioquic connect context manager handles closing
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
                    # Result from endpoint should be the return value of function.
                    # BaseCommunicator expects: {'status': int, 'error': str, 'response': Any}
                    return {"status": 0, "error": None, "response": result}
                except Exception as e:
                    return {"status": -1, "error": str(e), "response": None}
                    
        except Exception as e:
            # Connection failed
            return {"status": -1, "error": f"Connection failed: {e}", "response": None}
