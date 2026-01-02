from .base import BaseEndpoint
import asyncio
import pickle
import os
import ssl
from typing import Optional, cast

from aioquic.asyncio import serve, QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived, QuicEvent
from utils.cert_utils import generate_self_signed_cert

class QUICServerProtocol(QuicConnectionProtocol):
    """
    QUIC protocol handler for server-side requests.
    """
    def __init__(self, *args, **kwargs):
        # Extract server_instance from kwargs if passed (hacky way because aioquic factory)
        self.server_instance = kwargs.pop('server_instance', None)
        super().__init__(*args, **kwargs)
        self._buffers = {} # stream_id -> bytes

    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            if event.stream_id not in self._buffers:
                self._buffers[event.stream_id] = b""
            self._buffers[event.stream_id] += event.data
            
            if event.end_stream:
                data = self._buffers.pop(event.stream_id)
                self._handle_request(event.stream_id, data)

    def _handle_request(self, stream_id: int, data: bytes):
        """
        Deserializes the request, invokes the server method, and sends the response.
        """
        try:
            payload = pickle.loads(data)
            method_name = payload.get('method')
            args = payload.get('args', [])
            
            # Dispatch to server
            if not hasattr(self.server_instance, method_name):
                 raise AttributeError(f"Method {method_name} not found")
            
            method = getattr(self.server_instance, method_name)
            
            # Execute the method
            try:
                if isinstance(args, (list, tuple)):
                    result = method(*args)
                else:
                    result = method(args)
            except Exception as e:
                # print(f"Error executing method {method_name}: {e}")
                result = str(e) 
            
            # Send response
            resp_data = pickle.dumps(result)
            self._quic.send_stream_data(stream_id, resp_data, end_stream=True)
            self.transmit()
            
        except Exception as e:
            print(f"QUIC Server Error: {e}")

class QUICEndpoint(BaseEndpoint):
    """
    Endpoint implementation using QUIC protocol.
    """
    def __init__(self, server_instance):
        super().__init__(server_instance)
        self.server_instance = server_instance
        self._server = None
        self._loop = None

    async def start(self, ip: str, port: int):
        self._loop = asyncio.get_running_loop()
        
        # Ensure certs exist
        generate_self_signed_cert()
        
        configuration = QuicConfiguration(is_client=False)
        configuration.load_cert_chain("server.crt", "server.key")
        
        # Factory for protocol
        def create_protocol(*args, **kwargs):
            return QUICServerProtocol(server_instance=self.server_instance, *args, **kwargs)

        self._server = await serve(
            ip,
            port,
            configuration=configuration,
            create_protocol=create_protocol,
        )
        print(f"QUIC Endpoint started on {ip}:{port}")

    async def stop(self):
        if self._server:
            self._server.close()
            # Try to wait for close if possible, or just pass
            try:
                 pass
            except:
                pass
