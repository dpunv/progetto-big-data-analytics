import asyncio
import pickle
import ssl
from typing import Any, Optional, cast

from aioquic.asyncio import QuicConnectionProtocol, connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

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

        payload = {"method": query, "args": args}
        data = pickle.dumps(payload)

        self._stream_id = self._quic.get_next_available_stream_id()

        chunk_size = 1024  # Size for chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            is_last = i + chunk_size >= len(data)
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
            verify_mode=ssl.CERT_NONE,
        )

    async def send(self, ip: str, port: int, query: str, *args):
        try:
            async with connect(
                ip,
                port,
                configuration=self.configuration,
                create_protocol=OneShotClientProtocol,
                wait_connected=True,
            ) as protocol:
                client = cast(OneShotClientProtocol, protocol)

                try:

                    if query == "get_id":
                        result = await client.query(query, args)

                    elif query == "similarity":
                        result = await client.query(query, args)

                    elif query == "receive":
                        result = await client.query(query, args)

                    elif query == "i_am_coord":
                        result = await client.query(query, args)

                    elif query == "set_clusters":
                        result = await client.query(query, args)

                    elif query == "search_vectors_local":
                        result = await client.query(query, args)

                    elif query == "query":
                        result = await client.query(query, args)

                    elif query == "get_vector_digest":
                        result = await client.query(query, args)

                    elif query == "get_vectors_by_ids":
                        result = await client.query(query, args)

                    elif query == "get_partition_coordinator_id":
                        result = await client.query(query, args)

                    elif query == "respond_to_ping":
                        result = await client.query(query, args)

                    else:
                        return {
                            "status": -1,
                            "error": f"Unknown QUIC method: {query}",
                            "response": None,
                        }

                    if isinstance(result, dict) and "status" in result:
                        return result
                    else:
                        return {"status": 0, "error": None, "response": result}

                except Exception as e:
                    return {"status": -1, "error": str(e), "response": None}

        except Exception as e:
            return {"status": -1, "error": f"Connection failed: {e}", "response": None}
