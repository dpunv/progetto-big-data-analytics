import asyncio
import pickle

from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

from utils.cert_utils import generate_self_signed_cert

from .base import BaseEndpoint


class QUICServerProtocol(QuicConnectionProtocol):
    """
    QUIC protocol handler for server-side requests.
    """

    def __init__(self, *args, **kwargs):
        self.server_instance = kwargs.pop("server_instance", None)
        super().__init__(*args, **kwargs)
        self._buffers = {}
        self.loop = asyncio.get_event_loop()

    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            if event.stream_id not in self._buffers:
                self._buffers[event.stream_id] = b""
            self._buffers[event.stream_id] += event.data

            if event.end_stream:
                data = self._buffers.pop(event.stream_id)
                asyncio.create_task(self._handle_request(event.stream_id, data))

    async def _run(self, method, *args):
        """
        Runs a synchronous server method in an executor to avoid blocking the asyncio loop.
        """
        return await self.loop.run_in_executor(None, lambda: method(*args))

    async def get_id(self, args):
        return await self._run(self.server_instance.get_id)

    async def similarity(self, args):
        vec = args[0]
        return await self._run(self.server_instance.similarity, vec)

    async def receive(self, args):
        vectors = args[0]
        status = args[1]
        await self._run(self.server_instance.receive, vectors, status)
        return None

    async def i_am_coord(self, args):
        return await self._run(self.server_instance.i_am_coord)

    async def set_clusters(self, args):
        clusters = args[0]
        assignment = args[1]
        version = args[2] if len(args) > 2 else 0
        await self._run(self.server_instance.set_clusters, clusters, assignment, version)
        return None

    async def search_vectors_local(self, args):
        q_vecs = args[0]
        top_k = args[1]
        return await self._run(self.server_instance.search_vectors_local, q_vecs, top_k)

    async def query(self, args):
        q_vecs = args[0]
        status = args[1]
        return await self._run(self.server_instance.query, q_vecs, status)

    async def get_vector_digest(self, args):
        return await self._run(self.server_instance.get_vector_digest)

    async def get_vectors_by_ids(self, args):
        ids = args[0]
        return await self._run(self.server_instance.get_vectors_by_ids, ids)

    async def get_partition_coordinator_id(self, args):
        return await self._run(self.server_instance.get_partition_coordinator_id)

    async def respond_to_ping(self, args):
        return await self._run(self.server_instance.respond_to_ping)

    async def _handle_request(self, stream_id: int, data: bytes):
        """
        Deserializes the request, invokes the server method, and sends the response.
        """
        response_payload = {"status": 0, "error": None, "response": None}
        try:
            payload = pickle.loads(data)
            method_name = payload.get("method")
            args = payload.get("args", [])

            if method_name == "get_id":
                result = await self.get_id(args)
            elif method_name == "similarity":
                result = await self.similarity(args)
            elif method_name == "receive":
                result = await self.receive(args)
            elif method_name == "i_am_coord":
                result = await self.i_am_coord(args)
            elif method_name == "set_clusters":
                result = await self.set_clusters(args)
            elif method_name == "search_vectors_local":
                result = await self.search_vectors_local(args)
            elif method_name == "query":
                result = await self.query(args)
            elif method_name == "get_vector_digest":
                result = await self.get_vector_digest(args)
            elif method_name == "get_vectors_by_ids":
                result = await self.get_vectors_by_ids(args)
            elif method_name == "get_partition_coordinator_id":
                result = await self.get_partition_coordinator_id(args)
            elif method_name == "respond_to_ping":
                result = await self.respond_to_ping(args)
            else:
                raise AttributeError(f"Method {method_name} not found")

            response_payload["response"] = result

        except Exception as e:
            response_payload["status"] = -1
            response_payload["error"] = str(e)

        try:
            resp_data = pickle.dumps(response_payload)
            chunk_size = 1024
            for i in range(0, len(resp_data), chunk_size):
                chunk = resp_data[i : i + chunk_size]
                is_last = i + chunk_size >= len(resp_data)
                self._quic.send_stream_data(stream_id, chunk, end_stream=is_last)
            self.transmit()
        except Exception as e:
            print(f"Error sending response: {e}")


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

        generate_self_signed_cert()

        configuration = QuicConfiguration(is_client=False)
        configuration.load_cert_chain("server.crt", "server.key")

        def create_protocol(*args, **kwargs):
            return QUICServerProtocol(
                server_instance=self.server_instance, *args, **kwargs
            )

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
