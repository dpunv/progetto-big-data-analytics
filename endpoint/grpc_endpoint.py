import asyncio
import pickle

import grpc

from protos import p2p_pb2, p2p_pb2_grpc

from .base import BaseEndpoint


class P2PNodeServicer(p2p_pb2_grpc.P2PNodeServicer):
    """
    gRPC Servicer implementation. Handles incoming gRPC requests by delegating to the server instance.
    """

    def __init__(self, server_instance):
        self.server = server_instance
        self.loop = asyncio.get_event_loop()

    async def _run(self, method, *args):
        """
        Runs a synchronous server method in an executor to avoid blocking the asyncio loop.
        """
        return await self.loop.run_in_executor(None, lambda: method(*args))

    async def GetId(self, request, context):
        try:
            res = await self._run(self.server.get_id)
            return p2p_pb2.IdResponse(id=res)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Similarity(self, request, context):
        try:
            vec = list(request.values)
            res = await self._run(self.server.similarity, vec)
            return p2p_pb2.SimilarityResponse(score=res)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Receive(self, request, context):
        try:
            vectors = []
            for v in request.vectors:
                dests = frozenset(v.destinations) if v.destinations else frozenset()
                ver = (v.timestamp, v.version_node_id)

                vec_tuple = (list(v.values), v.id, v.payload, v.cluster_id, ver, dests)
                vectors.append(vec_tuple)

            await self._run(self.server.receive, vectors, request.status)
            return p2p_pb2.Empty()
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def IAmCoord(self, request, context):
        try:
            res = await self._run(self.server.i_am_coord)
            return p2p_pb2.BoolResponse(value=res)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def SetClusters(self, request, context):
        try:
            clusters = {}
            for cid, cdata in request.clusters.items():
                members = []
                for m in cdata.members:
                    dests = frozenset(m.destinations) if m.destinations else frozenset()
                    ver = (m.timestamp, m.version_node_id)
                    mt = (list(m.values), m.id, m.payload, m.cluster_id, ver, dests)
                    members.append(mt)

                clusters[cid] = {"center": list(cdata.center), "members": members}

            assignment = pickle.loads(request.assignment_pickle)

            await self._run(self.server.set_clusters, clusters, assignment)
            return p2p_pb2.Empty()
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def SearchVectorsLocal(self, request, context):
        try:
            q_vecs = []
            for v in request.vectors:
                q_vecs.append((list(v.values), v.id))

            results = await self._run(
                self.server.search_vectors_local, q_vecs, request.top_k
            )

            resp_items = []
            for r in results:
                resp_items.append(
                    p2p_pb2.SearchResultItem(
                        vector=r[0], id=r[1], payload=r[2], similarity=r[3]
                    )
                )
            return p2p_pb2.SearchResponse(results=resp_items)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Query(self, request, context):
        try:
            q_vecs = []
            for v in request.vectors:
                q_vecs.append((list(v.values), v.id))

            results = await self._run(self.server.query, q_vecs, request.status)

            if results is None:
                results = []

            resp_items = []
            for r in results:
                resp_items.append(
                    p2p_pb2.SearchResultItem(
                        vector=r[0], id=r[1], payload=r[2], similarity=r[3]
                    )
                )
            return p2p_pb2.QueryResponse(results=resp_items)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetVectorDigest(self, request, context):
        try:
            digest = await self._run(self.server.get_vector_digest)

            proto_digest = {}
            for vid, ver in digest.items():
                proto_digest[vid] = p2p_pb2.VersionInfo(
                    timestamp=ver[0], node_id=ver[1]
                )

            return p2p_pb2.DigestResponse(digest=proto_digest)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetVectorsByIds(self, request, context):
        try:
            ids = list(request.ids)
            vecs = await self._run(self.server.get_vectors_by_ids, ids)

            proto_vecs = []
            for v in vecs:
                dests = list(v[5]) if len(v) > 5 and v[5] else []
                ver = v[4] if len(v) > 4 else (0.0, 0)

                vc = p2p_pb2.VectorComplete(
                    values=list(v[0]),
                    id=v[1],
                    payload=v[2],
                    cluster_id=v[3],
                    timestamp=ver[0],
                    version_node_id=ver[1],
                    destinations=dests,
                )
                proto_vecs.append(vc)

            return p2p_pb2.VectorList(vectors=proto_vecs)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetPartitionCoordinatorId(self, request, context):
        try:
            res = await self._run(self.server.get_partition_coordinator_id)
            rid = res if res is not None else -1
            return p2p_pb2.IdResponse(id=rid)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RespondToPing(self, request, context):
        try:
            res = await self._run(self.server.respond_to_ping)
            return p2p_pb2.BoolResponse(value=res)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class GRPCEndpoint(BaseEndpoint):
    async def start(self, ip: str, port: int):
        self.server_grpc = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ]
        )
        p2p_pb2_grpc.add_P2PNodeServicer_to_server(
            P2PNodeServicer(self.server), self.server_grpc
        )

        listen_addr = f"[::]:{port}"
        if ip != "0.0.0.0":
            listen_addr = f"{ip}:{port}"

        self.server_grpc.add_insecure_port(listen_addr)
        print(f"GRPC Endpoint starting on {listen_addr}")
        await self.server_grpc.start()

    async def stop(self):
        if hasattr(self, "server_grpc"):
            await self.server_grpc.stop(5)
