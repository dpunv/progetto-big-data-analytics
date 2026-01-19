import asyncio

from aiohttp import web

from .base import BaseEndpoint


class HTTPEndpoint(BaseEndpoint):
    """
    Endpoint implementation using HTTP protocol.
    """



    async def start(self, ip: str, port: int):
        """
        Starts the HTTP web server.
        Configures the routes and handles the server lifecycle.
        """
        # Set max client size to 100MB to allow large vector payloads
        self.app = web.Application(client_max_size=1024 * 1024 * 100)

        # Register API endpoints
        self.app.router.add_post("/get_id", self.handle_get_id)
        self.app.router.add_post("/similarity", self.handle_similarity)
        self.app.router.add_post("/receive", self.handle_receive)
        self.app.router.add_post("/i_am_coord", self.handle_i_am_coord)
        self.app.router.add_post("/set_clusters", self.handle_set_clusters)
        self.app.router.add_post(
            "/search_vectors_local", self.handle_search_vectors_local
        )
        self.app.router.add_post("/query", self.handle_query)
        self.app.router.add_post("/get_vector_digest", self.handle_get_vector_digest)
        self.app.router.add_post("/get_vectors_by_ids", self.handle_get_vectors_by_ids)
        self.app.router.add_post(
            "/get_partition_coordinator_id", self.handle_get_partition_coordinator_id
        )
        self.app.router.add_post("/respond_to_ping", self.handle_respond_to_ping)

        # Start the app runner
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, ip, port)
        await self.site.start()
        print(f"HTTP Endpoint started on {ip}:{port}")

    async def _run_sync(self, func, *args):
        """
        Runs a synchronous function in a separate thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args))

    def json_response(self, data):
        """
        Helper method to wrap data in a standardized JSON response format.
        Structure: {"response": data}
        """
        return web.json_response({"response": data})

    async def handle_get_id(self, request):
        """
        Handles 'get_id' request. Returns the server ID.
        """
        try:
            res = await self._run_sync(self.server.get_id)
            return self.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_similarity(self, request):
        """
        Handles 'similarity' request. Calculates cosine similarity for a given vector.
        """
        try:
            data = await request.json()
            vec = data.get("vector")
            res = await self._run_sync(self.server.similarity, vec)
            return self.json_response(res)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_receive(self, request):
        """
        Handles 'receive' request. Accepts a batch of vectors (replication/gossip).
        Reconstructs vector tuples from the JSON payload.
        """
        try:
            data = await request.json()
            vectors_in = data.get("vectors")
            status = data.get("status")

            # Reconstruct list of tuples from JSON list of lists
            vectors_tuples = []
            for v_list in vectors_in:
                # v_list structure: [values, id, payload, cluster_id, [ts, nid], [dests]]
                
                # Default version and destinations
                ts = 0.0
                nid = 0
                dests = []

                # Extract version info if available
                if len(v_list) > 4:
                    if v_list[4]:
                        ts = v_list[4][0]
                        nid = v_list[4][1]

                # Extract destinations if available
                if len(v_list) > 5:
                    if v_list[5]:
                        dests = list(v_list[5])

                values = v_list[0]
                vid = v_list[1]
                payload = v_list[2]
                cid = v_list[3]

                ver_tuple = (ts, nid)
                dests_set = frozenset(dests)  # Convert back to set for internal logic

                vec_tuple = (values, vid, payload, cid, ver_tuple, dests_set)
                vectors_tuples.append(vec_tuple)

            await self._run_sync(self.server.receive, vectors_tuples, status)
            return self.json_response(None)
        except Exception as e:
            print(f"Error HTTP Receive: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_i_am_coord(self, request):
        """
        Handles 'i_am_coord' request. Checks if this server is the coordinator.
        """
        try:
            res = await self._run_sync(self.server.i_am_coord)
            return self.json_response(res)
        except Exception:
            return self.json_response(False)

    async def handle_set_clusters(self, request):
        """
        Handles 'set_clusters' request. Updates the cluster configuration.
        Reconstructs usage structs from JSON.
        """
        try:
            data = await request.json()
            clusters_raw = data.get("clusters")
            assignment_raw = data.get("assignment")

            # Reconstruct clusters dictionary
            # Keys might be strings in JSON, convert back to int
            clusters = {}
            for cid_str, cdata in clusters_raw.items():
                cid = int(cid_str)
                center = cdata["center"]
                members_raw = cdata["members"]

                members = []
                for m_list in members_raw:
                    # Reconstruct member tuple: (val, id, pl, cid, (ts, nid), dests)
                    ts = 0.0
                    nid = 0
                    dests = frozenset()

                    if len(m_list) > 4 and m_list[4]:
                        ts = m_list[4][0]
                        nid = m_list[4][1]
                    if len(m_list) > 5 and m_list[5]:
                        dests = frozenset(m_list[5])

                    mt = (m_list[0], m_list[1], m_list[2], m_list[3], (ts, nid), dests)
                    members.append(mt)

                clusters[cid] = {"center": center, "members": members}

            # Reconstruct assignment list
            assignment = []
            if assignment_raw:
                for item in assignment_raw:
                    assignment.append((item[0], item[1]))

            await self._run_sync(self.server.set_clusters, clusters, assignment)
            return self.json_response(None)
        except Exception as e:
            print(f"Error SetClusters: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_search_vectors_local(self, request):
        """
        Handles 'search_vectors_local' request. Performs KNN search locally.
        """
        try:
            data = await request.json()
            vecs_in = data.get("vectors")
            top_k = data.get("top_k")

            q_vecs = []
            for item in vecs_in:
                q_vecs.append((item["values"], item["id"]))

            results = await self._run_sync(
                self.server.search_vectors_local, q_vecs, top_k
            )
            
            # Serialize results
            json_results = []
            if results:
                for r in results:
                    # Convert numpy array to list if needed
                    v_list = r[0].tolist() if hasattr(r[0], "tolist") else list(r[0])
                    json_results.append((v_list, r[1], r[2], r[3]))

            return self.json_response(json_results)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_query(self, request):
        """
        Handles 'query' request. General purpose vector query (search).
        """
        try:
            data = await request.json()
            vecs_in = data.get("vectors")
            status = data.get("status")

            q_vecs = []
            for item in vecs_in:
                q_vecs.append((item["values"], item["id"]))

            results = await self._run_sync(self.server.query, q_vecs, status)
            
            json_results = []
            if results:
                for r in results:
                    v_list = r[0].tolist() if hasattr(r[0], "tolist") else list(r[0])
                    json_results.append((v_list, r[1], r[2], r[3]))
            return self.json_response(json_results)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_vector_digest(self, request):
        """
        Handles 'get_vector_digest'. Returns a summary of vector versions for anti-entropy.
        """
        try:
            digest = await self._run_sync(self.server.get_vector_digest)
            # Convert integer keys to strings for JSON compatibility
            json_digest = {}
            for k, v in digest.items():
                json_digest[str(k)] = v
            return self.json_response(json_digest)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_vectors_by_ids(self, request):
        """
        Handles 'get_vectors_by_ids'. Retrieves full vector data for given IDs.
        """
        try:
            data = await request.json()
            ids = data.get("ids")
            vecs = await self._run_sync(self.server.get_vectors_by_ids, ids)

            # Serialize vectors to JSON-compatible format
            json_vecs = []
            for v in vecs:
                v_list = list(v)
                if hasattr(v_list[0], "tolist"):
                    v_list[0] = v_list[0].tolist()

                # Convert sets to lists
                if len(v_list) > 5 and isinstance(v_list[5], (set, frozenset)):
                    v_list[5] = list(v_list[5])
                json_vecs.append(v_list)

            return self.json_response(json_vecs)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_partition_coordinator_id(self, request):
        """
        Returns the ID of the partition coordinator.
        """
        try:
            res = await self._run_sync(self.server.get_partition_coordinator_id)
            return self.json_response(res)
        except Exception:
            return self.json_response(-1)

    async def handle_respond_to_ping(self, request):
        """
        Responds to health check pings.
        """
        try:
            res = await self._run_sync(self.server.respond_to_ping)
            return self.json_response(res)
        except Exception:
            return self.json_response(False)

    async def stop(self):
        """
        Stops the HTTP server.
        """
        if hasattr(self, "runner"):
            await self.runner.cleanup()
