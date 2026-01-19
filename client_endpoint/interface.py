import asyncio
import logging

import aiohttp_cors
from aiohttp import web


class ClientEndpoint:
    """
    HTTP Endpoint for external Clients to interact with the Server.
    Exposes /add and /query.
    """

    def __init__(self, server_instance):
        self.server = server_instance
        # Increase max request size to 100MB to handle large vector batches
        self.app = web.Application(client_max_size=100 * 1024 * 1024)
        self.runner = None
        self.site = None

        # Configure CORS
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            },
        )

        # Define routes with CORS
        cors.add(self.app.router.add_post("/add", self.handle_add))
        cors.add(self.app.router.add_post("/query", self.handle_query))

    async def start(self, ip: str, port: int):
        """
        Starts the Client Endpoint HTTP server on the specified IP and port.
        """
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, ip, port)
        await self.site.start()
        print(f"Client Endpoint listening on {ip}:{port}")

    async def stop(self):
        """
        Stops the HTTP server.
        """
        if self.runner:
            await self.runner.cleanup()

    async def _run_sync(self, func, *args):
        """
        Runs a synchronous function in a separate thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args))

    async def handle_add(self, request):
        """
        Handles '/add' requests. Accepts a JSON payload containing vectors to add.
        Expected JSON format: {"vectors": [[vector_values, payload_text], ...]}
        """
        try:
            data = await request.json()
            vectors = data.get("vectors")
            if not vectors:
                return web.json_response({"error": "No vectors provided"}, status=400)

            # Invoke server logic in executor
            await self._run_sync(self.server.receive_from_client, vectors)
            return web.json_response({"status": "ok"})
        except Exception as e:
            logging.error(f"Error in /add: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_query(self, request):
        """
        Handles '/query' requests. Accepts a JSON payload containing query vectors.
        Expected JSON format: {"vectors": [vector_values, ...], "top_k": int}
        "top_k" defaults to 10 if not specified.
        """
        try:
            data = await request.json()
            vectors = data.get("vectors")
            top_k = data.get("top_k", 10)
            
            if not vectors:
                return web.json_response({"error": "No vectors provided"}, status=400)

            results = await self._run_sync(
                self.server.query_from_client, vectors, top_k
            )

            # Serialize results
            serialized_results = []
            for res in results:
                # res is a tuple: (vector, id, payload, similarity)
                vec = res[0]

                # Helper functions to safely convert numpy types to python types for JSON
                def safe_float(v):
                    if hasattr(v, "item"):
                        return float(v.item())
                    try:
                        return float(v)
                    except TypeError:
                        if hasattr(v, "__len__") and len(v) == 1:
                            return float(v[0])
                        raise

                def safe_int(v):
                    if hasattr(v, "item"):
                        return int(v.item())
                    return int(v)

                # Convert vector to list
                if hasattr(vec, "tolist"):
                    vec_list = vec.tolist()
                else:
                    vec_list = list(vec)

                vec_list = [safe_float(x) for x in vec_list]

                serialized_results.append(
                    {
                        "vector": vec_list,
                        "id": safe_int(res[1]),
                        "payload": res[2],
                        "distance": safe_float(res[3]),
                    }
                )

            return web.json_response({"results": serialized_results})
        except Exception as e:
            logging.error(f"Error in /query: {e}")
            return web.json_response({"error": str(e)}, status=500)
