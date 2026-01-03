from aiohttp import web
import asyncio
import json
import logging

import aiohttp_cors

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
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        
        # Define routes with CORS
        cors.add(self.app.router.add_post('/add', self.handle_add))
        cors.add(self.app.router.add_post('/query', self.handle_query))
    
    async def start(self, ip: str, port: int):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, ip, port)
        await self.site.start()
        print(f"Client Endpoint listening on {ip}:{port}")
        
    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
            
    async def _run_sync(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args))

    async def handle_add(self, request):
        try:
            data = await request.json()
            # Expected format: {"vectors": [[vector_data, payload_text], ...]}
            # Client.py sends: list of (embedding, text) tuples.
            # But wait, existing client.py uses server.receive_from_client directly with [(emb, text), ...].
            # Let's assume the JSON body will be {"vectors": [[emb, text], ...]}
            
            vectors = data.get('vectors')
            if not vectors:
                 return web.json_response({'error': "No vectors provided"}, status=400)
            
            # Convert list of lists to list of tuples/appropriate types if needed?
            # server.receive_from_client expects list of (vector, payload)
            # JSON arrays become lists in Python. That's fine.
            
            await self._run_sync(self.server.receive_from_client, vectors)
            return web.json_response({'status': 'ok'})
        except Exception as e:
            logging.error(f"Error in /add: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def handle_query(self, request):
        try:
            data = await request.json()
            # Expected format: {"vectors": [vector_data, ...]} or just one?
            # Client.py uses list of vectors for query_from_client.
            
            vectors = data.get('vectors')
            top_k = data.get('top_k', 10) # Default to 10 if not specified
            if not vectors:
                return web.json_response({'error': "No vectors provided"}, status=400)
                
            results = await self._run_sync(self.server.query_from_client, vectors, top_k)
            
            # We need to serialize numpy arrays if present in 'v'
            serialized_results = []
            for res in results:
                # res = (vector, id, payload, similarity)
                vec = res[0]
                def safe_float(v):
                    if hasattr(v, 'item'):
                        return float(v.item())
                    try:
                        return float(v)
                    except TypeError:
                         # Fallback for array-like of size 1
                         if hasattr(v, '__len__') and len(v) == 1:
                             return float(v[0])
                         raise

                def safe_int(v):
                    if hasattr(v, 'item'):
                        return int(v.item())
                    return int(v)

                # Ensure vector is list of python floats
                if hasattr(vec, 'tolist'):
                    vec_list = vec.tolist()
                else:
                    vec_list = list(vec)
                    
                vec_list = [safe_float(x) for x in vec_list]

                serialized_results.append({
                    'vector': vec_list,
                    'id': safe_int(res[1]),
                    'payload': res[2],
                    'distance': safe_float(res[3])
                })
            
            return web.json_response({'results': serialized_results})
        except Exception as e:
            logging.error(f"Error in /query: {e}")
            return web.json_response({'error': str(e)}, status=500)
