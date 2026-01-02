from .base import BaseEndpoint
from aiohttp import web
import pickle
import base64
import asyncio

class HTTPEndpoint(BaseEndpoint):
    async def start(self, ip: str, port: int):
        self.app = web.Application()
        # Ensure max client size is large enough for vectors
        # 100MB limit (default is usually small)
        self.app = web.Application(client_max_size=1024*1024*100)
        
        self.app.router.add_post('/{query}', self.handle_request)
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, ip, port)
        await self.site.start()
        print(f"HTTP Endpoint started on {ip}:{port}")
        
    async def handle_request(self, request):
        method_name = request.match_info.get('query', '')
        
        try:
            # Read body bytes
            body = await request.read()
            
            # Deserialize args
            # Expecting direct pickle bytes from the client request body
            try:
                payload = pickle.loads(body)
                args = payload.get("args", [])
            except Exception as e:
                return web.Response(status=400, text=f"Deserialization error: {str(e)}")

            # Dispatch to server
            if not hasattr(self.server, method_name):
                return web.Response(status=404, text=f"Method {method_name} not found")
            
            method = getattr(self.server, method_name)
            
            # Run method
            # Server methods are synchronous, so we run them directly.
            # However, if they are long running, they might block the event loop.
            # Ideally run in executor.
            loop = asyncio.get_event_loop()
            
            # If args is a tuple/list, unpack
            if isinstance(args, (list, tuple)):
                result = await loop.run_in_executor(None, lambda: method(*args))
            else:
                result = await loop.run_in_executor(None, lambda: method(args))
            
            # Serialize response
            # Pickle + Base64
            # We use Base64 because the communicator currently reads .text()
            resp_bytes = pickle.dumps(result)
            resp_b64 = base64.b64encode(resp_bytes).decode('utf-8')
            
            return web.Response(text=resp_b64)
            
        except Exception as e:
            print(f"Error handling request {method_name}: {e}")
            return web.Response(status=500, text=str(e))

    async def stop(self):
        if hasattr(self, 'runner'):
            await self.runner.cleanup()
