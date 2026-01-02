import aiohttp
import os
import pickle
import base64
from typing import Callable, Any, Union
from .base import BaseCommunicator

class HTTPCommunicator(BaseCommunicator):
    async def send(self, ip: str, port: int, query: str, *args):
        url = f"http://{ip}:{port}/{query}"
        
        # Backward compatibility with existing HTTP endpoint that expects:
        # pickle.loads(body) -> {'args': args_list}
        payload = {'args': args}
        data = pickle.dumps(payload)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, data=data) as response:
                    text = await response.text()
                    
                    # Deserialize response from base64 + pickle
                    try:
                        resp_bytes = base64.b64decode(text)
                        resp_obj = pickle.loads(resp_bytes)
                        return {"status": 0, "error": None, "response": resp_obj}
                    except Exception as e:
                        return {"status": -1, "error": f"Deserialization failed: {e}", "response": None}
                            
            except Exception as e:
                print(f"HTTP Send failed: {e}")
                return {"status": -1, "error": str(e), "response": None}
