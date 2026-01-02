import aiohttp
import os
from typing import Callable, Any, Union
from .base import BaseCommunicator

class HTTPCommunicator(BaseCommunicator):
    async def send(self, ip: str, port: int, query: str, data: Union[bytes, str]):
        url = f"http://{ip}:{port}/{query}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, data=data) as response:
                    text = await response.text()
                    return {"status": 0, "error": None, "response": text}
                            
            except Exception as e:
                print(f"HTTP Send failed: {e}")
                return {"status": -1, "error": str(e), "response": None}
