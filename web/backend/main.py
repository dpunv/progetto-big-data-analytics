import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

PORT = int(os.getenv("PORT", "8080"))
app = FastAPI(title="NoSQL Topology Viewer")

# Serve frontend
root = Path("/app/frontend")
app.mount("/static", StaticFiles(directory=root, html=False), name="static")

@app.get("/")
async def index():
    return FileResponse(root / "index.html")

@app.get("/healthz")
def healthz():
    return {"status":"ok"}
