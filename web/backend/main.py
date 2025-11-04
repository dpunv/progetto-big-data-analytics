import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

PORT = int(os.getenv("PORT", "8080"))
app = FastAPI(title="NoSQL Topology Viewer")

# Serve frontend from repository (local development).
# In Docker this may be mounted at /app/frontend; when running here we point
# to the repo path: web/frontend relative to this file.
root = (Path(__file__).resolve().parent.parent / "frontend")
if not root.exists():
    # Fallback to the container path if present
    root = Path("/app/frontend")

@app.get("/")
async def index():
    return FileResponse(root / "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
