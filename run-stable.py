#!/usr/bin/env python3
import sys
import subprocess
import time
import os
import signal
import atexit
import shutil
import requests
from pathlib import Path
import random

# --- Configuration ---
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FASTAPI_START_PORT = 8000          # node i will listen on FASTAPI_START_PORT + i  -> 8001, 8002, ...
QDRANT_START_PORT = 6333           # qdrant i maps to 6333 + 2*(i-1)             -> 6333, 6335, ...
QDRANT_PORT_STEP = 2
VECTOR_SIZE = 384                  # must match server.py default vector_size :contentReference[oaicite:2]{index=2}
REPLICAS_PER_NODE_VEC = 1          # how many representative vectors per node (>=1)

server_processes = []

def info(msg): print(f"[run-stable] {msg}")

def cleanup():
    info("Shutting down...")
    # Stop Python servers
    for p in server_processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try: p.kill()
            except Exception: pass

    # Stop and remove Docker containers and compose file
    try:
        subprocess.run(["docker", "compose", "-f", "compose.yml", "down"],
                       check=False, capture_output=True)
    except Exception:
        pass

    # Remove generated files
    if os.path.exists("compose.yml"):
        os.remove("compose.yml")

    # Remove qdrant storage dirs
    for i in range(1, N + 1):
        shutil.rmtree(f"qdrant_storage_{i}", ignore_errors=True)

    info("Cleanup complete.")

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

# --- 1) docker-compose.yml for Qdrant ---
info(f"Generating compose.yml for {N} nodes...")
with open("compose.yml", "w") as f:
    f.write("services:\n")
    for i in range(1, N + 1):
        qdrant_http_port = QDRANT_START_PORT + (i - 1) * QDRANT_PORT_STEP
        qdrant_grpc_port = qdrant_http_port + 1
        f.write(f"  qdrant-{i}:\n")
        f.write(f"    image: qdrant/qdrant:latest\n")
        f.write(f"    container_name: qdrant-{i}\n")
        f.write(f"    ports:\n")
        f.write(f"      - \"{qdrant_http_port}:6333\"\n")
        f.write(f"      - \"{qdrant_grpc_port}:6334\"\n")
        f.write(f"    volumes:\n")
        f.write(f"      - ./qdrant_storage_{i}:/qdrant/storage:z\n")
        f.write(f"    restart: unless-stopped\n")
info("compose.yml generated.")

# --- 2) Start Qdrant containers ---
info(f"Starting {N} Qdrant containers...")
subprocess.run(["docker", "compose", "-f", "compose.yml", "up", "-d"], check=True)
info("Waiting 10s for Qdrant to initialize...")
time.sleep(10)

# --- 3) Start FastAPI servers (server.py) ---
os.makedirs("logs", exist_ok=True)
DEBUG_MODE = False
info(f"Starting {N} FastAPI servers...")

for i in range(1, N + 1):
    node_id = f"node{i}"
    fastapi_port = FASTAPI_START_PORT + i
    qdrant_http_port = QDRANT_START_PORT + (i - 1) * QDRANT_PORT_STEP
    env = os.environ.copy()
    env["DEBUG"] = "true" if DEBUG_MODE else "false"

    out = open(f"logs/{node_id}_stdout.log", "w")
    err = open(f"logs/{node_id}_stderr.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "server.py", node_id, str(fastapi_port), str(qdrant_http_port)],
        env=env,
        stdout=out,
        stderr=err,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    server_processes.append(proc)
    info(f"Started {node_id} on http://localhost:{fastapi_port} (Qdrant {qdrant_http_port}), PID {proc.pid}")

def wait_healthy(n_nodes, start_port, timeout=60):
    info(f"Waiting for {n_nodes} servers to be healthy...")
    t0 = time.time()
    ready = set()
    while len(ready) < n_nodes and (time.time() - t0) < timeout:
        for i in range(1, n_nodes + 1):
            if i in ready: continue
            port = start_port + i
            try:
                r = requests.get(f"http://localhost:{port}/", timeout=2)
                if r.status_code == 200:
                    ready.add(i)
                    info(f"  ✓ node{i} ready on :{port}")
            except requests.RequestException:
                pass
        time.sleep(1)
    return len(ready) == n_nodes

if not wait_healthy(N, FASTAPI_START_PORT):
    info("WARNING: Not all servers responded healthy; continuing anyway.")

# --- 4) Bootstrap: set representative vectors & register peers (full-mesh) ---
def make_rep_vectors(i, size=VECTOR_SIZE, k=REPLICAS_PER_NODE_VEC):
    """
    Deterministic unit-like vectors per node:
    - for node i, place a '1' at indices [(i-1 + j*7) % size] to get k vectors,
      then L2-normalize server-side (server.py does normalization). :contentReference[oaicite:3]{index=3}
    """
    vecs = []
    for j in range(k):
        v = [0.0]*size
        idx = (i-1 + j*7) % size
        v[idx] = 1.0
        vecs.append(v)
    return vecs

def set_node_vectors(port, vectors):
    r = requests.post(f"http://localhost:{port}/set_node_vectors", json=vectors, timeout=10)
    r.raise_for_status()

def register_peer(port_src, peer_id, peer_url, peer_vectors):
    payload = {"peer_id": peer_id, "peer_url": peer_url, "node_vectors": peer_vectors}
    r = requests.post(f"http://localhost:{port_src}/register_peer", json=payload, timeout=10)
    r.raise_for_status()

info("Configuring representative vectors on each node...")
node_ports = {f"node{i}": FASTAPI_START_PORT + i for i in range(1, N + 1)}
node_vecs = {nid: make_rep_vectors(i) for i, nid in enumerate(node_ports.keys(), start=1)}

# 4.1 set vectors locally on each node (POST /set_node_vectors) :contentReference[oaicite:4]{index=4}
for nid, port in node_ports.items():
    set_node_vectors(port, node_vecs[nid])
    info(f"  • {nid}: set {len(node_vecs[nid])} representative vector(s)")

# 4.2 register peers in full mesh (POST /register_peer) – requires node vectors already set :contentReference[oaicite:5]{index=5}
info("Registering peers (full-mesh)…")
for src_id, src_port in node_ports.items():
    for dst_id, dst_port in node_ports.items():
        if src_id == dst_id: continue
        try:
            register_peer(
                port_src=src_port,
                peer_id=dst_id,
                peer_url=f"http://localhost:{dst_port}",
                peer_vectors=node_vecs[dst_id]
            )
            info(f"  • {src_id} ↔ registered {dst_id}")
        except requests.HTTPError as e:
            info(f"  ! Failed to register {dst_id} on {src_id}: {e}")

# --- 5) Keep process alive until Ctrl+C (portable) ---
print("\nSystem is active with peers connected. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    info("Stopping system...")
    sys.exit(0)
