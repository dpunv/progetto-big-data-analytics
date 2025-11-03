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

# --- Configuration ---
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FASTAPI_START_PORT = 8000
QDRANT_START_PORT = 6333
QDRANT_PORT_STEP = 2

server_processes = []

def cleanup():
    print("\nShutting down servers and containers...")

    # Stop Python servers
    for proc in server_processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()

    # Stop and remove Docker containers
    subprocess.run(["docker", "compose", "-f", "compose.yml", "down"],
                   check=False, capture_output=True)

    # Remove Qdrant storage directories
    for i in range(1, N + 1):
        storage_dir = f"qdrant_storage_{i}"
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir, ignore_errors=True)

    print("Cleanup complete.")

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

# --- 1. Generate docker-compose.yml ---
print(f"Generating compose.yml for {N} nodes...")

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

print("compose.yml generated successfully.")

# --- 2. Start Docker Containers ---
print(f"Starting {N} Qdrant containers with Docker Compose...")
subprocess.run(["docker", "compose", "-f", "compose.yml", "up", "-d"], check=True)

print("Waiting for Qdrant containers to initialize (10s)...")
time.sleep(10)

# --- 3. Start Python Servers ---
print(f"Starting {N} Python servers...")

os.makedirs("logs", exist_ok=True)
DEBUG_MODE = False

for i in range(1, N + 1):
    node_id = f"node{i}"
    fastapi_port = FASTAPI_START_PORT + i
    qdrant_http_port = QDRANT_START_PORT + (i - 1) * QDRANT_PORT_STEP

    env = os.environ.copy()
    env['DEBUG'] = 'true' if DEBUG_MODE else 'false'

    with open(f"logs/{node_id}_stdout.log", "w") as out, open(f"logs/{node_id}_stderr.log", "w") as err:
        proc = subprocess.Popen(
            [sys.executable, "server.py", node_id, str(fastapi_port), str(qdrant_http_port)],
            env=env,
            stdout=out,
            stderr=err
        )
        server_processes.append(proc)
        print(f"  - Started {node_id} on port {fastapi_port} (PID: {proc.pid})")

def wait_for_servers(n_nodes, start_port, timeout=60):
    print(f"Checking {n_nodes} servers for readiness...")
    start_time = time.time()
    ready_nodes = set()

    while len(ready_nodes) < n_nodes and (time.time() - start_time) < timeout:
        for i in range(1, n_nodes + 1):
            if i in ready_nodes:
                continue
            port = start_port + i
            try:
                r = requests.get(f"http://localhost:{port}/", timeout=2)
                if r.status_code == 200:
                    ready_nodes.add(i)
                    print(f"  ✓ Server {i} ready on port {port}")
            except:
                pass
        time.sleep(1)

    if len(ready_nodes) == n_nodes:
        print("✅ All servers are up and running.")
        return True
    else:
        print(f"⚠️ Only {len(ready_nodes)}/{n_nodes} servers responded.")
        return False

wait_for_servers(N, FASTAPI_START_PORT)

print("\nSystem is active. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping system...")
    sys.exit(0)
