#!/usr/bin/env python3
import sys
import subprocess
import time
import os
import signal
import atexit
import shutil
from pathlib import Path
import requests  # Add this import

# --- Configuration ---
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FASTAPI_START_PORT = 8000
QDRANT_START_PORT = 6333
QDRANT_PORT_STEP = 2

# --- PIDs List ---
server_processes = []

# --- Cleanup Function ---
def cleanup():
    print("\nShutting down...")
    
    # Kill all background server processes
    if server_processes:
        print(f"Stopping {len(server_processes)} Python servers...")
        for proc in server_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
    
    # Stop and remove Docker containers
    print("Stopping Docker containers...")
    try:
        subprocess.run(["docker", "compose", "-f", "compose.yml", "down"], 
                      check=False, capture_output=True)
    except:
        pass
    
    # Clean up generated files
    if os.path.exists("compose.yml"):
        os.remove("compose.yml")
    
    # Remove Qdrant storage directories
    for i in range(1, N + 1):
        storage_dir = f"qdrant_storage_{i}"
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir, ignore_errors=True)
    
    print("Cleanup complete.")

# Register cleanup handlers
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
print(f"Starting {N} Qdrant databases with Docker Compose...")
subprocess.run(["docker", "compose", "-f", "compose.yml", "up", "-d"], check=True)

print("Waiting for databases to initialize (10s)...")
time.sleep(10)

# --- 3. Start Python Servers in Background ---
print(f"Starting {N} Python servers...")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

for i in range(1, N + 1):
    node_id = f"node{i}"
    fastapi_port = FASTAPI_START_PORT + i
    qdrant_http_port = QDRANT_START_PORT + (i - 1) * QDRANT_PORT_STEP
    
    # Start server
    proc = subprocess.Popen(
        [sys.executable, "server.py", node_id, str(fastapi_port), str(qdrant_http_port)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    server_processes.append(proc)
    
    print(f"Started server '{node_id}' on port {fastapi_port} (Qdrant: {qdrant_http_port}, PID: {proc.pid})")
    print(f"  Logs: logs/{node_id}_stdout.log, logs/{node_id}_stderr.log")

print("Waiting for Python servers to start (5s)...")
time.sleep(5)

def wait_for_servers(n_nodes, start_port, timeout=120):  # INCREASED: 60 → 120 seconds
    """
    Wait for all Python servers to be ready by checking their health endpoints.
    
    Args:
        n_nodes: Number of nodes to wait for
        start_port: Starting port number (FASTAPI_START_PORT)
        timeout: Maximum time to wait in seconds
    
    Returns:
        bool: True if all servers are ready, False if timeout
    """
    print(f"Waiting for {n_nodes} servers to be ready...")
    start_time = time.time()
    ready_nodes = set()
    
    # NEW: Track consecutive failures to detect stuck servers
    failure_counts = {i: 0 for i in range(1, n_nodes + 1)}
    max_consecutive_failures = 10
    
    while len(ready_nodes) < n_nodes:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"\n❌ Timeout after {elapsed:.1f}s! Only {len(ready_nodes)}/{n_nodes} servers ready.")
            print(f"Missing servers: {[f'node{i}' for i in range(1, n_nodes + 1) if i not in ready_nodes]}")
            return False
        
        # Check all nodes in parallel-like manner (don't wait 2s per node)
        for i in range(1, n_nodes + 1):
            if i in ready_nodes:
                continue
                
            port = start_port + i
            try:
                # REDUCED: 2s → 0.5s timeout (faster checks)
                response = requests.get(f"http://localhost:{port}/", timeout=0.5)
                if response.status_code == 200:
                    ready_nodes.add(i)
                    failure_counts[i] = 0  # Reset failure count
                    print(f"  ✓ node{i} (port {port}) ready [{len(ready_nodes)}/{n_nodes}] ({elapsed:.1f}s)")
                else:
                    failure_counts[i] += 1
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
                failure_counts[i] += 1
                
            # NEW: Detect stuck servers early
            if failure_counts[i] > max_consecutive_failures:
                print(f"  ⚠️  node{i} (port {port}) not responding after {failure_counts[i]} attempts")
                print(f"     Check logs/node{i}_stdout.log and logs/node{i}_stderr.log")
        
        if len(ready_nodes) < n_nodes:
            # REDUCED: 1s → 0.3s sleep (check more frequently)
            time.sleep(0.3)
    
    print(f"\n✅ All {n_nodes} servers ready in {time.time() - start_time:.1f}s!")
    return True

# Replace the simple sleep with a proper health check
if not wait_for_servers(N, FASTAPI_START_PORT, timeout=120):  # INCREASED: 60 → 120s
    print("\n" + "="*60)
    print("ERROR: Not all servers started successfully!")
    print("="*60)
    print("\nDiagnostic steps:")
    print("1. Check if ports are already in use:")
    for i in range(1, N + 1):
        port = FASTAPI_START_PORT + i
        print(f"   netstat -ano | findstr :{port}")
    
    print("\n2. Check server logs:")
    for i in range(1, N + 1):
        node_id = f"node{i}"
        stdout_log = f"logs/{node_id}_stdout.log"
        stderr_log = f"logs/{node_id}_stderr.log"
        
        if os.path.exists(stdout_log):
            print(f"\n   --- {stdout_log} (last 20 lines) ---")
            with open(stdout_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                print(''.join(lines[-20:]))
        
        if os.path.exists(stderr_log):
            print(f"\n   --- {stderr_log} (last 20 lines) ---")
            with open(stderr_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                print(''.join(lines[-20:]))
    
    print("\n3. Check if Qdrant containers are running:")
    subprocess.run(["docker", "ps", "-a"], check=False)
    
    sys.exit(1)

# --- 4. Run Main Application ---
print("=" * 41)
print(f"Running main application (qdrant_app.py) for {N} nodes...")
print("=" * 41)

result = subprocess.run([sys.executable, "qdrant_app.py", str(N)])

print("=" * 41)
print("Application finished.")
print("=" * 41)

sys.exit(result.returncode)