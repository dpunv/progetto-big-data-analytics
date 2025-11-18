import json
import time
import subprocess
import sys
import os
import shutil
import signal
import atexit
import requests
import qdrant_module

server_processes = []

def write_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

def write_docker_compose(qdrant_ports):
    print(f"Generating compose.yml for {len(qdrant_ports)} nodes...")

    with open("compose.yml", "w") as f:
        f.write("services:\n")
        
        for i, port in enumerate(qdrant_ports):
            qdrant_http_port = port
            qdrant_grpc_port = qdrant_http_port + 1
            
            f.write(f"  qdrant-{i+1}:\n")
            f.write(f"    image: qdrant/qdrant:latest\n")
            f.write(f"    container_name: qdrant-{i+1}\n")
            f.write(f"    ports:\n")
            f.write(f"      - \"{qdrant_http_port}:6333\"\n")
            f.write(f"      - \"{qdrant_grpc_port}:6334\"\n")
            f.write(f"    environment:\n")
            f.write(f"      - QDRANT__STORAGE__STRICT_MODE=false\n")
            f.write(f"    volumes:\n")
            f.write(f"      - ./qdrant_storage_{i+1}:/qdrant/storage\n")
            f.write(f"    restart: unless-stopped\n")

    print("compose.yml generated successfully.")

def launch_docker():
    print(f"Starting Qdrant databases with Docker Compose...")
    subprocess.run(["docker", "compose", "-f", "compose.yml", "up", "-d"], check=True)

    print("Waiting for databases to initialize (10s)...")
    time.sleep(10)

def launch_and_wait_for_qdrant(qdrant_ports, timeout=60):
    print(f"Starting {len(qdrant_ports)} Qdrant databases with Docker Compose...")
    try:
        subprocess.run(["docker", "compose", "-f", "compose.yml", "up", "-d"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'avvio di Docker Compose: {e}")
        return False

    print("Waiting for Qdrant nodes to be ready (checking /readyz)...")
    start_time = time.time()
    ready_nodes = set()
    
    while len(ready_nodes) < len(qdrant_ports):
        if time.time() - start_time > timeout:
            print(f"Timeout! Only {len(ready_nodes)}/{len(qdrant_ports)} Qdrant nodes are ready.")
            return False

        for port in qdrant_ports:
            if port in ready_nodes:
                continue
            url = f"http://localhost:{port}/readyz"
            print(f"Checking Qdrant node on port {port} at {url}...")
            try:
                response = requests.get(url, timeout=1) 
                
                if response.status_code == 200:
                    ready_nodes.add(port)
                    print(f"  ✓ Qdrant node on port {port} is ready. ({len(ready_nodes)}/{len(qdrant_ports)})")
            except requests.exceptions.RequestException:
                pass
        if len(ready_nodes) < len(qdrant_ports):
            time.sleep(2)

    print("All Qdrant nodes are ready.")
    return True

def launch_servers(fast_api_ports, qdrant_ports, coordinator_url='http://localhost:8001', replicas=3, num_before_clustering=1000):
    for i in range(1, len(fast_api_ports) + 1):
        node_id = f"node{i}"
        fastapi_port = fast_api_ports[i-1]
        qdrant_http_port = qdrant_ports[i-1]
        
        qdrant_url = f'http://localhost:{qdrant_http_port}'

        proc = subprocess.Popen(
            [sys.executable, "server.py",
                '--node-name', node_id,
                '--node-url', f'http://localhost:{fastapi_port}',
                '--qdrant-url', qdrant_url,
                '--coordinator-url', coordinator_url,
                '--replicas', str(replicas),
                '--num-before-clustering', str(num_before_clustering)
            ],
            stdout=sys.stdout,  # MODIFIED: Redirect stdout
            stderr=sys.stderr,  # MODIFIED: Redirect stderr
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        server_processes.append(proc)
        
        qdrant_module.create_collection(qdrant_url, 'vectors', 384)
        
        print(f"Started server '{node_id}' on port {fastapi_port} (Qdrant: {qdrant_http_port}, PID: {proc.pid})")

    return server_processes

def wait_for_servers(fast_api_ports, timeout=60):
    start_time = time.time()
    ready_nodes = []
    while len(ready_nodes) < len(fast_api_ports):
        if time.time() - start_time > timeout:
            print(f"Timeout! Only {len(ready_nodes)}/{len(fast_api_ports)} servers are ready.")
            return False
        for i, port in enumerate(fast_api_ports):
            if i in ready_nodes:
                continue
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=2)
                if response.status_code == 200:
                    ready_nodes.append(i)
                    print(f"  ✓ Server on port {port} is ready ({len(ready_nodes)}/{len(fast_api_ports)})")
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
                pass
        
        if len(ready_nodes) < len(fast_api_ports):
            time.sleep(1)
    return True

def cleaning(N):
    print("\nShutting down...")
    
    if server_processes:
        print(f"Stopping {len(server_processes)} Python servers...")
        for proc in server_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()

    print("Stopping Docker containers...")
    try:
        subprocess.run(["docker", "compose", "-f", "compose.yml", "down"], 
                        check=False, capture_output=True)
    except:
        pass
    
    if os.path.exists("compose.yml"):
        os.remove("compose.yml")

    if os.path.exists("config.json"):
        os.remove("config.json")
    
    for i in range(1, N + 1):
        storage_dir = f"qdrant_storage_{i}"
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir, ignore_errors=True)

    print("Cleanup complete.")

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    FASTAPI_START_PORT = 8000
    QDRANT_START_PORT = 6333
    QDRANT_PORT_STEP = 2

    atexit.register(cleaning, N)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    fast_api_ports = [FASTAPI_START_PORT + i + 1 for i in range(N)]
    qdrant_ports = [QDRANT_START_PORT + (i * QDRANT_PORT_STEP) for i in range(N)]

    config = {
        'servers': [{'id': f'node{i+1}', 'url': f'http://localhost:{FASTAPI_START_PORT + i + 1}', 'is_coordinator': False if i != 0 else True} for i in range(N)],
        'batch_size': 256,
        'num_vectors': 65536,
        'num_before_clustering': 8_192,
        'replicas': 4
    }
    # step 1: writing configuration to json file
    write_config(config)

    # step 2: writing docker compose file
    write_docker_compose(qdrant_ports)

    if not launch_and_wait_for_qdrant(qdrant_ports):
        print("Failed to start Qdrant servers. Exiting.")
        sys.exit(1) # Esce con un codice di errore

    # step 4: launch servers
    launch_servers(fast_api_ports, qdrant_ports, replicas=config['replicas'], num_before_clustering=config['num_before_clustering'])

    # step 5: check servers health
    if not wait_for_servers(fast_api_ports):
        sys.exit()
    
    # step 6: launch client
    result = subprocess.run([sys.executable, "client.py"])

    # step 7: terminate application
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()