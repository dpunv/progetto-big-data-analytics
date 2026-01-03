
import subprocess
import time
import sys
import os
import requests
import atexit
import argparse
import socket
import json

# Configuration
QDRANT_IMAGE = "qdrant/qdrant:latest"

processes = []
containers = []

def get_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def cleanup():
    print("\nShutting down...")
    
    # 1. Stop Python servers
    for p in processes:
        if p.poll() is None:
            p.terminate()
    
    # Wait for termination
    time.sleep(1)
    for p in processes:
         if p.poll() is None:
            p.kill()
    
    # 2. Stop Docker containers
    if containers:
        print(f"Stopping {len(containers)} Docker containers...")
        # Join container names
        cmd = f"docker rm -f {' '.join(containers)}"
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)
    
    print("Cleanup complete.")

atexit.register(cleanup)

def wait_for_qdrant(port):
    url = f"http://localhost:{port}/readyz"
    for _ in range(30):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def get_docker_port(container_name, internal_port=6333):
    """Get the host port mapped to the container's internal port"""
    cmd = f"docker port {container_name} {internal_port}"
    try:
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        # Output format is usually "0.0.0.0:32768" or similar
        if result:
            return int(result.split(":")[-1])
    except:
        pass
    return None

    # ... (previous code)

def load_vectors(port, num_vectors=50000):
    print(f"\nLoading {num_vectors} vectors from embeddings.json...")
    try:
        if not os.path.exists('embeddings.json'):
             print("Error: embeddings.json not found. Skipping data load.")
             return

        with open('embeddings.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading embeddings.json: {e}")
        return

    if len(data) < num_vectors:
        print(f"Warning: requested {num_vectors} but file only has {len(data)}. Loading all.")
        num_vectors = len(data)

    # Prepare data: list of [embedding, text]
    # interface.py expects {"vectors": [[emb, text], ...]}
    batch_data = [(d['embedding'], d['text']) for d in data[:num_vectors]]
    
    batch_size = 500
    total = len(batch_data)
    url = f"http://localhost:{port}/add"
    
    print(f"Target URL: {url}")
    
    start_time = time.time()
    for i in range(0, total, batch_size):
        batch = batch_data[i : i + batch_size]
        try:
            resp = requests.post(url, json={"vectors": batch})
            if resp.status_code != 200:
                print(f"\nFailed to upload batch {i}: {resp.status_code} - {resp.text}")
            else:
                print(f"\rProgress: {min(i + batch_size, total)}/{total} vectors loaded...", end="")
        except Exception as e:
            print(f"\nConnection error during upload: {e}")
            break
            
    print(f"\nData loading finished in {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Launch a local distributed cluster")
    parser.add_argument("--nodes", type=int, default=2, help="Number of server nodes to launch (default: 2)")
    args = parser.parse_args()

    num_nodes = args.nodes
    print(f"Starting {num_nodes}-Node Cluster...")
    
    coordinator_info = None # (ip, port)
    coordinator_client_port = None
    
    # 3. Start Nodes
    env = os.environ.copy()
    
    for i in range(num_nodes):
        print(f"\n--- Setting up Node {i} ---")
        
        # A. Start Qdrant Container with Random Port
        c_name = f"qdrant-node-{i}-{int(time.time())}"
        # -P publishes all exposed ports to random host ports
        cmd_docker = f"docker run -d --name {c_name} -P {QDRANT_IMAGE}"
        subprocess.run(cmd_docker, shell=True, check=True)
        containers.append(c_name)
        
        # Get assigned HTTP port
        qdrant_port = None
        for _ in range(10):
            qdrant_port = get_docker_port(c_name, 6333)
            if qdrant_port:
                break
            time.sleep(0.5)
            
        if not qdrant_port:
            print(f"Failed to get port for {c_name}")
            sys.exit(1)
            
        print(f"Qdrant {i} running on localhost:{qdrant_port}")
        
        if not wait_for_qdrant(qdrant_port):
            print(f"Qdrant {i} failed to become ready.")
            sys.exit(1)
            
        # B. Start Python Server with Free Port
        server_port = get_free_port()
        
        cmd_args = [
            sys.executable, "server.py",
            "--id", str(i),
            "--intra-port", str(server_port),
            "--qdrant", f"http://localhost:{qdrant_port}",
            "--endpoint", "HTTP"
        ]
        
        if i == 0:
            cmd_args.append("--coordinator")
            
            # Allocate dedicated port for client interface (inter-port)
            client_port = get_free_port()
            while client_port == server_port:
                time.sleep(0.1)
                client_port = get_free_port()
                
            cmd_args.extend(["--inter-port", str(client_port)])
            
            coordinator_info = f"127.0.0.1:{server_port}"
            coordinator_client_port = client_port
            print(f"Node 0 (Coordinator) starting on intra-port {server_port}, inter-port {client_port}")
        else:
            # Connect to coordinator
            cmd_args.extend(["--peers", coordinator_info])
            print(f"Node {i} starting on intra-port {server_port} (connected to {coordinator_info})")
            
        p = subprocess.Popen(cmd_args, env=env)
        processes.append(p)
        
        # Give it a moment to bind before ensuring next Loop
        time.sleep(1)

    print("\nCluster is running! Loading data...")
    
    # Load Initial Data
    if coordinator_client_port:
        # Give the server a second to fully start the HTTP endpoint
        time.sleep(2)
        
        # Start Web Client
        print("\nStarting Web Client...")
        web_client_port = get_free_port()
        while web_client_port == coordinator_client_port:
             web_client_port = get_free_port()
        
        # Write config.js with the server URL for the web client to use
        config_path = os.path.join("web_client", "config.js")
        with open(config_path, 'w') as f:
            f.write(f"// Auto-generated by startup.py\n")
            f.write(f"window.SERVER_URL = 'http://localhost:{coordinator_client_port}';\n")
        print(f"Web Client configured to connect to server at http://localhost:{coordinator_client_port}")
             
        # Serve the web_client directory
        cmd_web = [sys.executable, "-m", "http.server", str(web_client_port)]
        p_web = subprocess.Popen(cmd_web, cwd="web_client", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p_web)
        
        print(f"Web Client running on http://localhost:{web_client_port}")
        
        load_vectors(coordinator_client_port, 50000)
    
    print("\nSetup Complete. Press Ctrl+C to stop.")
    
    # Wait for processes
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
