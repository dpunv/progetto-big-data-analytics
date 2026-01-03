
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

def main():
    parser = argparse.ArgumentParser(description="Launch a local distributed cluster")
    parser.add_argument("--nodes", type=int, default=2, help="Number of server nodes to launch (default: 2)")
    args = parser.parse_args()

    num_nodes = args.nodes
    print(f"Starting {num_nodes}-Node Cluster...")
    
    # 1. Clean previous runs (optional, but good practice if names clash)
    # Since we generate unique names anyway, maybe just ensure clean cleanup?
    # We'll use specific names for this session
    
    coordinator_info = None # (ip, port)
    
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
            print(f"Node 0 (Coordinator) starting on intra-port {server_port}, inter-port {client_port}")
        else:
            # Connect to coordinator
            cmd_args.extend(["--peers", coordinator_info])
            print(f"Node {i} starting on intra-port {server_port} (connected to {coordinator_info})")
            
        p = subprocess.Popen(cmd_args, env=env)
        processes.append(p)
        
        # Give it a moment to bind before ensuring next Loop
        time.sleep(1)

    print("\nCluster is running! Press Ctrl+C to stop.")
    
    # Wait for processes
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
