import argparse
import atexit
import os
import socket
import subprocess
import sys
import time

import requests

import generate_peers

QDRANT_IMAGE = "qdrant/qdrant:latest"

processes = []
containers = []


def get_free_port():
    """Finds an available TCP port on localhost.
    
    Returns:
        int: A random available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def cleanup():
    """Terminates child processes and removes Docker containers created during the session."""
    print("\nShutting down...")

    for p in processes:
        if p.poll() is None:
            p.terminate()

    time.sleep(1)
    for p in processes:
        if p.poll() is None:
            p.kill()

    if containers:
        print(f"Stopping {len(containers)} Docker containers...")
        cmd = f"docker rm -f {' '.join(containers)}"
        subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL)

    print("Cleanup complete.")


atexit.register(cleanup)


def wait_for_qdrant(port):
    url = f"http://127.0.0.1:{port}/readyz"
    for _ in range(30):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def wait_for_server_health(port):
    """Wait for the Python server to be ready serving the Client Endpoint."""
    url = f"http://127.0.0.1:{port}/health"
    print(f"Waiting for server health at {url}...")
    for _ in range(60):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                print("\nServer is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    return False


def get_docker_port(container_name, internal_port=6333):
    """Get the host port mapped to the container's internal port"""
    cmd = f"docker port {container_name} {internal_port}"
    try:
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        if result:
            return int(result.split(":")[-1])
    except:
        pass
    return None



def load_vectors(port, num_vectors=51200):
    import pandas as pd

    print(f"\nLoading {num_vectors} vectors from embeddings.parquet...")
    try:
        if not os.path.exists("embeddings.parquet"):
            print("Error: embeddings.parquet not found. Skipping data load.")
            return

        df = pd.read_parquet("embeddings.parquet")
        data = [
            {
                "embedding": (
                    row["embedding"].tolist()
                    if hasattr(row["embedding"], "tolist")
                    else list(row["embedding"])
                ),
                "text": row["sentence"],
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"Error reading embeddings.parquet: {e}")
        return

    if len(data) < num_vectors:
        print(
            f"Warning: requested {num_vectors} but file only has {len(data)}. Loading all."
        )
        num_vectors = len(data)

    batch_data = [(d["embedding"], d["text"]) for d in data[:num_vectors]]

    batch_size = 1024
    total = len(batch_data)
    url = f"http://127.0.0.1:{port}/add"

    print(f"Target URL: {url}")

    start_time = time.time()
    for i in range(0, total, batch_size):
        batch = batch_data[i : i + batch_size]
        try:
            resp = requests.post(url, json={"vectors": batch})
            if resp.status_code != 200:
                print(f"\nFailed to upload batch {i}: {resp.status_code} - {resp.text}")
            else:
                print(
                    f"\rProgress: {min(i + batch_size, total)}/{total} vectors loaded...",
                    end="",
                )
        except Exception as e:
            print(f"\nConnection error during upload: {e}")
            break

    print(f"\nData loading finished in {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Launch a local distributed cluster")
    parser.add_argument(
        "--nodes",
        type=int,
        default=2,
        help="Number of server nodes to launch (default: 2)",
    )
    args = parser.parse_args()

    num_nodes = args.nodes
    print(f"Starting {num_nodes}-Node Cluster...")

    coordinator_info = None
    coordinator_client_port = None

    env = os.environ.copy()

    peers_config = []
    node_configs = []

    print("\n--- Configuring Cluster Nodes ---")
    for i in range(num_nodes):
        c_name = f"qdrant-node-{i}-{int(time.time())}"
        cmd_docker = f"docker run -d --name {c_name} -P {QDRANT_IMAGE}"
        subprocess.run(cmd_docker, shell=True, check=True)
        containers.append(c_name)

        qdrant_port = None
        for _ in range(10):
            qdrant_port = get_docker_port(c_name, 6333)
            if qdrant_port:
                break
            time.sleep(0.5)

        if not qdrant_port:
            print(f"Failed to get port for {c_name}")
            sys.exit(1)

        if not wait_for_qdrant(qdrant_port):
            print(f"Qdrant {i} failed to become ready.")
            sys.exit(1)

        server_port = get_free_port()

        client_port = None
        if i == 0:
            client_port = get_free_port()
            while client_port == server_port:
                time.sleep(0.1)
                client_port = get_free_port()
            coordinator_client_port = client_port

        peers_config.append({"id": i, "url": "127.0.0.1", "port": server_port})

        node_configs.append(
            {
                "id": i,
                "qdrant_port": qdrant_port,
                "server_port": server_port,
                "client_port": client_port,
            }
        )

        time.sleep(0.1)

    print("\nGenerating peers.json...")
    generate_peers.create_peers_config(peers_config)

    print("\n--- Launching Servers ---")
    for config in node_configs:
        i = config["id"]
        server_port = config["server_port"]
        qdrant_port = config["qdrant_port"]

        cmd_args = [
            sys.executable,
            "server.py",
            "--id",
            str(i),
            "--intra-port",
            str(server_port),
            "--qdrant",
            f"http://127.0.0.1:{qdrant_port}",
            "--endpoint",
            "HTTP",
            "--peers-file",
            "peers.json",
        ]

        if i == 0:
            cmd_args.append("--coordinator")
            if config["client_port"]:
                cmd_args.extend(["--inter-port", str(config["client_port"])])
            print(
                f"Node 0 (Coordinator) starting on intra-port {server_port}, inter-port {config['client_port']}"
            )
        else:
            print(f"Node {i} starting on intra-port {server_port}")

        p = subprocess.Popen(cmd_args, env=env)
        processes.append(p)
        time.sleep(1)

    if coordinator_client_port:
        time.sleep(2)

        print("\nStarting Web Client...")
        web_client_port = get_free_port()
        while web_client_port == coordinator_client_port:
            web_client_port = get_free_port()

        config_path = os.path.join("web_client", "config.js")
        with open(config_path, "w") as f:
            f.write("// Auto-generated by startup.py\n")
            f.write(
                f"window.SERVER_URL = 'http://127.0.0.1:{coordinator_client_port}';\n"
            )
        print(
            f"Web Client configured to connect to server at http://127.0.0.1:{coordinator_client_port}"
        )

        cmd_web = [sys.executable, "-m", "http.server", str(web_client_port)]
        p_web = subprocess.Popen(
            cmd_web,
            cwd="web_client",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(p_web)

        print(f"Web Client running on http://127.0.0.1:{web_client_port}")

        if wait_for_server_health(coordinator_client_port):
            print("\nCluster is running! Loading data...")
            load_vectors(coordinator_client_port, 51200)
        else:
            print(
                f"\nError: Server at {coordinator_client_port} failed to become ready."
            )

    print("\nSetup Complete. Press Ctrl+C to stop.")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
