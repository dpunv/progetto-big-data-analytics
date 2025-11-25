import json
import re
import time
import subprocess
import sys
import os
import shutil
import signal
import atexit
import requests
import qdrant_module
import logging

# Setup local logging for run.py
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run.log", mode='w')
    ]
)
logger = logging.getLogger("Runner")

server_processes = []

def write_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)
    logger.info("Config file written.")

def write_docker_compose(qdrant_ports):
    logger.info(f"Generating compose.yml for {len(qdrant_ports)} nodes...")

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

    logger.info("compose.yml generated successfully.")

def launch_and_wait_for_qdrant(qdrant_ports, timeout=60):
    logger.info(f"Starting {len(qdrant_ports)} Qdrant databases with Docker Compose...")
    try:
        subprocess.run(["docker", "compose", "-f", "compose.yml", "up", "-d"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting Docker Compose: {e}")
        return False

    logger.info("Waiting for Qdrant nodes to be ready (checking /readyz)...")
    start_time = time.time()
    ready_nodes = set()
    
    while len(ready_nodes) < len(qdrant_ports):
        if time.time() - start_time > timeout:
            logger.error(f"Timeout! Only {len(ready_nodes)}/{len(qdrant_ports)} Qdrant nodes are ready.")
            return False

        for port in qdrant_ports:
            if port in ready_nodes:
                continue
            url = f"http://localhost:{port}/readyz"
            # logger.debug(f"Checking Qdrant node on port {port} at {url}...")
            try:
                response = requests.get(url, timeout=1) 
                
                if response.status_code == 200:
                    ready_nodes.add(port)
                    logger.info(f"  [OK] Qdrant node on port {port} is ready. ({len(ready_nodes)}/{len(qdrant_ports)})")
            except requests.exceptions.RequestException:
                pass
        if len(ready_nodes) < len(qdrant_ports):
            time.sleep(2)

    logger.info("All Qdrant nodes are ready.")
    return True

def launch_servers(fast_api_ports, qdrant_ports, grpc_ports, metrics_ports, coordinator_url='http://localhost:8001', replicas=3, num_before_clustering=1000, batch_size=256, batch_size_retry=64):
    for i in range(1, len(fast_api_ports) + 1):
        node_id = f"node{i}"
        fastapi_port = fast_api_ports[i-1]
        qdrant_http_port = qdrant_ports[i-1]
        grpc_port = grpc_ports[i-1]
        metrics_port = metrics_ports[i-1]
        
        qdrant_url = f'http://localhost:{qdrant_http_port}'
        log_file = f'logs/{node_id}.log'

        proc = subprocess.Popen(
            [sys.executable, "server.py",
                '--node-name', node_id,
                '--node-url', f'http://localhost:{fastapi_port}',
                '--node-grpc-url', f'localhost:{grpc_port}',
                '--qdrant-url', qdrant_url,
                '--coordinator-url', coordinator_url,
                '--replicas', str(replicas),
                '--metrics-port', str(metrics_port),
                '--num-before-clustering', str(num_before_clustering),
                '--log-file', log_file,
                '--batch-size', str(batch_size),
                '--batch-size-retry', str(batch_size_retry)
            ],
            # We do NOT redirect stdout/stderr here so that the server process
            # can write to its own log file cleanly via logging module,
            # but we can let it print to the shell (subprocess default) if we want to see live output,
            # OR we can silence it. 
            # Given the user request, we rely on the internal file logging of the server.
            # We keep stdout visible for basic liveness check in the terminal if needed.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        server_processes.append(proc)
        
        qdrant_module.create_collection(qdrant_url, 'vectors', 384)
        
        logger.info(f"Started server '{node_id}' on port {fastapi_port}, Metrics={metrics_port}, (Qdrant: {qdrant_http_port}, PID: {proc.pid}). Log: {log_file}")

    return server_processes

def wait_for_servers(fast_api_ports, timeout=60):
    start_time = time.time()
    ready_nodes = []
    while len(ready_nodes) < len(fast_api_ports):
        if time.time() - start_time > timeout:
            logger.error(f"Timeout! Only {len(ready_nodes)}/{len(fast_api_ports)} servers are ready.")
            return False
        for i, port in enumerate(fast_api_ports):
            if i in ready_nodes:
                continue
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=2)
                if response.status_code == 200:
                    ready_nodes.append(i)
                    logger.info(f"  [OK] Server on port {port} is ready ({len(ready_nodes)}/{len(fast_api_ports)})")
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
                pass
        
        if len(ready_nodes) < len(fast_api_ports):
            time.sleep(1)
    return True

def cleaning(N):
    logger.info("Shutting down...")
    
    if server_processes:
        logger.info(f"Stopping {len(server_processes)} Python servers...")
        for proc in server_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()

    logger.info("Stopping Docker containers...")
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

    logger.info("Cleanup complete.")

def collect_prometheus_metrics(start_port=10001, num_nodes=3):
    print(f"\n{'='*60}")
    print(f" REPORT METRICHE CLUSTER ({num_nodes} NODI)")
    print(f"{'='*60}")
    histograms = {} 
    counters = {}

    for i in range(num_nodes):
        port = start_port + i
        url = f"http://localhost:{port}/metrics"
        
        try:
            response = requests.get(url, timeout=2)
            if response.status_code != 200: continue
            
            lines = response.text.split('\n')
            
            for line in lines:
                if line.startswith('#') or not line: continue
                if '_created' in line: continue
                if '_bucket' in line: continue
                match_op = re.search(r'operation="([^"]+)"', line)
                if match_op:
                    op_name = match_op.group(1)
                else:
                    op_name = line.split('{')[0] if '{' in line else line.split(' ')[0]
                try:
                    value = float(line.split(' ')[-1])
                except ValueError:
                    continue

                if '_latency_seconds_count' in line or '_latency_seconds_sum' in line:
                    if op_name not in histograms: histograms[op_name] = {'count': 0, 'sum': 0}
                    
                    if '_count' in line:
                        histograms[op_name]['count'] += value
                    elif '_sum' in line:
                        histograms[op_name]['sum'] += value

                elif '_total' in line:
                    if 'python_' in line or 'process_' in line: continue
                    
                    if op_name not in counters: counters[op_name] = 0
                    counters[op_name] += value

        except Exception as e:
            pass

    print(f"{'OPERAZIONE (Tempo)':<30} | {'REQ':<10} | {'AVG (s)':<15}")
    print("-" * 60)
    for op, data in sorted(histograms.items()):
        count = data['count']
        if count > 0:
            avg = data['sum'] / count
            print(f"{op:<30} | {int(count):<10} | {avg:.5f}")
    print("-" * 60)

    print("=" * 60 + "\n")

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    FASTAPI_START_PORT = 8000
    GRPC_START_PORT = 9000
    QDRANT_START_PORT = 6333
    QDRANT_PORT_STEP = 2
    METRICS_START_PORT = 10000

    atexit.register(cleaning, N)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    fast_api_ports = [FASTAPI_START_PORT + i + 1 for i in range(N)]
    qdrant_ports = [QDRANT_START_PORT + (i * QDRANT_PORT_STEP) for i in range(N)]
    grpc_ports = [GRPC_START_PORT + i + 1 for i in range(N)] 
    metrics_ports = [METRICS_START_PORT + i + 1 for i in range(N)] 
    config = {
        'servers': [{'id': f'node{i+1}', 'url': f'http://localhost:{FASTAPI_START_PORT + i + 1}', 'grpc_url': f'localhost:{GRPC_START_PORT + i + 1}', 'is_coordinator': False if i != 0 else True} for i in range(N)],
        'batch_size': 1800,
        'batch_size_retry': 500,
        'num_vectors': 131072,
        'num_before_clustering': 16384,
        'replicas': 2
    }

    write_config(config)

    write_docker_compose(qdrant_ports)

    if not launch_and_wait_for_qdrant(qdrant_ports):
        logger.error("Failed to start Qdrant servers. Exiting.")
        sys.exit(1)

    launch_servers(fast_api_ports, qdrant_ports, grpc_ports, metrics_ports, replicas=config['replicas'], num_before_clustering=config['num_before_clustering'], batch_size=config['batch_size'], batch_size_retry=config['batch_size_retry'])

    if not wait_for_servers(fast_api_ports):
        sys.exit()
    
    logger.info("Launching Client...")
    result = subprocess.run([sys.executable, "client.py", "--log-file", "logs/client.log"])

    collect_prometheus_metrics(start_port=metrics_ports[0], num_nodes=N)
    
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()