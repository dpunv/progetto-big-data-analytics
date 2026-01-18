"""
Comprehensive Benchmark Suite for Distributed Vector Database

This script runs various benchmark scenarios to test the performance of the
in-process distributed vector database system. It measures throughput, latency
percentiles, jitter, and per-server performance.

Usage:
    python benchmark.py [options]

Options:
    --data PATH       Path to embeddings parquet file (default: embeddings.parquet)
    --output PATH     Output report file (default: benchmark_report.txt)
    --servers N       Number of servers to create (default: 8)
    --replication N   Replication factor (default: 4)
    --warmup N        Warmup vectors before benchmark (default: 2500)
"""

import argparse
import logging
import os
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
import server as sv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Each scenario tests the system with different workloads

SCENARIOS = {
    # Quick verification (5s, 2 threads) to confirm connectivity and basic operations
    # (insert/query) without stressing the system, before running real tests.
    "sanity_check": {
        "duration": 5,  # Test duration in seconds
        "concurrency": 2,  # Number of parallel threads (simultaneous clients)
        "mix": {"insert": 0.5, "query": 0.5},  # Probability: 50% insert, 50% query
        "desc": "Quick check to verify system stability",
    },
    "balanced": {
        "duration": 30,  # Longer test for statistically significant results
        "concurrency": 20,  # 20 simulated clients working in parallel
        "mix": {"insert": 0.5, "query": 0.5},
        "desc": "Balanced read/write workload",
    },
    "read_heavy": {
        "duration": 30,
        "concurrency": 20,
        "mix": {"insert": 0.2, "query": 0.8},
        "desc": "Retrieval-heavy (80% queries)",
    },
    "write_heavy": {
        "duration": 30,
        "concurrency": 20,
        "mix": {"insert": 0.8, "query": 0.2},
        "desc": "Ingestion-heavy (80% inserts)",
    },
    "stress_test": {
        "duration": 60,  # Prolonged test to see performance degradation
        "concurrency": 40,  # High concurrency to stress the system
        "mix": {"insert": 0.4, "query": 0.6},
        "desc": "High concurrency stress test",
    },
}


# --- BENCHMARK CLIENT LOGIC ---
#
# IMPORTANT: How load balancing works
# ====================================
# Each operation (insert or query) is sent to ONE randomly chosen server
# among all available servers.
#
# This simulates a real load balancer distributing traffic.
# AGGREGATE metrics (first section of report) sum results from ALL servers.
# PER-SERVER metrics (final section) show performance of each server.


class BenchmarkClient:
    def __init__(self, servers: List[sv.Server]):
        self.servers = servers
        self.request_id = 0
        self.lock = threading.Lock()

    def get_id(self):
        with self.lock:
            self.request_id += 1
            return self.request_id

    def get_target_server(self) -> sv.Server:
        # LOAD BALANCING: randomly choose a server among available ones
        # Each request may go to a different server!
        return random.choice(self.servers)

    def insert(self, vector: List[float], payload_text: str) -> Tuple[float, bool, int]:
        """Sends an insert request and returns (latency, success, server_id)."""
        server = self.get_target_server()
        server_id = server.get_id()

        # Format: List[Tuple[Vector, Payload]]
        content = [(vector, payload_text)]

        start = time.time()
        success = False
        try:
            server.receive_from_client(content)
            success = True
        except Exception as e:
            logger.error(f"Insert failed on server {server_id}: {e}")

        return time.time() - start, success, server_id

    def query(self, vector: List[float]) -> Tuple[float, bool, int]:
        """Sends a query request and returns (latency, success, server_id)."""
        server = self.get_target_server()
        server_id = server.get_id()

        start = time.time()
        success = False
        try:
            results = server.query_from_client([vector])
            if results is not None:
                success = True
            else:
                logger.error(f"Query returned None on server {server_id}")
        except Exception as e:
            logger.error(f"Query failed on server {server_id}: {e}")

        return time.time() - start, success, server_id


def load_data(filepath: str) -> List[Dict[str, Any]]:
    import pandas as pd
    logger.info(f"Loading vectors from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        # Convert DataFrame to list of dicts with 'embedding' and 'text' keys
        data = [
            {"embedding": row["embedding"].tolist() if hasattr(row["embedding"], "tolist") else list(row["embedding"]), 
             "text": row["sentence"]}
            for _, row in df.iterrows()
        ]
        logger.info(f"Loaded {len(data)} vectors.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []


# --- DOCKER CLUSTER ORCHESTRATION ---

def generate_compose_and_dirs(num_servers: int, start_port: int = 6333):
    """Generate compose.yml and create storage directories for Qdrant nodes."""
    logger.info(f"Generating compose.yml for {num_servers} Qdrant nodes...")

    services = []
    for i in range(num_servers):
        http_port = start_port + (i * 2)
        grpc_port = http_port + 1
        storage_dir = f"./qdrant_storage_{i}"

        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)

        service_def = f"""  qdrant-{i}:
    image: qdrant/qdrant:latest
    container_name: qdrant-{i}
    ports:
      - "{http_port}:6333"
      - "{grpc_port}:6334"
    volumes:
      - {storage_dir}:/qdrant/storage
    restart: unless-stopped"""
        services.append(service_def)

    compose_content = "services:\n" + "\n".join(services)

    with open("compose.yml", "w") as f:
        f.write(compose_content)

    logger.info("compose.yml generated.")


def wait_for_qdrant_cluster(num_servers: int, start_port: int = 6333, timeout: int = 60) -> bool:
    """Wait for all Qdrant nodes to be ready."""
    logger.info("Waiting for Qdrant nodes to be ready...")
    ready_count = 0
    start_time = time.time()

    while ready_count < num_servers:
        if time.time() - start_time > timeout:
            logger.error("Timeout waiting for Qdrant nodes.")
            return False

        ready_count = 0
        for i in range(num_servers):
            port = start_port + (i * 2)
            try:
                resp = requests.get(f"http://localhost:{port}/readyz", timeout=1)
                if resp.status_code == 200:
                    ready_count += 1
            except Exception:
                pass

        if ready_count < num_servers:
            time.sleep(1)

    logger.info(f"All {num_servers} Qdrant nodes are ready.")
    return True


def cleanup_cluster(num_servers: int):
    """Stop Docker containers and clean up storage directories."""
    logger.info("Cleaning up Docker cluster...")
    try:
        subprocess.run(["docker", "compose", "down"], check=True, capture_output=True)
        logger.info("Docker containers stopped.")
    except Exception as e:
        logger.warning(f"Error stopping docker: {e}")

    # Clean up storage directories
    for i in range(num_servers):
        storage_dir = f"./qdrant_storage_{i}"
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
    logger.info("Storage directories removed.")


def start_docker_cluster(num_servers: int, start_port: int = 6333) -> bool:
    """Generate compose file and start Docker containers."""
    generate_compose_and_dirs(num_servers, start_port)

    logger.info("Starting Docker Compose...")
    try:
        subprocess.run(["docker", "compose", "up", "-d"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Docker containers: {e}")
        return False

    return wait_for_qdrant_cluster(num_servers, start_port)


def get_stats_summary(name: str, latencies: List[float], errors: int = 0) -> str:
    """
    Generate a statistical summary of metrics for an operation type.

    METRICS CALCULATED:
    - Error Rate: percentage of failed operations
    - Latencies: avg, P50 (median), P95, P99
    - Jitter (Std Dev): measures variability/predictability of latencies
    - Tail Latency Ratio (P99/P50): measures system consistency
    """
    n = len(latencies)
    total_attempts = n + errors

    if total_attempts == 0:
        return f"No {name} operations recorded."

    error_rate = (errors / total_attempts) * 100

    if n == 0:
        return (
            f"\n--- {name.upper()} Metrics (0 success, {errors} errors) ---\n"
            f"Error Rate:      {error_rate:.2f}%"
        )

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if n >= 20 else max(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98] if n >= 100 else max(latencies)

    # METRIC 1: Jitter (Standard Deviation)
    # Measures how much latencies vary from the mean.
    # - Low jitter = predictable and stable system
    # - High jitter = highly variable latencies, less reliable system
    #
    #    Value     Meaning
    #    < 40ms    EXCELLENT - Very stable system
    #    40-90ms   GOOD - Normal variability
    #    > 90ms    HIGH - Unpredictable system
    jitter = statistics.stdev(latencies) if n >= 2 else 0.0

    # METRIC 2: Tail Latency Ratio (P99/P50)
    # Measures how much slower the slowest requests (outliers) are vs median.
    # - Ratio ~1-2x: EXCELLENT, very consistent system, few outliers
    # - Ratio 3-5x : NORMAL under moderate load
    # - Ratio >5x  : UNSTABLE, many "unlucky" requests are very slow
    # Important for guaranteeing SLA (Service Level Agreement) in production.
    tail_ratio = p99 / p50 if p50 > 0 else 0.0

    return (
        f"\n--- {name.upper()} Metrics ({n} success, {errors} errors) ---\n"
        f"Error Rate:      {error_rate:.2f}%\n"
        f"Average Latency: {avg * 1000:.2f} ms\n"
        f"P50 Latency:     {p50 * 1000:.2f} ms\n"
        f"P95 Latency:     {p95 * 1000:.2f} ms\n"
        f"P99 Latency:     {p99 * 1000:.2f} ms\n"
        f"Jitter (StdDev): {jitter * 1000:.2f} ms\n"
        f"Tail Ratio:      {tail_ratio:.2f}x (P99/P50)"
    )


def run_benchmark(
    client: BenchmarkClient,
    data: List[Dict[str, Any]],
    duration: int,  # Total test duration (e.g., 30 seconds)
    concurrency: int,  # Number of parallel threads (e.g., 10 workers)
    mix: Dict[str, float],  # Operation probabilities (e.g., 50% insert, 50% query)
):
    """
    HOW THE BENCHMARK WORKS:

    1. Creates N worker threads (concurrency), each working in parallel
    2. Each worker runs a continuous loop for 'duration' seconds
    3. In each iteration, the worker randomly decides to do insert or query
       based on the 'mix' (e.g., 50% probability insert, 50% query)
    4. Each operation measures latency and tracks success/error

    PRACTICAL EXAMPLE (balanced scenario):
    - concurrency=10 means 10 threads working simultaneously
    - duration=30 means each thread works for 30 seconds
    - mix=50/50 means each thread, at each iteration, has 50% probability
      of doing insert and 50% of doing query
    - If each operation takes ~150ms, each thread does ~200 operations in 30s
    - Total: 10 threads × 200 ops = ~2000 total operations
    - Of which ~1000 insert and ~1000 query (for 50/50 mix)
    """
    stop_event = threading.Event()  # Signal to stop all workers

    # Structure to collect all latencies and errors
    results = {
        "insert": [],  # List of all successful insert latencies
        "query": [],  # List of all successful query latencies
        "errors": {
            "insert": 0,  # Failed insert counter
            "query": 0,  # Failed query counter
        },
        # NEW: Per-server tracking to identify bottlenecks
        "per_server": {},  # Dict: {server_id: {'insert': [latencies], 'query': [latencies], 'errors': count}}
    }
    results_lock = threading.Lock()  # For thread-safe access to results

    insert_ratio = mix.get("insert", 0.0)
    query_ratio = mix.get("query", 0.0)
    total_weight = insert_ratio + query_ratio

    if total_weight == 0:
        logger.error("Invalid mix configuration: weights sum to 0")
        return results

    # Calculate threshold for deciding insert vs query
    # E.g.: if insert=0.5, query=0.5 → threshold=0.5
    # random.random() < 0.5 has 50% probability of being True
    normalized_insert_threshold = insert_ratio / total_weight

    def worker():
        """
        Function executed by each worker thread.
        Each worker runs a loop until stop_event is set.

        BEHAVIOR OF A SINGLE WORKER:
        - Infinite loop until stop_event (after 'duration' seconds)
        - Each iteration: randomly chooses insert or query
        - Executes the operation and records latency/error
        - Continues as fast as possible (no sleep)

        With concurrency=10, there are 10 workers doing this in parallel!
        """
        local_results = {"insert": [], "query": []}
        local_errors = {"insert": 0, "query": 0}
        # NEW: Local per-server tracking
        local_per_server = {}  # {server_id: {'insert': [], 'query': [], 'errors': 0}}

        while not stop_event.is_set():  # Continue until time expires
            # Probabilistic decision: insert or query?
            # random.random() generates a number between 0 and 1
            op_type = (
                "insert" if random.random() < normalized_insert_threshold else "query"
            )

            # Choose a random vector from the dataset
            item = random.choice(data)
            vector = item["embedding"]
            text = item.get("text", "")

            # Execute the chosen operation and measure latency
            if op_type == "insert":
                latency, success, server_id = client.insert(vector, text)
                # Initialize per-server structure if needed
                if server_id not in local_per_server:
                    local_per_server[server_id] = {
                        "insert": [],
                        "query": [],
                        "errors": 0,
                    }
                if success:
                    local_results["insert"].append(latency)
                    local_per_server[server_id]["insert"].append(latency)
                else:
                    local_errors["insert"] += 1
                    local_per_server[server_id]["errors"] += 1
            else:
                latency, success, server_id = client.query(vector)
                # Initialize per-server structure if needed
                if server_id not in local_per_server:
                    local_per_server[server_id] = {
                        "insert": [],
                        "query": [],
                        "errors": 0,
                    }
                if success:
                    local_results["query"].append(latency)
                    local_per_server[server_id]["query"].append(latency)
                else:
                    local_errors["query"] += 1
                    local_per_server[server_id]["errors"] += 1

        # At the end, aggregate local results into global results
        with results_lock:
            results["insert"].extend(local_results["insert"])
            results["query"].extend(local_results["query"])
            results["errors"]["insert"] += local_errors["insert"]
            results["errors"]["query"] += local_errors["query"]
            # Aggregate per-server data
            for server_id, data_server in local_per_server.items():
                if server_id not in results["per_server"]:
                    results["per_server"][server_id] = {
                        "insert": [],
                        "query": [],
                        "errors": 0,
                    }
                results["per_server"][server_id]["insert"].extend(data_server["insert"])
                results["per_server"][server_id]["query"].extend(data_server["query"])
                results["per_server"][server_id]["errors"] += data_server["errors"]

    logger.info(
        f"Starting benchmark with {concurrency} threads for {duration} seconds..."
    )
    logger.info(
        f"Workload: {insert_ratio * 100:.1f}% Insert, {query_ratio * 100:.1f}% Query"
    )

    # WORKER ORCHESTRATION:
    # 1. Create a thread pool
    # 2. Launch 'concurrency' workers in parallel (e.g., 10 threads)
    # 3. Wait 'duration' seconds (e.g., 30s)
    # 4. Signal stop to all workers
    # 5. Wait for all workers to finish and collect results
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Launch all workers simultaneously
        futures = [executor.submit(worker) for _ in range(concurrency)]

        # Let workers run for 'duration' seconds
        time.sleep(duration)

        # Signal all workers to stop
        stop_event.set()

        # Wait for all workers to finish
        for f in futures:
            f.result()

    return results


# --- SUITE ORCHESTRATION ---


def get_vector_count(servers: List[sv.Server]) -> int:
    """Get total vector count across all servers."""
    try:
        return sum(s.count() for s in servers)
    except Exception as e:
        logger.error(f"Failed to get count: {e}")
    return -1


def get_per_server_summary(per_server_data: Dict[int, Dict]) -> str:
    """
    Generate a report of performance for each individual server.
    Useful for identifying bottlenecks and slow servers.

    OUTPUT:
    - Average latency per server
    - Number of operations per server
    - Comparison between servers (% difference from average)
    """
    if not per_server_data:
        return "\nNo per-server data available.\n"

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("FINAL SUMMARY: PER-SERVER PERFORMANCE ANALYSIS")
    lines.append("(Aggregated across ALL scenarios above)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("This section shows the TOTAL performance of each server")
    lines.append("across all benchmark scenarios combined.")

    # Calculate metrics for each server
    server_stats = {}
    all_latencies = []  # For calculating global average

    for server_id, data in per_server_data.items():
        all_lats = data["insert"] + data["query"]
        if all_lats:
            avg_lat = statistics.mean(all_lats)
            total_ops = len(all_lats)
            errors = data["errors"]
            server_stats[server_id] = {
                "avg_latency": avg_lat,
                "total_ops": total_ops,
                "insert_count": len(data["insert"]),
                "query_count": len(data["query"]),
                "errors": errors,
                "insert_avg": statistics.mean(data["insert"]) if data["insert"] else 0,
                "query_avg": statistics.mean(data["query"]) if data["query"] else 0,
            }
            all_latencies.extend(all_lats)

    if not server_stats:
        return "\nNo successful operations recorded per server.\n"

    # Global average for comparison
    global_avg = statistics.mean(all_latencies) if all_latencies else 0

    lines.append(f"\nGlobal Average Latency: {global_avg * 1000:.2f} ms")
    lines.append("-" * 60)

    # Report for each server
    for server_id, stats in sorted(server_stats.items()):
        # Calculate % difference from global average
        diff_pct = (
            ((stats["avg_latency"] - global_avg) / global_avg * 100)
            if global_avg > 0
            else 0
        )
        # Performance indicator
        status = (
            "[OK]" if diff_pct <= 10 else ("[WARN]" if diff_pct <= 30 else "[SLOW]")
        )

        lines.append(f"\nServer {server_id}:")
        lines.append(
            f"  Total Operations: {stats['total_ops']} ({stats['insert_count']} insert, {stats['query_count']} query)"
        )
        lines.append(f"  Errors: {stats['errors']}")
        lines.append(f"  Avg Latency (all): {stats['avg_latency'] * 1000:.2f} ms")
        lines.append(f"  Avg Insert Latency: {stats['insert_avg'] * 1000:.2f} ms")
        lines.append(f"  Avg Query Latency: {stats['query_avg'] * 1000:.2f} ms")
        lines.append(f"  Diff from Global: {diff_pct:+.1f}% {status}")

    # Bottleneck analysis
    lines.append("\n" + "-" * 60)
    lines.append("BOTTLENECK ANALYSIS:")

    # Find fastest and slowest servers
    sorted_by_lat = sorted(server_stats.items(), key=lambda x: x[1]["avg_latency"])
    fastest = sorted_by_lat[0]
    slowest = sorted_by_lat[-1]

    speed_diff = (
        (
            (slowest[1]["avg_latency"] - fastest[1]["avg_latency"])
            / fastest[1]["avg_latency"]
            * 100
        )
        if fastest[1]["avg_latency"] > 0
        else 0
    )

    lines.append(
        f"  Fastest: Server {fastest[0]} ({fastest[1]['avg_latency'] * 1000:.2f} ms)"
    )
    lines.append(
        f"  Slowest: Server {slowest[0]} ({slowest[1]['avg_latency'] * 1000:.2f} ms)"
    )
    lines.append(f"  Speed Difference: {speed_diff:.1f}%")

    if speed_diff > 50:
        lines.append(f"  WARNING: Server {slowest[0]} is significantly slower!")
        lines.append("     Consider investigating load or resource issues.")
    elif speed_diff > 20:
        lines.append("  NOTICE: Moderate performance difference between servers.")
    else:
        lines.append("  OK: All servers performing similarly. No bottleneck detected.")

    lines.append("=" * 60 + "\n")

    return "\n".join(lines)


def wait_for_clustered(servers: List[sv.Server], timeout: int = 120) -> bool:
    """
    POLLING: Wait for all servers to reach 'clustered' status.

    This is necessary because after warmup the system might still be
    in 'clustering' phase (while computing centroids or distributing
    vectors) and queries would fail or be incomplete.
    """
    logger.info(
        f"Waiting for all {len(servers)} servers to reach CLUSTERED status (timeout: {timeout}s)..."
    )
    start_time = time.time()

    while time.time() - start_time < timeout:
        all_ready = True
        for server in servers:
            try:
                status = server.get_status()
                if status != "clustered":
                    all_ready = False
                    break
            except Exception:
                all_ready = False
                break

        if all_ready:
            logger.info("All servers are CLUSTERED and ready for benchmark.")
            return True

        time.sleep(0.5)
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            logger.info(f"Still waiting... ({elapsed}s elapsed)")

    logger.warning(
        "Timeout reached waiting for servers to cluster. Proceeding anyway..."
    )
    return False


def wait_for_coordinator_discovery(servers: List[sv.Server], timeout: int = 30) -> bool:
    """
    Wait for all non-coordinator servers to be able to find the coordinator.
    
    This is critical for proper operation - if servers can't find the coordinator
    during bootstrap phase, they won't be able to forward vectors and data will be lost.
    """
    logger.info("Waiting for coordinator discovery...")
    start_time = time.time()
    
    # Find which server is the coordinator
    coord_id = None
    for s in servers:
        if s.i_am_coord():
            coord_id = s.get_id()
            break
    
    if coord_id is None:
        logger.error("No coordinator found among servers!")
        return False
    
    logger.info(f"Coordinator is server {coord_id}. Waiting for all peers to discover it...")
    
    while time.time() - start_time < timeout:
        all_found = True
        for server in servers:
            if server.i_am_coord():
                continue  # Coordinator doesn't need to find itself
            
            coord = server.coordinator()
            if coord is None:
                all_found = False
                break
        
        if all_found:
            logger.info("All servers can reach the coordinator. Ready for warmup.")
            return True
        
        time.sleep(0.5)
    
    logger.warning(f"Timeout waiting for coordinator discovery. Vectors may be dropped!")
    return False


def wait_for_queues(servers: List[sv.Server], timeout: int = 300) -> bool:
    """Wait for all server queues to drain."""
    logger.info("Waiting for server queues to drain...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        total_q = sum(s.get_queue_size() for s in servers)
        any_clustering = any(s.is_clustering() for s in servers)

        if total_q == 0 and not any_clustering:
            logger.info("All queues drained.")
            return True

        sys.stdout.write(
            f"\r    Queues pending: {total_q}, Clustering: {any_clustering}   "
        )
        sys.stdout.flush()
        time.sleep(0.5)

    print()
    logger.warning("Timeout waiting for queues to drain!")
    return False


def populate_and_wait_clustering(
    client: BenchmarkClient,
    data: List[Dict],
    servers: List[sv.Server],
    target_vectors: int = 10000,
):
    """
    WARMUP PHASE: Prepare the system before benchmarks

    PURPOSE:
    - Insert a significant number of vectors (default 10000)
    - Trigger automatic clustering of the system (threshold is 8192)
    - Bring the system to a "warm" and stable state
    - Avoid measuring initialization time in the first benchmark

    HOW IT WORKS:
    - Uses 20 concurrent threads to speed up insertion
    - Each thread continues inserting until we reach target_vectors
    - Timeout errors are normal (system under load) and are ignored
    - Only counts successful insertions
    """
    # Ensure we insert at least enough to trigger clustering (threshold is 8192)
    min_for_clustering = 8500
    if target_vectors < min_for_clustering:
        logger.info(
            f"Adjusting warmup count from {target_vectors} to {min_for_clustering} (clustering threshold)"
        )
        target_vectors = min_for_clustering

    logger.info(f"WARMUP: Populating {target_vectors} vectors to trigger clustering...")
    logger.info("This ensures the system is in a clustered state before benchmarking.")

    concurrency = 20  # 20 parallel threads for fast warmup
    successful_inserts = 0
    lock = threading.Lock()

    def warmup_worker():
        nonlocal successful_inserts
        while True:
            # Check if we've reached the target
            with lock:
                if successful_inserts >= target_vectors:
                    return

            # Choose a random vector and try to insert it
            item = random.choice(data)
            vector = item["embedding"]
            text = item.get("text", "")

            try:
                _, success, _ = client.insert(vector, text)
                if success:
                    with lock:
                        successful_inserts += 1
                        # Progress indicator every 100 insertions
                        if successful_inserts % 100 == 0:
                            sys.stdout.write(
                                f"\rInserted {successful_inserts}/{target_vectors}"
                            )
                            sys.stdout.flush()
            except Exception:
                # Ignore errors during warmup (normal timeouts under load)
                pass

    logger.info(f"Starting concurrent warmup with {concurrency} threads...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(warmup_worker) for _ in range(concurrency)]
        for f in futures:
            f.result()

    print()  # Newline
    logger.info(f"Warmup complete. {successful_inserts} vectors inserted.")

    # Wait for queues to drain
    wait_for_queues(servers)

    # Wait for all servers to reach clustered state
    wait_for_clustered(servers)


def run_suite(
    data_path: str,
    output_file: str,
    num_servers: int,
    replication_factor: int,
    warmup_count: int,
    cluster_mode: bool = False,
    start_port: int = 6333,
):
    """Run the complete benchmark suite."""

    logger.info(f"Loading data from {data_path}...")
    data = load_data(data_path)
    if not data:
        logger.error("Failed to load data. Aborting.")
        sys.exit(1)

    # Configuration based on client.py
    num_vectors_before_clustering = 8192

    # Start Docker cluster if cluster mode is enabled
    if cluster_mode:
        logger.info("Starting Qdrant cluster with Docker...")
        if not start_docker_cluster(num_servers, start_port):
            logger.error("Failed to start Qdrant cluster. Aborting.")
            sys.exit(1)
        logger.info("Qdrant cluster started successfully.")

    # Create servers
    logger.info(
        f"Creating {num_servers} servers with replication factor {replication_factor}..."
    )
    if cluster_mode:
        logger.info(f"Using Docker cluster mode (ports starting at {start_port})")
    else:
        logger.info("Using in-memory Qdrant storage")

    servers = []
    for i in range(num_servers):
        # Determine Qdrant URL based on mode
        if cluster_mode:
            # Each server connects to its own Qdrant container
            port = start_port + (i * 2)
            qdrant_url = f"http://localhost:{port}"
        else:
            qdrant_url = ":memory:"

        # Server with highest ID is initial coordinator
        servers.append(
            sv.Server(
                i,
                i == num_servers - 1,
                num_vectors_before_clustering,
                replication_factor,
                port=8000 + i,
                qdrant_url=qdrant_url,
            )
        )

    # Register peers
    for server in servers:
        for peer in servers:
            if server.get_id() == peer.get_id():
                continue
            server.add_peer(peer)

    logger.info(f"{num_servers} servers started and connected")
    
    # Wait for coordinator discovery before any operations
    # This is critical - without this, non-coordinator servers won't be able
    # to forward vectors to the coordinator and data will be lost
    if not wait_for_coordinator_discovery(servers):
        logger.error("Failed to establish coordinator discovery. Aborting.")
        for s in servers:
            s.stop()
        if cluster_mode:
            cleanup_cluster(num_servers)
        sys.exit(1)

    client = BenchmarkClient(servers)

    # Initial Warmup
    initial_count = get_vector_count(servers)
    logger.info(f"Initial System Vector Count: {initial_count}")

    populate_and_wait_clustering(client, data, servers, warmup_count)

    post_warmup_count = get_vector_count(servers)
    logger.info(
        f"Post-Warmup Vector Count: {post_warmup_count} (+{post_warmup_count - initial_count})"
    )

    # Write report header
    with open(output_file, "w") as f:
        f.write("========================================================\n")
        f.write("DISTRIBUTED VECTOR DB BENCHMARK SUITE REPORT\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Servers: {num_servers}\n")
        f.write(f"Replication Factor: {replication_factor}\n")
        f.write(f"Initial Count: {initial_count}\n")
        f.write(f"Post-Warmup Count: {post_warmup_count}\n")
        f.write("========================================================\n\n")

    # Structure to accumulate per-server data from all scenarios
    all_per_server_data = {}

    for name, config in SCENARIOS.items():
        logger.info(f"--- Running Scenario: {name.upper()} ---")
        logger.info(f"Description: {config['desc']}")

        pre_test_count = get_vector_count(servers)
        logger.info(f"Pre-Test Count: {pre_test_count}")

        start_time = time.time()
        results = run_benchmark(
            client, data, config["duration"], config["concurrency"], config["mix"]
        )
        total_time = time.time() - start_time

        # Wait for queues to drain before counting
        wait_for_queues(servers, timeout=60)

        post_test_count = get_vector_count(servers)
        count_delta = post_test_count - pre_test_count
        logger.info(f"Post-Test Count: {post_test_count} (Delta: +{count_delta})")

        # Calculate summary metrics
        total_ops = (
            len(results["insert"])
            + len(results["query"])
            + results["errors"]["insert"]
            + results["errors"]["query"]
        )
        throughput = total_ops / total_time if total_time > 0 else 0

        # ============================================================
        # GENERATE REPORT - AGGREGATE METRICS (all servers together)
        # ============================================================
        # The metrics below are the SUM/AVERAGE of all servers.
        # Requests were distributed randomly among servers.

        server_ids = [s.get_id() for s in servers]
        report_section = [
            "--------------------------------------------------------",
            f"SCENARIO: {name.upper()}",
            f"Description: {config['desc']}",
            f"Duration: {config['duration']}s, Concurrency: {config['concurrency']}, Mix: {config['mix']}",
            "",
            "[AGGREGATE METRICS - All servers combined]",
            f"(Requests distributed randomly across servers: {server_ids})",
            "",
            f"Total Requests: {total_ops}",
            f"Throughput: {throughput:.2f} req/s",
            f"Vectors Stored Delta: +{count_delta} (Validation)",
            get_stats_summary("Insert", results["insert"], results["errors"]["insert"]),
            get_stats_summary("Query", results["query"], results["errors"]["query"]),
        ]

        # ============================================================
        # GENERATE REPORT - PER-SERVER METRICS (detail per scenario)
        # ============================================================
        # Here we show how each individual server performed in this scenario

        report_section.append("")
        report_section.append("[PER-SERVER BREAKDOWN - This scenario only]")

        for server_id in sorted(results["per_server"].keys()):
            srv_data = results["per_server"][server_id]
            srv_insert_count = len(srv_data["insert"])
            srv_query_count = len(srv_data["query"])
            srv_total = srv_insert_count + srv_query_count
            srv_errors = srv_data["errors"]
            srv_insert_avg = (
                statistics.mean(srv_data["insert"]) * 1000 if srv_data["insert"] else 0
            )
            srv_query_avg = (
                statistics.mean(srv_data["query"]) * 1000 if srv_data["query"] else 0
            )

            report_section.append(
                f"  Server {server_id}: {srv_total} ops ({srv_insert_count} ins/{srv_query_count} qry), "
                f"Insert: {srv_insert_avg:.0f}ms, Query: {srv_query_avg:.0f}ms, Errors: {srv_errors}"
            )

        report_section.append(
            "--------------------------------------------------------\n"
        )

        # STREAMING: write immediately after each scenario (skip sanity_check)
        if name != "sanity_check":
            with open(output_file, "a") as f:
                f.write("\n".join(report_section) + "\n")
                f.flush()  # Force write to disk

        # Accumulate per-server data for FINAL SUMMARY (all scenarios together, except sanity_check)
        if name != "sanity_check":
            for server_id, server_data in results["per_server"].items():
                if server_id not in all_per_server_data:
                    all_per_server_data[server_id] = {
                        "insert": [],
                        "query": [],
                        "errors": 0,
                    }
                all_per_server_data[server_id]["insert"].extend(server_data["insert"])
                all_per_server_data[server_id]["query"].extend(server_data["query"])
                all_per_server_data[server_id]["errors"] += server_data["errors"]

        logger.info(f"Finished {name}. Throughput: {throughput:.2f} req/s")
        # Cool down between tests
        time.sleep(2)

    # Write per-server report at the end
    with open(output_file, "a") as f:
        f.write(get_per_server_summary(all_per_server_data))

        # Add system-specific metrics
        f.write("\n" + "=" * 60 + "\n")
        f.write("SYSTEM-SPECIFIC METRICS\n")
        f.write("=" * 60 + "\n")
        f.write("\nFinal Server Status:\n")
        for s in servers:
            queue_size = s.get_queue_size()
            hints = s.hinted_handoff.count()
            count = s.count()
            status = s.get_status()
            coord = "COORD" if s.is_coordinator else ""
            f.write(
                f"  Server {s.get_id()}: {count} vectors, queue: {queue_size}, hints: {hints}, status: {status} {coord}\n"
            )

        total_vectors = get_vector_count(servers)
        f.write(f"\nTotal vectors across cluster: {total_vectors}\n")

    # Stop servers
    logger.info("Stopping servers...")
    for s in servers:
        s.stop()

    # Cleanup Docker cluster if cluster mode was enabled
    if cluster_mode:
        cleanup_cluster(num_servers)

    logger.info(f"Benchmark suite completed. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Full Benchmark Suite")
    parser.add_argument(
        "--data", type=str, default="embeddings.parquet", help="Path to data file"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_report.txt", help="Output report file"
    )
    parser.add_argument("--servers", type=int, default=8, help="Number of servers")
    parser.add_argument("--replication", type=int, default=4, help="Replication factor")
    parser.add_argument("--warmup", type=int, default=10000, help="Warmup vector count")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Enable Docker cluster mode (deploy real Qdrant instances)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=6333,
        help="Starting port for Qdrant containers (default: 6333)",
    )

    args = parser.parse_args()

    run_suite(
        args.data,
        args.output,
        args.servers,
        args.replication,
        args.warmup,
        cluster_mode=args.cluster,
        start_port=args.start_port,
    )


if __name__ == "__main__":
    main()
