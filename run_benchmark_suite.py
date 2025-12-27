import argparse
import time
import sys
import json
import logging
import requests
import threading
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

SCENARIOS = {
    "sanity_check": {
        "duration": 5,
        "concurrency": 2,
        "mix": {"insert": 0.5, "query": 0.5},
        "desc": "Quick check to verify system stability"
    },
    "balanced": {
        "duration": 30,
        "concurrency": 10,
        "mix": {"insert": 0.5, "query": 0.5},
        "desc": "Balanced read/write workload"
    },
    "read_heavy": {
        "duration": 30,
        "concurrency": 10,
        "mix": {"insert": 0.1, "query": 0.9},
        "desc": "Retrieval-heavy (90% queries)"
    },
    "write_heavy": {
        "duration": 30,
        "concurrency": 10,
        "mix": {"insert": 0.9, "query": 0.1},
        "desc": "Ingestion-heavy (90% inserts)"
    },
    "stress_test": {
        "duration": 60,
        "concurrency": 50,
        "mix": {"insert": 0.5, "query": 0.5},
        "desc": "High concurrency stress test"
    }
}

# --- BENCHMARK CLIENT LOGIC (formerly benchmark.py) ---

class BenchmarkClient:
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.request_id = 0
        self.lock = threading.Lock()

    def get_id(self):
        with self.lock:
            self.request_id += 1
            return self.request_id

    def get_target_url(self):
        return random.choice(self.urls)

    def insert(self, vector: List[float], payload_text: str) -> Tuple[float, bool]:
        """Sends an insert request and returns (latency, success)."""
        url = self.get_target_url()
        req_id = self.get_id()
        
        # Format: List[Tuple[Vector, Payload]]
        content = [(vector, payload_text)]
        
        payload = {
            'id': req_id,
            'content': content
        }
        
        start = time.time()
        success = False
        try:
            resp = requests.post(f"{url}/add", json=payload, timeout=10)
            if resp.status_code == 200:
                success = True
            else:
                logger.error(f"Insert failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            
        return time.time() - start, success

    def query(self, vector: List[float]) -> Tuple[float, bool]:
        """Sends a query request and returns (latency, success)."""
        url = self.get_target_url()
        req_id = self.get_id()
        
        payload = {
            'id': req_id,
            'query': [vector],
            'topk': 5
        }
        
        start = time.time()
        success = False
        try:
            resp = requests.post(f"{url}/query", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'success':
                    success = True
                else:
                    logger.error(f"Query returned failure status: {data}")
            else:
                logger.error(f"Query failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
        return time.time() - start, success

def load_data(filepath: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading vectors from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} vectors.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []

def get_stats_summary(name: str, latencies: List[float], errors: int = 0) -> str:
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
    
    return (
        f"\n--- {name.upper()} Metrics ({n} success, {errors} errors) ---\n"
        f"Error Rate:      {error_rate:.2f}%\n"
        f"Average Latency: {avg*1000:.2f} ms\n"
        f"P50 Latency:     {p50*1000:.2f} ms\n"
        f"P95 Latency:     {p95*1000:.2f} ms\n"
        f"P99 Latency:     {p99*1000:.2f} ms"
    )

def run_benchmark(
    client: BenchmarkClient,
    data: List[Dict[str, Any]],
    duration: int,
    concurrency: int,
    mix: Dict[str, float]
):
    stop_event = threading.Event()
    results = {
        'insert': [],
        'query': [],
        'errors': {
            'insert': 0,
            'query': 0
        }
    }
    results_lock = threading.Lock()
    
    insert_ratio = mix.get('insert', 0.0)
    query_ratio = mix.get('query', 0.0)
    total_weight = insert_ratio + query_ratio
    
    if total_weight == 0:
        logger.error("Invalid mix configuration: weights sum to 0")
        return results

    normalized_insert_threshold = insert_ratio / total_weight

    def worker():
        local_results = {'insert': [], 'query': []}
        local_errors = {'insert': 0, 'query': 0}
        
        while not stop_event.is_set():
            # Pick operation based on mix
            op_type = 'insert' if random.random() < normalized_insert_threshold else 'query'
            
            # Pick random vector from data
            item = random.choice(data)
            vector = item['embedding']
            text = item.get('text', '')

            if op_type == 'insert':
                latency, success = client.insert(vector, text)
                if success:
                    local_results['insert'].append(latency)
                else:
                    local_errors['insert'] += 1
            else:
                latency, success = client.query(vector)
                if success:
                    local_results['query'].append(latency)
                else:
                    local_errors['query'] += 1
        
        with results_lock:
            results['insert'].extend(local_results['insert'])
            results['query'].extend(local_results['query'])
            results['errors']['insert'] += local_errors['insert']
            results['errors']['query'] += local_errors['query']

    logger.info(f"Starting benchmark with {concurrency} threads for {duration} seconds...")
    logger.info(f"Workload: {insert_ratio*100:.1f}% Insert, {query_ratio*100:.1f}% Query")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        time.sleep(duration)
        stop_event.set()
        for f in futures:
            f.result()

    return results

# --- SUITE ORCHESTRATION ---

def get_vector_count(client_url):
    try:
        resp = requests.get(f"{client_url}/count", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # If data is a dict (per-node count), sum values. If int, return it.
            if isinstance(data, dict):
                 return sum([v for v in data.values() if isinstance(v, int)])
            return int(data)
    except Exception as e:
        logger.error(f"Failed to get count from {client_url}: {e}")
    return -1

def populate_and_wait_clustering(client, data, target_vectors=2500):
    logger.info(f"WARMUP: Populating {target_vectors} vectors to trigger clustering...")
    logger.info("This ensures the system is in a clustered state before benchmarking.")
    
    concurrency = 20
    successful_inserts = 0
    lock = threading.Lock()
    
    def warmup_worker():
        nonlocal successful_inserts
        while True:
            with lock:
                if successful_inserts >= target_vectors:
                    return
                current_count = successful_inserts
            
            item = random.choice(data)
            vector = item['embedding']
            text = item.get('text', '')
            
            try:
                _, success = client.insert(vector, text)
                if success:
                    with lock:
                        successful_inserts += 1
                        if successful_inserts % 100 == 0:
                            sys.stdout.write(f"\rInserted {successful_inserts}/{target_vectors}")
                            sys.stdout.flush()
            except Exception:
                pass

    logger.info(f"Starting concurrent warmup with {concurrency} threads...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(warmup_worker) for _ in range(concurrency)]
        for f in futures:
            f.result()
            
    print() # Newline
    logger.info(f"Warmup complete. {successful_inserts} vectors inserted.")
    
    logger.info("Waiting 5s to allow for clustering/stabilization...")
    time.sleep(5) 

def run_suite(urls_str, data_path, output_file):
    logger.info(f"Loading data from {data_path}...")
    data = load_data(data_path)
    if not data:
        logger.error("Failed to load data. Aborting.")
        sys.exit(1)

    urls = [u.strip() for u in urls_str.split(',')]
    client = BenchmarkClient(urls)
    
    # Initial Warmup
    initial_count = get_vector_count(urls[0])
    logger.info(f"Initial System Vector Count: {initial_count}")
    
    populate_and_wait_clustering(client, data)
    
    post_warmup_count = get_vector_count(urls[0])
    logger.info(f"Post-Warmup Vector Count: {post_warmup_count} (+{post_warmup_count - initial_count})")
    
    with open(output_file, "w") as f:
        f.write("========================================================\n")
        f.write(f"DISTRIBUTED VECTOR DB BENCHMARK SUITE REPORT\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Target URLs: {urls_str}\n")
        f.write(f"Initial Count: {initial_count}\n")
        f.write(f"Post-Warmup Count: {post_warmup_count}\n")
        f.write("========================================================\n\n")

    for name, config in SCENARIOS.items():
        logger.info(f"--- Running Scenario: {name.upper()} ---")
        logger.info(f"Description: {config['desc']}")
        
        pre_test_count = get_vector_count(urls[0])
        logger.info(f"Pre-Test Count: {pre_test_count}")

        start_time = time.time()
        results = run_benchmark(
            client, 
            data, 
            config['duration'], 
            config['concurrency'], 
            config['mix']
        )
        total_time = time.time() - start_time
        
        post_test_count = get_vector_count(urls[0])
        count_delta = post_test_count - pre_test_count
        logger.info(f"Post-Test Count: {post_test_count} (Delta: +{count_delta})")
        
        # Calculate summary metrics
        total_ops = len(results['insert']) + len(results['query']) + results['errors']['insert'] + results['errors']['query']
        throughput = total_ops / total_time if total_time > 0 else 0
        
        # Generate Report Section
        report_section = [
            f"--------------------------------------------------------",
            f"SCENARIO: {name.upper()}",
            f"Description: {config['desc']}",
            f"Duration: {config['duration']}s, Concurrency: {config['concurrency']}, Mix: {config['mix']}",
            f"Total Requests: {total_ops}",
            f"Throughput: {throughput:.2f} req/s",
            f"Vectors Stored Delta: +{count_delta} (Validation)",
            get_stats_summary("Insert", results['insert'], results['errors']['insert']),
            get_stats_summary("Query", results['query'], results['errors']['query']),
            f"--------------------------------------------------------\n"
        ]
        
        with open(output_file, "a") as f:
            f.write("\n".join(report_section) + "\n")
        
        logger.info(f"Finished {name}. Throughput: {throughput:.2f} req/s")
        # Cool down between tests
        time.sleep(2)

    logger.info(f"Benchmark suite completed. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Full Benchmark Suite")
    parser.add_argument("--urls", type=str, default="http://127.0.0.1:8001", help="Comma-separated server URLs")
    parser.add_argument("--data", type=str, default="embeddings.json", help="Path to data file")
    parser.add_argument("--output", type=str, default="benchmark_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    run_suite(args.urls, args.data, args.output)

if __name__ == "__main__":
    main()
