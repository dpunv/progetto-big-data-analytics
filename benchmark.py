import time
import json
import random
import requests
import argparse
import threading
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from compound_types import ListOfVectorsWithPayload, ListOfVectors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        # Format payload as expected by the server (ListOfVectorsWithPayload)
        # Type: List[Tuple[Vector, Payload]] where Payload is str (from compound_types.py)
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
    
    # Normalize ratios if needed, but assuming user gives correctly summing to 1.0 or weights
    total_weight = insert_ratio + query_ratio
    if total_weight == 0:
        logger.error("Invalid mix configuration: weights sum to 0")
        return

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

def print_stats(name: str, latencies: List[float], errors: int = 0):
    print(get_stats_summary(name, latencies, errors))

def main():
    parser = argparse.ArgumentParser(description="Distributed Vector DB Benchmark")
    parser.add_argument("--urls", type=str, default="http://127.0.0.1:8001", help="Comma-separated server URLs")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--mix", type=str, default="insert:0.75,query:0.25", help="Workload mix (e.g., insert:0.75,query:0.25)")
    parser.add_argument("--data", type=str, default="embeddings.json", help="Path to data file")
    
    args = parser.parse_args()
    
    urls = [u.strip() for u in args.urls.split(',')]
    mix_parts = [p.split(':') for p in args.mix.split(',')]
    mix = {k: float(v) for k, v in mix_parts}
    
    data = load_data(args.data)
    if not data:
        return

    client = BenchmarkClient(urls)
    
    start_time = time.time()
    results = run_benchmark(client, data, args.duration, args.concurrency, mix)
    total_time = time.time() - start_time
    
    total_ops = len(results['insert']) + len(results['query'])
    throughput = total_ops / total_time
    
    print("\n" + "="*40)
    print(f"BENCHMARK COMPLETED")
    print("="*40)
    print(f"Total Duration:  {total_time:.2f} s")
    print(f"Total Requests:  {total_ops}")
    print(f"Throughput:      {throughput:.2f} req/s")
    
    print_stats("Insert", results['insert'], results['errors']['insert'])
    print_stats("Query", results['query'], results['errors']['query'])
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
