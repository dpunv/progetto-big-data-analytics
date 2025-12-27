import argparse
import time
import sys
import json
import logging
import requests
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from benchmark import BenchmarkClient, load_data, run_benchmark, get_stats_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    start_time = time.time()
    
    # We use the existing run_benchmark logic but with 100% insert
    # Calculate approx duration needed: assume 200 inserts/sec -> 12.5s
    # Better: just insert in a loop until we reach target
    
    # Use ThreadPoolExecutor for concurrent warmup
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
            
            # Use random choice for simplicity in warmup
            item = random.choice(data)
            vector = item['embedding']
            text = item.get('text', '')
            
            try:
                # Using client.insert which returns (latency, success)
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
    # Ideally check /metrics or status endpoint for 'is_clustered'
    # but strictly requested is just to ensure we start effectively.



def run_suite(urls_str, data_path, output_file):
    logger.info(f"Loading data from {data_path}...")
    data = load_data(data_path)
    if not data:
        logger.error("Failed to load data. Aborting.")
        sys.exit(1)

    urls = [u.strip() for u in urls_str.split(',')]
    client = BenchmarkClient(urls)
    
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
        throughput = total_ops / total_time
        
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
