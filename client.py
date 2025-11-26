import numpy as np
import requests
import json
from compound_types import *
import time
import metrics
import utils
import argparse
import logging
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import math

logger = logging.getLogger(__name__)

request_ids = 0
def get_id():
    global request_ids
    request_ids += 1
    return request_ids

class Server:
    def __init__(self, id, url, grpc_url, is_coordinator):
        self.id = id
        self.url = url
        self.grpc_url = grpc_url
        self.is_coordinator = is_coordinator
        self.peers = []
    
    def register_peers(self, peers_list):
        self.peers = peers_list
        payload = {
            'id': get_id(),
            'peers': self.peers
        }
        try:
            requests.post(f'{self.url}/register_peers', json=payload)
        except Exception as e:
            logger.error(f"Failed to register peers on {self.id}: {e}")
    
    def send_vectors(self, vectors_list: ListOfVectorsWithPayload):
        payload = {
            'id': get_id(),
            'content': vectors_list
        }
        try:
            requests.post(f'{self.url}/add', json=payload)
        except Exception as e:
            logger.error(f"Failed to send vectors to {self.id}: {e}")
            raise
        
    
    def query_vectors(self, vectors_list: ListOfVectors):
        payload = {
            'id': get_id(),
            'query': vectors_list,
            'topk': 5
        }
        try:
            return requests.post(f'{self.url}/query', json=payload)
        except Exception as e:
            logger.error(f"Query failed on {self.id}: {e}")
            return None

    def get_count(self):
        try:
            return requests.get(f'{self.url}/count')
        except Exception as e:
            logger.error(f"Count failed on {self.id}: {e}")
            return None

def run_client(server, queries, topk=5):   
    logger.info(f"Starting queries to {server.url}...")
    start_time = time.time()
    response=server.query_vectors(queries)
    elapsed = time.time() - start_time
    if elapsed > 0:
        avg_lat = elapsed / len(queries)
        logger.info(f"Completed {len(queries)} queries in {elapsed:.2f}s (Avg Latency: {avg_lat:.4f}s)")
        
        print("\n" + "="*40)
        print(" CLIENT SIDE METRICS REPORT")
        print("="*40)
        print(f" Total Queries:   {len(queries)}")
        print(f" Total Time:      {elapsed:.2f} s")
        print(f" Avg Latency:     {avg_lat:.4f} s")
        print("="*40 + "\n")
    
        with open("logs/client_metrics.txt", "w") as f:
            f.write(f"CLIENT_AVG_LATENCY={avg_lat}\n")
            f.write(f"CLIENT_TOTAL_REQ={len(queries)}\n")
    else:
        logger.warning("No successful queries recorded.")

    return response

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, default="logs/client.log", help="Path to log file")
    args = parser.parse_args()

    # Setup Logging
    handlers = []
    log_level = logging.INFO

    if not utils.LOGGING_ENABLED:
        log_level = logging.CRITICAL
    elif args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    
    logger.info('Client application started')
    
    config = []
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        data = []
        with open('embeddings.json') as f:
            data = json.load(f)
        logger.info('Data and configuration loaded')
    except Exception as e:
        logger.critical(f"Failed to load config/data: {e}")
        sys.exit(1)

    servers = []
    for server in config['servers']:
        servers.append(Server(server['id'], server['url'], server['grpc_url'], server['is_coordinator']))
    
    for server in servers:
        logger.info(f'Sending peers to server {server.id}')
        server.register_peers([(s.id, s.url, s.grpc_url) for s in servers if s.id != server.id])
    
    logger.info('Peers registered on all servers')

    batch_size_send = config['batch_size']
    batch_size_send_retry = config['batch_size_retry']

    vector_sent = 0
    total_batches = math.ceil(min(len(data), config['num_vectors']) / batch_size_send)
    
    # Log initial statistics
    unique_vectors_to_send = min(len(data), config['num_vectors'])
    expected_total_with_replicas = unique_vectors_to_send * config['replicas']
    logger.info("=" * 60)
    logger.info("INITIAL VECTOR STATISTICS")
    logger.info(f"Unique vectors to send: {unique_vectors_to_send}")
    logger.info(f"Replication factor: {config['replicas']}")
    logger.info(f"Expected total vectors (with replicas): {expected_total_with_replicas}")
    logger.info("=" * 60)
    
    # Prepare all batches with their target servers (round-robin assignment)
    batches_with_servers = []
    for i in range(total_batches):
        start_idx = i * batch_size_send
        end_idx = min((i + 1) * batch_size_send, unique_vectors_to_send)
        batch_data = [(el['embedding'], el['text']) for el in data[start_idx: end_idx]]
        target_server = servers[i % len(servers)]
        batches_with_servers.append((batch_data, target_server, i + 1))
        vector_sent += len(batch_data)
    
    # Calculate number of threads: min(num_servers, cpu_count * 2)
    num_threads = min(len(servers), os.cpu_count() * 2)
    logger.info(f"Starting parallel batch sending with {num_threads} threads for {total_batches} batches across {len(servers)} servers")
    
    # Worker function for thread pool
    def send_batch_worker(args):
        batch_data, server, batch_num = args
        thread_name = threading.current_thread().name
        logger.info(f'[{thread_name}] Sending batch {batch_num}/{total_batches} ({len(batch_data)} vectors) to {server.id}')
        
        try:
            # First attempt: Send full batch (likely 1024)
            server.send_vectors(batch_data)
            logger.info(f'[{thread_name}] Batch {batch_num} sent successfully to {server.id}')
            return (batch_num, True, None)
            
        except Exception as e:
            # Fallback logic: If batch is large, split into smaller chunks (batch_size_retry)
            if len(batch_data) > batch_size_send_retry:
                logger.warning(f'[{thread_name}] Batch {batch_num} FAILED to {server.id} with size {len(batch_data)}. Retrying with batch_size={batch_size_send_retry}... Error: {e}')
                
                fallback_batch_size = batch_size_send_retry
                total_sub_batches = (len(batch_data) + fallback_batch_size - 1) // fallback_batch_size
                
                try:
                    for i in range(total_sub_batches):
                        sub_batch = batch_data[i * fallback_batch_size : (i + 1) * fallback_batch_size]
                        logger.info(f'[{thread_name}] Sending sub-batch {i+1}/{total_sub_batches} of batch {batch_num} to {server.id}')
                        server.send_vectors(sub_batch)
                    
                    logger.info(f'[{thread_name}] Batch {batch_num} sent successfully (via fallback) to {server.id}')
                    return (batch_num, True, None)
                    
                except Exception as e2:
                    logger.error(f'[{thread_name}] Batch {batch_num} FAILED during fallback to {server.id}: {e2}')
                    return (batch_num, False, str(e2))
            else:
                # If batch is already small, just fail
                logger.error(f'[{thread_name}] Batch {batch_num} FAILED to {server.id}: {e}')
                return (batch_num, False, str(e))
    
    # Execute parallel sending
    start_time = time.time()
    failed_batches = []
    
    with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='BatchSender') as executor:
        results = executor.map(send_batch_worker, batches_with_servers)
        for batch_num, success, error in results:
            if not success:
                failed_batches.append((batch_num, error))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Parallel sending completed in {elapsed_time:.2f}s")
    
    if failed_batches:
        logger.error(f"Failed to send {len(failed_batches)} batches: {failed_batches}")
    else:
        logger.info("All batches sent successfully!")
    
    time.sleep(1)

    logger.info('Sending query')

    query_vector = data[0]['embedding']
    res_obj = run_client(servers[0], [query_vector])
    #res_obj = servers[0].query_vectors([query_vector])
    if res_obj:
        res = res_obj.json()['results']
        for result in res:
            logger.info(f"Result: {result['id']}: {result['score']} -> {result['payload']['string']}")

    logger.info("### FOR CORRESPONDENCE ###")

    distances_calcs = [(vector, utils.cosine_similarity(vector['embedding'], query_vector)) for vector in data[:vector_sent]]
    distances_calcs.sort(key=lambda x: x[1], reverse=True)
    for v, d in distances_calcs[:5]:
        logger.info(f"Ground Truth: {d} -> {v['text']}")

    logger.info('Getting vector counts (polling for consistency)...')
    expected_total = vector_sent * config['replicas']
    if int(config['num_before_clustering']) <= vector_sent:
        max_retries = 30
        for i in range(max_retries):
            count_res = servers[1 if len(servers) > 1 else 0].get_count()
            if count_res:
                counts = count_res.json()
                total = sum([count for _, count in counts.items()])
                logger.info(f'Counts: {counts} - Total: {total}/{expected_total}')
                
                if total >= expected_total:
                    break
            time.sleep(2)
            
        
        # Final statistics
        logger.info("=" * 60)
        logger.info("FINAL VECTOR STATISTICS")
        logger.info(f"Unique vectors sent: {vector_sent}")
        logger.info(f"Expected total with replicas: {vector_sent * config['replicas']}")
        logger.info(f"Actual total stored: {total}")
        
        if total < vector_sent * config['replicas']:
            missing = (vector_sent * config['replicas']) - total
            logger.warning(f"Missing vectors: {missing} ({missing/(vector_sent * config['replicas'])*100:.2f}%)")
        elif total > vector_sent * config['replicas']:
            extra = total - (vector_sent * config['replicas'])
            logger.info(f"Extra vectors: {extra} ({extra/(vector_sent * config['replicas'])*100:.2f}%)")
        else:
            logger.info("Perfect match: all expected replicas are stored!")
        
    logger.info("=" * 60)

    logger.info('Client application end')

if __name__ == '__main__':
    main()