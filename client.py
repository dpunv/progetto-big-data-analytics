import requests
import json
from compound_types import *
import time
import utils
import argparse
import logging
import sys

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
        logger.info(f'Sending {len(vectors_list)} vectors to {self.id}')
        payload = {
            'id': get_id(),
            'content': vectors_list
        }
        try:
            requests.post(f'{self.url}/add', json=payload)
            logger.info('Vectors sent successfully')
        except Exception as e:
            logger.error(f"Failed to send vectors to {self.id}: {e}")
        
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, default="logs/client.log", help="Path to log file")
    args = parser.parse_args()

    # Setup Logging
    handlers = []
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
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
    vector_sent = 0
    total_batches = min(len(data), config['num_vectors']) // batch_size_send
    
    # Log initial statistics
    unique_vectors_to_send = min(len(data), config['num_vectors'])
    expected_total_with_replicas = unique_vectors_to_send * config['replicas']
    logger.info("=" * 60)
    logger.info("INITIAL VECTOR STATISTICS")
    logger.info(f"Unique vectors to send: {unique_vectors_to_send}")
    logger.info(f"Replication factor: {config['replicas']}")
    logger.info(f"Expected total vectors (with replicas): {expected_total_with_replicas}")
    logger.info("=" * 60)
    
    for i in range(min(len(data), config['num_vectors'] // batch_size_send)):
        batch = [(el['embedding'], el['text']) for el in data[i * batch_size_send: (i+1) * (batch_size_send)]]
        logger.info(f'Sending vector batch {i+1} / {total_batches}: {len(batch)} vectors')

        vector_sent += ((i+1) * (batch_size_send)) - (i * batch_size_send)
        servers[i%len(servers)].send_vectors(batch)
    
    time.sleep(1)

    logger.info('Sending query')

    query_vector = data[0]['embedding']
    res_obj = servers[0].query_vectors([query_vector])
    if res_obj:
        res = res_obj.json()['results']
        for result in res:
            logger.info(f"Result: {result['id']}: {result['score']} -> {result['payload']['string']}")

    logger.info("### FOR CORRESPONDENCE ###")

    distances_calcs = [(vector, utils.cosine_similarity(vector['embedding'], query_vector)) for vector in data[:vector_sent]]
    distances_calcs.sort(key=lambda x: x[1], reverse=True)
    for v, d in distances_calcs[:5]:
        logger.info(f"Ground Truth: {d} -> {v['text']}")

    logger.info('Getting vector counts...')
    count_res = servers[1 if len(servers) > 1 else 0].get_count()
    if count_res:
        counts = count_res.json()
        total = sum([count for _, count in counts.items()])
        logger.info(f'Counts: {counts} - Total: {total}')
        
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