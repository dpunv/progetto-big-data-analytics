import requests
import json
from compound_types import *

request_ids = 0
def get_id():
    global request_ids
    request_ids += 1
    return request_ids

class Server:
    def __init__(self, id, url, is_coordinator):
        self.id = id
        self.url = url
        self.is_coordinator = is_coordinator
        self.peers = []
    
    def register_peers(self, peers_list):
        self.peers = peers_list
        payload = {
            'id': get_id(),
            'peers': self.peers
        }
        requests.post(f'{self.url}/register_peers', json=payload)
    
    def send_vectors(self, vectors_list: ListOfVectorsWithPayload):
        start_color = '\033[33m'
        end_color = '\033[0m'
        print(f'{start_color}send vectors start{end_color}')
        payload = {
            'id': get_id(),
            'content': vectors_list
        }
        print(f'{start_color}payload object created{end_color}')

        requests.post(f'{self.url}/add', json=payload)

        print(f'{start_color}end vector send{end_color}')
        
    
    def query_vectors(self, vectors_list: ListOfVectors):
        payload = {
            'id': get_id(),
            'query': vectors_list,
            'topk': 5
        }
        return requests.post(f'{self.url}/query', json=payload)

def main():
    start_color = '\033[33m'
    end_color = '\033[0m'
    
    print(f'{start_color}client started{end_color}')
    
    config = []
    with open('config.json', 'r') as f:
        config = json.load(f)
    data = []
    with open('embeddings.json') as f:
        data = json.load(f)
        
    
    print(f'{start_color}data and configuration loaded{end_color}')

    servers = []
    for server in config['servers']:
        servers.append(Server(server['id'], server['url'], server['is_coordinator']))
    
    for server in servers:
        print(f'{start_color}sending peers to server {server.id}{end_color}')
        server.register_peers([(s.id, s.url) for s in servers if s.id != server.id])
    
    print(f'{start_color}peers registered{end_color}')

    batch_size_send = config['batch_size']
    
    for i in range(min(len(data), config['num_vectors'] // batch_size_send) - 1):
        batch = [(el['embedding'], el['text']) for el in data[i * batch_size_send: (i+1) * (batch_size_send)]]
        print(f'{start_color}sending vector batch {i+1} / {min(len(data), config['num_vectors'] // batch_size_send) - 1}: {len(batch)} vectors {end_color}')

        servers[i%len(servers)].send_vectors(batch)
    
    print(f'{start_color}sending query{end_color}')

    print(servers[0].query_vectors([data[-1]['embedding']]))

    print(f'{start_color}client application end{end_color}')

if __name__ == '__main__':
    main()
