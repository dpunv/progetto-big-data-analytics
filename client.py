import server as sv
import json
import sys
from typing import List
import concurrent.futures
import time

start_time = time.time()

print("starting client")
# read the data
data = []
with open('embeddings.json', 'r') as f:
    data = json.load(f)
vectors = [(d['embedding'], d['text']) for d in data]
print("data read")

# configuration
num_vectors = 32768
num_vectors_before_clustering = 8192
num_servers = 3
replication_factor = 2
batch_size = 512
print("configuration defined")

# create servers
servers = []
for i in range(num_servers):
    servers.append(sv.Server(i, i==0, num_vectors_before_clustering, replication_factor, 8000+i))
print("servers started")

# register peers
for server in servers:
    for peer in servers:
        if server.get_id() == peer.get_id():
            continue
        server.add_peer(peer)
print("peers registered")
time.sleep(5)

# add vectors
def send_batch(server_idx, batch_vectors, batch_idx, total_batches):
    servers[server_idx].receive_from_client(batch_vectors)
    print(f"batch {batch_idx}/{total_batches} sent")

total_batches = int(num_vectors/batch_size)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(total_batches):
        batch_vectors = vectors[batch_size*i:batch_size*(i+1)]
        server_idx = i % len(servers)
        futures.append(executor.submit(send_batch, server_idx, batch_vectors, i+1, total_batches))
    
    concurrent.futures.wait(futures)

print("all batch sent")

# Give some time for async processing on server side to complete
time.sleep(60) 

# query a vector:
query_vector = [vectors[0][0]]
results = [(id, payload, distance) for _, id, payload, distance in servers[0].query_from_client(query_vector)]
print("vector queried:\n", "\n".join([' -> '.join([str(distance), payload]) for _, payload, distance in results]))

correspondence = sorted([(sv.cosine_similarity(v, query_vector[0]), payload) for v, payload in vectors[:num_vectors]], key=lambda x: x[0], reverse=True)[:5]
print("correspondence:\n", "\n".join([' -> '.join([str(distance), payload]) for distance, payload in correspondence]))

print("counting vectors per server:")
total = 0
for s in servers:
    total += s.count()
    print(f"\t{s.get_id()} -> {s.count()}")
print(f"total: {total}")
print(f"expected: {num_vectors * replication_factor}")

# Stop servers
for s in servers:
    s.stop()

end_time = time.time()
print(f"Total time: {end_time - start_time}", file=sys.stderr)
