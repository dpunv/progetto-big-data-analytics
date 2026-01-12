import concurrent.futures
import json
import sys
import time

import server as sv

start_time = time.time()

print("starting client")
# read the data
data = []
with open("embeddings.json", "r") as f:
    data = json.load(f)

# configuration
num_vectors = 16000
num_vectors_before_clustering = 5000
num_servers = 8
replication_factor = 3
batch_size = 512
vectors = [(d["embedding"], d["text"]) for d in data[:num_vectors]]
print("data read")

import os

# Qdrant Storage Configuration
# Options: ":memory:" for in-memory, or a local path (e.g., "./qdrant_data") for persistence.
# qdrant_url = ":memory:"
base_qdrant_url = "./qdrant_data"
# Override for previous hardcoded value
# base_qdrant_url = ":memory:"
print("configuration defined")

# Check for multi-node mode (env var set by run.py)
multi_node_mode = os.environ.get("MULTI_NODE_MODE", "false").lower() == "true"
if multi_node_mode:
    print("Running in MULTI-NODE mode (separate Qdrant instances)")
else:
    print(f"Running in SINGLE-INSTANCE mode: {base_qdrant_url}")

# create servers
servers = []
for i in range(num_servers):
    if multi_node_mode:
        # http://localhost:6333, 6335, ...
        port = 6333 + (i * 2)
        node_url = f"http://localhost:{port}"
    else:
        node_url = base_qdrant_url

    servers.append(
        sv.Server(
            i,
            i == 0,
            num_vectors_before_clustering,
            replication_factor,
            port=8000 + i,
            qdrant_url=node_url,
        )
    )
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


import math

total_batches = math.ceil(num_vectors / batch_size)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(total_batches):
        batch_vectors = vectors[batch_size * i : batch_size * (i + 1)]
        server_idx = i % len(servers)
        futures.append(
            executor.submit(send_batch, server_idx, batch_vectors, i + 1, total_batches)
        )

    concurrent.futures.wait(futures)

print("all batch sent")

# Give some time for async processing on server side to complete
# Wait for queues to drain
print("Waiting for servers to process all vectors...")
while True:
    total_queue = sum(s.get_queue_size() for s in servers)
    clustering = any(s.is_clustering() for s in servers)
    if total_queue == 0 and not clustering:
        break
    print(f"\rQueue size: {total_queue} (Clustering: {clustering})", end="")
    time.sleep(1)
print("\nProcessing complete")

# query a vector:
query_vector = [vectors[0][0]]
results = [
    (id, payload, distance)
    for _, id, payload, distance in servers[0].query_from_client(query_vector)
]
print(
    "vector queried:\n",
    "\n".join(
        [" -> ".join([str(distance), payload]) for _, payload, distance in results]
    ),
)

correspondence = sorted(
    [
        (sv.cosine_similarity(v, query_vector[0]), payload)
        for v, payload in vectors[:num_vectors]
    ],
    key=lambda x: x[0],
    reverse=True,
)[:5]
print(
    "correspondence:\n",
    "\n".join(
        [" -> ".join([str(distance), payload]) for distance, payload in correspondence]
    ),
)

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
