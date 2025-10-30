import requests
import numpy as np
import uuid
import time
import json
import sys
import utils

# --- Configuration ---
# The number of nodes is now read from the command line
try:
    # Read N from the first command-line argument
    NUM_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid argument. Using default of 3 nodes.")
    NUM_NODES = 3

print(f"--- Running Qdrant App for {NUM_NODES} nodes ---")

NUM_VECTORS = 9950 # Use 1000 for a quick test
VECTOR_SIZE = 384    # IMPORTANT: This MUST match the vector_size in run_node()

# Check if VECTOR_SIZE is large enough for our one-hot encoding
if VECTOR_SIZE < NUM_NODES:
    print(f"Error: VECTOR_SIZE ({VECTOR_SIZE}) must be >= NUM_NODES ({NUM_NODES})")
    print("Please increase VECTOR_SIZE in qdrant_app.py and server.py")
    sys.exit(1)

# --- Dynamic Node Configuration ---
NODE_URLS = []
NODE_VECS = []

for i in range(1, NUM_NODES + 1):
    # e.g., http://localhost:8001, http://localhost:8002, ...
    NODE_URLS.append(f"http://localhost:{8000 + i}")
    
    # Create a unique representative vector for each node (one-hot encoding)
    # node1 -> [1.0, 0.0, 0.0, ...]
    # node2 -> [0.0, 1.0, 0.0, ...]
    # node3 -> [0.0, 0.0, 1.0, ...]
    vec = [0.0] * VECTOR_SIZE
    vec[i-1] = 1.0  # Set the i-th dimension to 1.0
    NODE_VECS.append(vec)

# --- Helper Functions ---
data = {}
try:
    with open('embeddings.json', 'r') as f:
        data = json.load(f)
    if len(data) < NUM_VECTORS + 1:
         print(f"Warning: embeddings.json has only {len(data)} items, but {NUM_VECTORS}+1 are needed.")
         # Pad with random data if insufficient
         for i in range(len(data), NUM_VECTORS + 1):
             data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})
except FileNotFoundError:
    print("embeddings.json not found. Generating random data...")
    for i in range(NUM_VECTORS + 1):
        data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})


def register_peers():
    """
    Dynamically set each node's representative vector and register them with each other.
    """
    print("--- 1. Setting Node Vectors & Registering Peers ---")
    try:
        # 1. Set the representative vector for every node
        print("Setting node vectors...")
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_url = NODE_URLS[i]
            node_vec = NODE_VECS[i]
            
            r_set = requests.post(f"{node_url}/set_node_vector", json=node_vec, timeout=5)
            r_set.raise_for_status()
            print(f"  {node_id}: Set representative vector (index {i} = 1.0)")

        # 2. Register all peers with all other peers (N * (N-1) requests)
        print("\nRegistering peers...")
        for i in range(NUM_NODES): # This is the "host" node
            host_id = f"node{i+1}"
            host_url = NODE_URLS[i]
            
            peers_registered = 0
            for j in range(NUM_NODES): # This is the "peer" node
                if i == j:
                    continue # Don't register with self
                
                peer_id = f"node{j+1}"
                peer_url = NODE_URLS[j]
                peer_vec = NODE_VECS[j]
                
                payload = {
                    "peer_id": peer_id,
                    "peer_url": peer_url,
                    "node_vector": peer_vec # Send the peer's vector to the host
                }
                
                r_reg = requests.post(f"{host_url}/register_peer", json=payload, timeout=5)
                r_reg.raise_for_status()
                peers_registered += 1
            
            print(f"  Host {host_id}: Registered {peers_registered} peers.")

        print("Peers registered and vectors exchanged successfully.\n")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error setting/registering peers: {e}")
        print("!!! Please ensure all servers are running.")
        sys.exit(1)


def insert_vectors():
    """
    Insert vectors, round-robin sending them to nodes for routing.
    """
    print(f"--- 2. Inserting {NUM_VECTORS} Vectors (Size {VECTOR_SIZE}) ---")
    
    insertions = []
    
    for i in range(NUM_VECTORS):
        # Determine which node to send to (round-robin for load balancing)
        node_url_index = i % NUM_NODES
        node_url = NODE_URLS[node_url_index]
        sent_to_node_id = f"node{node_url_index + 1}"
        
        # Get embedding from loaded data
        final_vec = data[i]['embedding']
        
        vector_data = {
            "id": str(uuid.uuid4()),
            "vector": final_vec,
            "payload": {
                "source_type": "json_data",
                "sent_to_node": sent_to_node_id,
                "index": i
            }
        }
        
        try:
            # Use the /add_vector endpoint (which handles routing)
            response = requests.post(
                f"{node_url}/add_vector",
                json=vector_data,
                timeout=10
            )
            response.raise_for_status()
            
            # Log the routing action
            res_data = response.json()
            stored_at = res_data.get('node_id')
            insertions.append(stored_at)
            
            if (i + 1) % 100 == 0:
                print(f"Inserted {i + 1}/{NUM_VECTORS}... (Last: sent to {sent_to_node_id}, stored at {stored_at})")
                
        except requests.exceptions.RequestException as e:
            print(f"!!! Error inserting vector {i}: {e}")
            
    print(f"Vector insertion complete.\n")
    return insertions


def check_counts(insertions: list):
    """Check the final vector counts on each node."""
    print("--- 3. Checking Vector Counts ---")
    try:
        all_counts = []
        total_count = 0
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_url = NODE_URLS[i]
            
            r = requests.get(f"{node_url}/count", timeout=5)
            r.raise_for_status()
            count = r.json().get('count', 0)
            
            all_counts.append((node_id, count))
            total_count += count

        print("Vector counts per node:")
        for node_id, count in all_counts:
            print(f"  - {node_id} Count: {count}")
        
        print(f"Total Vectors Stored: {total_count} (Expected: {NUM_VECTORS})")
        
        # Check from test script perspective
        print("\n(Client-side insertion log check):")
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            script_count = insertions.count(node_id)
            print(f"  - {node_id} received: {script_count}")
        
        if total_count == NUM_VECTORS:
            print("✅  Total counts match total inserted vectors.")
        else:
            print("⚠️  Counts do NOT match total inserted vectors!")

        # Check if routing was *roughly* balanced
        expected_avg = NUM_VECTORS / NUM_NODES
        tolerance = NUM_VECTORS * 0.1 # 10% tolerance
        is_balanced = True
        
        for node_id, count in all_counts:
             if abs(count - expected_avg) > tolerance:
                 is_balanced = False

        if is_balanced:
            print(f"✅  Distribution is roughly balanced (within 10%).")
        else:
            print(f"⚠️  Distribution seems unbalanced!")
        print("")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error checking counts: {e}")


def run_queries():
    """Run a single federated search query."""
    print("--- 4. Running Federated Search Query ---")
    
    # Use the last embedding as the query vector
    query_vector = data[NUM_VECTORS]['embedding']
    # print(f"Query Vector (first 5 dims): {[round(x, 2) for x in query_vector[:5]]}...\n")

    try:
        # Run Federated Search (always from Node 1, doesn't matter which)
        r_fed = requests.post(
            f"{NODE_URLS[0]}/search/federated",
            json=query_vector,
            params={"top_k": 5} # Request top 5 from EACH node
        )
        r_fed.raise_for_status()
        fed_results = r_fed.json()
        
        print(f"Federated search complete:")
        results_data = fed_results.get('results', {})
        
        all_results = []
        best_overall_node = "N/A"
        best_overall_score = -2.0 # Cosine similarity can be -1
        
        for i in range(NUM_NODES):
            node_id = f"node{i+1}"
            node_vector = NODE_VECS[i]
            node_similarity = utils.cosine_similarity(query_vector, node_vector)
            node_results_list = results_data.get(node_id, [])
            node_results_count = len(node_results_list)
            
            max_score = node_results_list[0].get('score', -1) if node_results_count > 0 else -1
            min_score = node_results_list[-1].get('score', -1) if node_results_count > 0 else -1
            
            print(f"  - Results from {node_id}: {node_results_count} (Best: {max_score:.4f}, Worst: {min_score:.4f}, Node: {node_similarity})")
            
            if max_score > best_overall_score:
                best_overall_score = max_score
                best_overall_node = node_id
        
        print(f"\n  - Total Results Found: {fed_results.get('total_results')}")
        print(f"ℹ️  {best_overall_node} had the best matching vector (Score: {best_overall_score:.4f}).")

    except requests.exceptions.RequestException as e:
        print(f"!!! Error running federated query: {e}")


def main_app():
    print(f"Starting Qdrant Smart Sharding Test ({NUM_NODES} Nodes)...")
    print(f"Please ensure all servers and DBs are running.")
    print(f"Test will insert {NUM_VECTORS} vectors.\n")
    
    # Give servers a moment to start
    time.sleep(2)
    
    start_time = time.time()
    
    register_peers()
    insertions = insert_vectors()
    check_counts(insertions)
    run_queries()
    
    end_time = time.time()
    print(f"\n--- Test Complete in {end_time - start_time:.2f} seconds ---")

# --- Main Execution ---
if __name__ == "__main__":
    main_app()
