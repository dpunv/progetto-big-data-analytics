import requests
import numpy as np
import uuid
import time
import json
import sys
import os  # <-- NEW IMPORT
import utils

# --- Configuration ---
try:
    NUM_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
except ValueError:
    print("Invalid argument. Using default of 3 nodes.")
    NUM_NODES = 3

print(f"--- Running Qdrant App for {NUM_NODES} nodes ---")

NUM_VECTORS = 140000
VECTOR_SIZE = 384
BATCH_SIZE = 128

# --- NEW: Training Configuration ---
TRAINING_VECTORS = 10000  # Fixed size for training
CENTROIDS_FILE = 'centroids.json'  # File to save/load "weights"
# -----------------------------------

if VECTOR_SIZE < NUM_NODES:
    print(f"Error: VECTOR_SIZE ({VECTOR_SIZE}) must be >= NUM_NODES ({NUM_NODES})")
    print("Please increase VECTOR_SIZE in qdrant_app.py and server.py")
    sys.exit(1)

# --- Helper Functions ---
data = {}
try:
    with open('embeddings.json', 'r') as f:
        data = json.load(f)
    if len(data) < NUM_VECTORS + 1:
        print(f"Warning: embeddings.json has only {len(data)} items, but {NUM_VECTORS}+1 are needed.")
        for i in range(len(data), NUM_VECTORS + 1):
            data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})
except FileNotFoundError:
    print("embeddings.json not found. Generating random data...")
    for i in range(NUM_VECTORS + 1):
        data.append({"embedding": np.random.rand(VECTOR_SIZE).tolist()})

# --- NEW LOGIC: Train-Once Centroids (Load or Calculate) ---
print("\n--- 0. Loading/Computing Node Centroids ---")
NODE_VECS = []  # This will contain our centroids

try:
    # 1. Check if centroids file exists
    if os.path.exists(CENTROIDS_FILE):
        print(f"📂 Found existing centroids file '{CENTROIDS_FILE}'...")
        with open(CENTROIDS_FILE, 'r') as f:
            NODE_VECS = json.load(f)
        
        # 2. Verify the number of centroids matches requested nodes
        if len(NODE_VECS) == NUM_NODES:
            print(f"✅ Successfully loaded {len(NODE_VECS)} centroids from '{CENTROIDS_FILE}'.")
            print(f"   Skipping K-means training (using cached centroids).")
        else:
            print(f"⚠️  Warning: '{CENTROIDS_FILE}' has {len(NODE_VECS)} centroids, but {NUM_NODES} are needed.")
            print("   Forcing re-training...")
            NODE_VECS = []  # Force re-training
    
    # 3. If NODE_VECS is empty (file not found or mismatch), train
    if not NODE_VECS:
        if not os.path.exists(CENTROIDS_FILE):
            print(f"Centroids file not found. Starting new training...")
        
        # 4. Check we have enough data for training
        if len(data) < TRAINING_VECTORS:
            print(f"❌ ERROR: Insufficient data for training.")
            print(f"   Need {TRAINING_VECTORS} vectors, but 'embeddings.json' only has {len(data)}.")
            print(f"   Please run 'python take_some.py' to generate more embeddings.")
            sys.exit(1)
        
        print(f"🎓 Starting K-Means training on {TRAINING_VECTORS} vectors...")
        start_train = time.time()
        
        # 5. Extract vectors for training
        training_data_vectors = [i['embedding'] for i in data[:TRAINING_VECTORS]]
        
        # 6. Calculate centroids
        calculated_centroids_np = utils.find_kmeans_centroids(training_data_vectors, NUM_NODES)
        NODE_VECS = calculated_centroids_np.tolist()
        
        train_time = time.time() - start_train
        
        # 7. Save new centroids ("weights") to disk
        with open(CENTROIDS_FILE, 'w') as f:
            json.dump(NODE_VECS, f, indent=2)
        
        file_size_kb = os.path.getsize(CENTROIDS_FILE) / 1024
        print(f"✅ Training completed in {train_time:.2f}s.")
        print(f"   {len(NODE_VECS)} centroids saved to '{CENTROIDS_FILE}' ({file_size_kb:.2f} KB).")

except Exception as e:
    print(f"❌ FATAL ERROR during centroids loading/training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- END NEW LOGIC ---

# --- Dynamic Node Configuration ---
NODE_URLS = []

for i in range(1, NUM_NODES + 1):
    NODE_URLS.append(f"http://localhost:{8000 + i}")

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
            node_vec = NODE_VECS[i]  # Use loaded/calculated vectors
            
            r_set = requests.post(f"{node_url}/set_node_vector", json=node_vec, timeout=5)
            r_set.raise_for_status()
            print(f"  {node_id}: Set representative vector (first 3 dims: {[round(v, 3) for v in node_vec[:3]]}...)")

        # 2. Register all peers with all other peers (N * (N-1) requests)
        print("\nRegistering peers...")
        for i in range(NUM_NODES):
            host_id = f"node{i+1}"
            host_url = NODE_URLS[i]
            
            peers_registered = 0
            for j in range(NUM_NODES):
                if i == j:
                    continue
                
                peer_id = f"node{j+1}"
                peer_url = NODE_URLS[j]
                peer_vec = NODE_VECS[j]  # Use loaded/calculated vectors
                
                payload = {
                    "peer_id": peer_id,
                    "peer_url": peer_url,
                    "node_vector": peer_vec
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


# --- MODIFIED: Replaced insert_vectors with insert_vectors_bulk ---
def insert_vectors_bulk():
    """
    Insert vectors in batches, round-robin sending them to nodes for routing.
    """
    print(f"--- 2. Inserting {NUM_VECTORS} Vectors (Size {VECTOR_SIZE}) in batches of {BATCH_SIZE} ---")
    
    insertions = [] # This will store the final node_id for each vector
    
    num_batches = (NUM_VECTORS + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, NUM_VECTORS)
        
        # Determine which node to send to (round-robin for load balancing)
        node_url_index = i % NUM_NODES
        node_url = NODE_URLS[node_url_index]
        sent_to_node_id = f"node{node_url_index + 1}"
        
        # --- Create the batch ---
        batch_payload = []
        for j in range(start_index, end_index):
            final_vec = data[j]['embedding']
            vector_data = {
                "id": str(uuid.uuid4()),
                "vector": final_vec,
                "payload": {
                    "source_type": "json_data",
                    "sent_to_node": sent_to_node_id,
                    "index": j
                }
            }
            batch_payload.append(vector_data)
        
        if not batch_payload:
            continue # Should not happen, but good check
            
        try:
            # Use the new /add_vectors_bulk endpoint
            response = requests.post(
                f"{node_url}/add_vectors_bulk",
                json=batch_payload, # Send the list of vectors directly
                timeout=30 # Increased timeout for bulk
            )
            response.raise_for_status()
            
            # Log the routing action
            res_data = response.json()
            # Server returns a dict like {"batches": {"node1": 50, "node3": 78}}
            batches = res_data.get('batches', {})
            
            # This is the crucial part: update the insertions list
            total_in_batch = 0
            for node_id, count in batches.items():
                insertions.extend([node_id] * count)
                total_in_batch += count
            
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                print(f"Batch {i + 1}/{num_batches} inserted... (Sent to {sent_to_node_id}, "
                      f"server routed {total_in_batch} vecs to {len(batches)} nodes)")
                
        except requests.exceptions.RequestException as e:
            print(f"!!! Error inserting batch {i+1}: {e}")
            
    print(f"Vector bulk insertion complete.\n")
    return insertions
# --- END MODIFICATION ---


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
    
    # --- MODIFIED: Call the bulk function ---
    insertions = insert_vectors_bulk()
    # --- END MODIFICATION ---
    
    # --- NEW: Add a delay to allow background tasks to finish ---
    print("--- Waiting 5s for background insertions to settle... ---")
    time.sleep(5) 
    # In a real system, you might need a more robust check
    # --- END NEW ---
    
    check_counts(insertions)
    run_queries()
    
    end_time = time.time()
    print(f"\n--- Test Complete in {end_time - start_time:.2f} seconds ---")

# --- Main Execution ---
if __name__ == "__main__":
    main_app()