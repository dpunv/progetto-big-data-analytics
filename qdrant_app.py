import requests
import numpy as np
import uuid
import time
import json

# --- Configuration ---
NODE1_URL = "http://localhost:8001"
NODE2_URL = "http://localhost:8002"
NODE_URLS = [NODE1_URL, NODE2_URL]

NUM_VECTORS = 10000 # Use 1000 for a quick test
VECTOR_SIZE = 128    # IMPORTANT: This MUST match the vector_size in run_node()

# --- Node vectors for routing ---
# We still need representative vectors for the nodes
NODE1_VEC = [0.0 for _ in range(VECTOR_SIZE)]
NODE1_VEC[0] = 1.0
NODE2_VEC = [0.0 for _ in range(VECTOR_SIZE)]
NODE2_VEC[1] = 1.0

# --- Helper Functions ---

def register_peers():
    """
    Set each node's representative vector and register them with each other.
    """
    print("--- 1. Setting Node Vectors & Registering Peers ---")
    try:
        # 1. Set the representative vector for Node 1
        r_set1 = requests.post(f"{NODE1_URL}/set_node_vector", json=NODE1_VEC)
        r_set1.raise_for_status()
        print(f"Node 1: Set representative vector to {NODE1_VEC}")

        # 2. Set the representative vector for Node 2
        r_set2 = requests.post(f"{NODE2_URL}/set_node_vector", json=NODE2_VEC)
        r_set2.raise_for_status()
        print(f"Node 2: Set representative vector to {NODE2_VEC}")

        # 3. Register Node 2 with Node 1
        payload_n1 = {
            "peer_id": "node2",
            "peer_url": NODE2_URL,
            "node_vector": NODE2_VEC # Send Node 2's vector to Node 1
        }
        r1 = requests.post(f"{NODE1_URL}/register_peer", json=payload_n1)
        r1.raise_for_status()
        print(f"Node 1 -> Node 2 Registration: {r1.json().get('message')}")
        
        # 4. Register Node 1 with Node 2
        payload_n2 = {
            "peer_id": "node1",
            "peer_url": NODE1_URL,
            "node_vector": NODE1_VEC # Send Node 1's vector to Node 2
        }
        r2 = requests.post(f"{NODE2_URL}/register_peer", json=payload_n2)
        r2.raise_for_status()
        print(f"Node 2 -> Node 1 Registration: {r2.json().get('message')}")
        
        print("Peers registered and vectors exchanged successfully.\n")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error setting/registering peers: {e}")
        print("!!! Please ensure both servers are running.")
        exit(1)

# --- MODIFIED FUNCTION ---
def insert_vectors():
    """
    Insert completely random vectors, alternating between nodes.
    The nodes themselves will route the vectors based on similarity.
    """
    print(f"--- 2. Inserting {NUM_VECTORS} RANDOM Vectors (Size {VECTOR_SIZE}) ---")
    
    insertions = []
    
    for i in range(NUM_VECTORS):
        # Determine which node to send to (round-robin for load balancing)
        node_url = NODE_URLS[i % len(NODE_URLS)]
        
        # --- Generate a COMPLETELY RANDOM vector ---
        final_vec = np.random.rand(VECTOR_SIZE).tolist()
        
        vector_data = {
            "id": str(uuid.uuid4()),
            "vector": final_vec,
            "payload": {
                "source_type": "random",
                "sent_to_node": "node1" if node_url == NODE1_URL else "node2",
                "index": i
            }
        }
        
        try:
            # Use the /add_vector endpoint (which now handles routing)
            response = requests.post(
                f"{node_url}/add_vector",
                json=vector_data
            )
            response.raise_for_status()
            
            # Log the routing action
            res_data = response.json()
            stored_at = res_data.get('node_id')
            insertions.append(stored_at)
            
            if (i + 1) % 100 == 0:
                print(f"Inserted {i + 1}/{NUM_VECTORS}... (Last: sent to {vector_data['payload']['sent_to_node']}, stored at {stored_at})")
                
        except requests.exceptions.RequestException as e:
            print(f"!!! Error inserting vector {i}: {e}")
            
    print(f"Vector insertion complete.\n")
    return insertions

# --- MODIFIED FUNCTION ---
def check_counts(insertions: list):
    """Check the final vector counts on each node."""
    print("--- 3. Checking Vector Counts (Random Distribution) ---")
    try:
        r1 = requests.get(f"{NODE1_URL}/count")
        r2 = requests.get(f"{NODE2_URL}/count")
        
        count1 = r1.json().get('count', 0)
        count2 = r2.json().get('count', 0)
        
        print(f"Node 1 Count: {count1}")
        print(f"Node 2 Count: {count2}")
        print(f"Total Vectors: {count1 + count2}")
        
        # Check from test script perspective
        script_count1 = insertions.count("node1")
        script_count2 = insertions.count("node2")
        print(f"(Client-side log check: node1={script_count1}, node2={script_count2})")
        
        if count1 + count2 == NUM_VECTORS:
            print("✅  Total counts match total inserted vectors.")
        else:
            print("⚠️  Counts do NOT match total inserted vectors!")

        # Check if routing was *roughly* balanced
        expected_half = NUM_VECTORS / 2
        if abs(count1 - expected_half) < (NUM_VECTORS * 0.1): # 10% tolerance
             print(f"✅  Distribution is roughly balanced (within 10%).")
        else:
             print(f"⚠️  Distribution seems unbalanced! (Count1: {count1}, Count2: {count2})")
        print("")
        
    except requests.exceptions.RequestException as e:
        print(f"!!! Error checking counts: {e}")

# --- MODIFIED FUNCTION ---
def run_queries():
    """Run a single federated search query with a random vector."""
    print("--- 4. Running Random Federated Search Query ---")
    
    # 1. Generate a random query vector
    query_vector = np.random.rand(VECTOR_SIZE).tolist()
    print(f"Query Vector: {[round(x, 2) for x in query_vector]}\n")

    try:
        # Run Federated Search (always from Node 1, doesn't matter)
        r_fed = requests.post(
            f"{NODE1_URL}/search/federated",
            json=query_vector,
            params={"top_k": 5} # Request top 5 from EACH node
        )
        r_fed.raise_for_status()
        fed_results = r_fed.json()
        
        print(f"Federated search complete:")
        results_data = fed_results.get('results', {})
        
        node1_results_list = results_data.get('node1', [])
        node2_results_list = results_data.get('node2', [])
        
        node1_results_count = len(node1_results_list)
        node2_results_count = len(node2_results_list)
        
        max_score_node1 = node1_results_list[0].get('score', 0) if node1_results_count > 0 else 0
        max_score_node2 = node2_results_list[0].get('score', 0) if node2_results_count > 0 else 0

        min_score_node1 = node1_results_list[4].get('score', 0) if node1_results_count > 0 else 0
        min_score_node2 = node2_results_list[4].get('score', 0) if node2_results_count > 0 else 0
        
        print(f"  - Total Results Found: {fed_results.get('total_results')}")
        print(f"  - Results from Node 1: {node1_results_count} (Best Score: {max_score_node1:.4f} - Worst Score: {min_score_node1:.4f})")
        print(f"  - Results from Node 2: {node2_results_count} (Best Score: {max_score_node2:.4f} - Worst Score: {min_score_node2:.4f})")
        
        if max_score_node1 > max_score_node2:
             print("ℹ️  Node 1 had the best matching vector for this random query.")
        elif max_score_node2 > max_score_node1:
             print("ℹ️  Node 2 had the best matching vector for this random query.")
        else:
             print("ℹ️  Scores are equal or no results found.")
        
        print("\nFull Federated Results:")
        print(json.dumps(results_data, indent=2))
        print("")

    except requests.exceptions.RequestException as e:
        print(f"!!! Error running random query: {e}")
# --- END MODIFIED FUNCTIONS ---


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Qdrant Smart Sharding Test (Random Distribution)...")
    print("Please ensure you have 2 Qdrant instances running (e.g., on ports 6333 and 6334)")
    print("And 2 FastAPI servers running (e.g., on ports 8001 and 8002)\n")
    
    # Give servers a moment to start
    time.sleep(2)
    
    register_peers()
    insertions = insert_vectors()
    check_counts(insertions)
    run_queries()
    
    print("--- Test Complete ---")