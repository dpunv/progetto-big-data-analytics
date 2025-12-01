import requests
import sys
import time

def fetch_metrics(num_nodes=4, start_port=8001):
    print(f"Fetching metrics from {num_nodes} nodes starting at port {start_port}...")
    
    for i in range(num_nodes):
        port = start_port + i
        url = f"http://localhost:{port}/metrics"
        node_name = f"node{i+1}"
        
        print(f"\n--- Metrics for {node_name} ({url}) ---")
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(response.text)
            else:
                print(f"Error: Status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to {node_name}: {e}")

if __name__ == "__main__":
    num_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    fetch_metrics(num_nodes)
