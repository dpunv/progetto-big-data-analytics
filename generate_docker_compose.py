import sys
import yaml

def generate_docker_compose(num_nodes: int):
    """
    Genera dinamicamente un file docker-compose.yml con N nodi Qdrant.
    
    Args:
        num_nodes: Numero di nodi da creare
    """
    
    base_port = 6333
    services = {}
    
    # Genera configurazione per ogni nodo
    for i in range(1, num_nodes + 1):
        port = base_port + (i - 1)
        service_name = f"qdrant_node_{i}"
        
        services[service_name] = {
            "image": "qdrant/qdrant:latest",
            "container_name": service_name,
            "ports": [f"{port}:6333"],
            "volumes": [
                f"./qdrant_storage/node_{i}:/qdrant/storage",
                f"./qdrant_snapshots/node_{i}:/qdrant/snapshots"
            ],
            "environment": [
                f"QDRANT_API_KEY=",
                "QDRANT_TELEMETRY_DISABLED=true"
            ],
            "networks": ["qdrant_network"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://localhost:6333/health"],
                "interval": "10s",
                "timeout": "5s",
                "retries": 5
            }
        }
    
    # Struttura completa docker-compose
    compose_data = {
        "version": "3.8",
        "services": services,
        "networks": {
            "qdrant_network": {
                "driver": "bridge"
            }
        },
        "volumes": {}
    }
    
    # Salva il file
    with open("docker-compose.yml", "w") as f:
        yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Generated docker-compose.yml with {num_nodes} Qdrant nodes")
    print(f"  Ports: {base_port} - {base_port + num_nodes - 1}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_docker_compose.py <number_of_nodes>")
        sys.exit(1)
    
    try:
        num_nodes = int(sys.argv[1])
        if num_nodes < 1:
            raise ValueError("Number of nodes must be at least 1")
        generate_docker_compose(num_nodes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
