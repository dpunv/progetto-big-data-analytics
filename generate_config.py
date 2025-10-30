import sys
import yaml
from pathlib import Path

def generate_config(num_nodes: int):
    """Genera file config.yaml con numero di nodi parametrico."""
    
    if num_nodes <= 0:
        print("Error: Number of nodes must be positive")
        sys.exit(1)
    
    # Crea lista di nodi
    nodes = [f"Node_{i+1}" for i in range(num_nodes)]
    
    # Template configurazione
    config = {
        'system': {
            'num_nodes': num_nodes,
            'num_clusters': 12,  # Puoi anche renderlo parametrico
            'data_partition_method': 'semantic_clustering'
        },
        'nodes': {node: {'port': 5000 + i, 'host': 'localhost'} for i, node in enumerate(nodes)},
        'logging': {
            'level': 'INFO',
            'file': 'routing.log'
        }
    }
    
    # Salva YAML
    config_path = Path('config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Configuration created: {config_path}")
    print(f"   Nodes: {num_nodes}")
    print(f"   Node list: {', '.join(nodes)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generate_config.py <num_nodes>")
        sys.exit(1)
    
    try:
        num_nodes = int(sys.argv[1])
        generate_config(num_nodes)
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)
