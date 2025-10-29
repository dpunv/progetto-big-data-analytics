# config.py
import os

# Qdrant Cluster Configuration
QDRANT_NODES = {
    "node-1": "http://localhost:6333",
    "node-2": "http://localhost:7333",
    "node-3": "http://localhost:8333",
}

COLLECTION_NAME = "semantic_collection"
VECTOR_DIMENSION = 128  # Dimensione degli embedding di esempio
N_CLUSTERS = 16       # Numero di cluster iniziali (coarse quantizer)

# Quantizer Configuration
QUANTIZER_PATH = "quantizer.pkl"
SAMPLE_DATA_SIZE_FOR_TRAINING = 20000

# Rebalancer Configuration
REBALANCER_THRESHOLD = 7000  # Numero massimo di punti per cluster prima dello split
SPLIT_FACTOR = 2             # In quanti sub-cluster dividere un hotspot

# Simulation Configuration
TOTAL_VECTORS_TO_INSERT = 20000
HOTSPOT_CLUSTER_ID = 5         # ID del cluster che renderemo "hot"
HOTSPOT_BIAS_FACTOR = 0.6      # 60% dei nuovi inserimenti andrà in questo cluster