# config.py
"""Configurazione sistema di semantic clustering."""

# ============================================================================
# NODI QDRANT
# ============================================================================

QDRANT_NODES = {
    "node-1": "http://localhost:6333",
    "node-2": "http://localhost:7333",
    "node-3": "http://localhost:8333",
}

COLLECTION_NAME = "semantic_vectors"

# ============================================================================
# QUANTIZER (K-MEANS)
# ============================================================================

VECTOR_DIMENSION = 384  # sentence-transformers dimension

N_CLUSTERS = 10  # Numero cluster K-means

QUANTIZER_PATH = "quantizer_centroids.pkl"

SAMPLE_DATA_SIZE_FOR_TRAINING = 10000  # Samples per training K-means

# ============================================================================
# DATA INGESTION
# ============================================================================

TOTAL_VECTORS_TO_INSERT = 50000  # Vettori da inserire

# ============================================================================
# WIKIPEDIA DATASET
# ============================================================================

USE_REAL_DATA = True  # True = Wikipedia, False = random

"""
SE USE_REAL_DATA = True:
- Scarica ~10K frasi da Wikipedia (prima volta: ~5-10 min)
- Genera embeddings con sentence-transformers
- Cluster semanticamente distinti (tech, sport, science, etc.)
- Cache salvata per riutilizzo

SE USE_REAL_DATA = False:
- Genera vettori random uniformi
- Cluster molto simili (distanze 0.83-1.01)
- Utile solo per test veloci
"""