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
# WIKIPEDIA DATASET
# ============================================================================

TOTAL_VECTORS_TO_INSERT = 10000  # MODIFICATO: Numero frasi Wikipedia da scaricare

WIKIPEDIA_LANGUAGE = 'en'  # Lingua: 'en', 'it', 'es', etc.

WIKIPEDIA_CACHE_PATH = 'wikipedia_embeddings_cache.pkl'

"""
WIKIPEDIA EMBEDDINGS:
- Scarica frasi REALI da Wikipedia
- Genera embeddings con sentence-transformers (384-dim)
- Inserisce in Qdrant con metadata (testo originale)
- Prima volta: ~5-10 min download
- Successivamente: carica da cache
"""