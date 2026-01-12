import json
import logging
import os
import sys
from typing import List

import numpy as np

# Get logger for this module
logger = logging.getLogger(__name__)

# Centralized logging toggle
LOGGING_ENABLED = True


def load_vectors(filename, max_vectors):
    if not os.path.exists(filename):
        logger.error(f"Error: File not found at '{filename}'")
        sys.exit(1)
    try:
        with open(filename, "r") as f:
            vectors = [item["embedding"] for item in json.load(f)]
        if max_vectors > 0:
            vectors = vectors[:max_vectors]

        X = np.array(vectors)

        if len(X.shape) != 2 or X.shape[1] == 0:
            logger.error(f"Error: Data in '{filename}' is not a valid 2D array.")
            sys.exit(1)
        return X
    except Exception as e:
        logger.error(f"Error loading or processing '{filename}': {e}")
        sys.exit(1)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    a = np.asarray(v1)
    b = np.asarray(v2)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def get_collection_name(name, cluster):
    return f"{name}_{cluster}"
