import numpy as np
from typing import List

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1_arr = np.array(v1, dtype=np.float32)
    v2_arr = np.array(v2, dtype=np.float32)
    
    dot_product = np.dot(v1_arr, v2_arr)
    norm_v1 = np.linalg.norm(v1_arr)
    norm_v2 = np.linalg.norm(v2_arr)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return float(dot_product / (norm_v1 * norm_v2))
