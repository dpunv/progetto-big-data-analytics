"""
Vector operations and similarity metrics.
"""
import numpy as np
from typing import List, Union


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)
    Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
        
    Example:
        >>> v1 = [1, 0, 0]
        >>> v2 = [1, 0, 0]
        >>> cosine_similarity(v1, v2)
        1.0
    """
    # Convert to numpy arrays
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    
    # Calculate norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)


def euclidean_distance(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    
    return float(np.linalg.norm(a - b))


def normalize_vector(vec: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector
    """
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    
    if norm == 0:
        return arr
    
    return arr / norm
