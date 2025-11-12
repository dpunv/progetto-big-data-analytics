import numpy as np
import os
import sys
import json

def load_vectors(filename, max_vectors):
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        sys.exit(1)
    try:
        with open(filename, 'r') as f:
            vectors = [item['embedding'] for item in json.load(f)]
        if max_vectors > 0:
            vectors = vectors[:max_vectors]
        
        X = np.array(vectors)
        
        if len(X.shape) != 2 or X.shape[1] == 0:
            print(f"Error: Data in '{filename}' is not a valid 2D array.")
            sys.exit(1)
        return X
    except Exception as e:
        print(f"Error loading or processing '{filename}': {e}")
        sys.exit(1)

