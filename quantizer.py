# quantizer.py
import faiss
import numpy as np
import joblib
import os

def build_quantizer(embeddings_sample: np.ndarray, n_clusters: int, save_path: str):
    """
    Costruisce un quantizzatore K-means usando Faiss e lo salva su disco.
    
    SCOPO:
    - Crea un modello K-means che divide lo spazio vettoriale in N cluster semantici
    - Questo permette di raggruppare vettori simili insieme
    
    COME FUNZIONA:
    - Faiss K-means trova N centroidi ottimali analizzando un campione di embeddings
    - I centroidi rappresentano i "punti centrali" di ogni cluster semantico
    - Questi centroidi vengono salvati su disco per riutilizzo futuro
    
    PERCHÉ È IMPORTANTE:
    - Permette di fare "sharding semantico": vettori simili vanno nello stesso cluster
    - È più efficiente del random sharding perché query simili cercano nello stesso nodo
    
    Args:
        embeddings_sample: Campione di vettori per training del K-means
        n_clusters: Numero di cluster da creare (es. 10, 20, 100)
        save_path: Percorso dove salvare i centroidi
        
    Returns:
        centroids: Array numpy con i centroidi di ogni cluster
    """
    print(f"Building quantizer with {n_clusters} clusters using Faiss K-means...")
    d = embeddings_sample.shape[1]  # Dimensione dei vettori (es. 128, 384, 768)
    
    # Crea e addestra il modello K-means
    kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, verbose=True)
    kmeans.train(embeddings_sample)
    
    # Salviamo solo i centroidi, che sono ciò che ci serve per la predizione.
    centroids = kmeans.centroids
    joblib.dump(centroids, save_path)
    print(f"Quantizer saved to {save_path}")
    return centroids

def load_quantizer(path: str) -> np.ndarray:
    """
    Carica i centroidi del quantizzatore da un file.
    
    SCOPO:
    - Recupera il modello K-means precedentemente addestrato
    
    PERCHÉ È IMPORTANTE:
    - Evita di ri-addestrare il modello ogni volta (costoso computazionalmente)
    - Garantisce consistenza: usiamo sempre gli stessi cluster
    
    Args:
        path: Percorso del file contenente i centroidi
        
    Returns:
        centroids: Array numpy con i centroidi caricati
        
    Raises:
        FileNotFoundError: Se il file non esiste (devi prima fare build)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Quantizer file not found at {path}. Please build it first.")
    print(f"Loading quantizer from {path}...")
    return joblib.load(path)

def predict_cluster(centroids: np.ndarray, vector: np.ndarray) -> int:
    """
    Assegna un vettore al cluster più vicino calcolando la distanza dai centroidi.
    
    SCOPO:
    - Determina a quale cluster semantico appartiene un nuovo vettore
    
    COME FUNZIONA:
    - Calcola la distanza euclidea tra il vettore e tutti i centroidi
    - Sceglie il cluster con centroide più vicino (distanza minima)
    
    PERCHÉ È IMPORTANTE:
    - Questa è la funzione chiave per il routing semantico
    - Decide su quale nodo Qdrant andrà memorizzato/cercato il vettore
    - Vettori simili avranno distanze simili dai centroidi → stesso cluster → stesso nodo
    
    ESEMPIO:
    - Se hai embedding di "cane" e cerchi dove metterlo
    - Calcola distanza da centroide "animali", "tecnologia", "cibo", etc.
    - "animali" è più vicino → cluster 3 → nodo-2
    
    Args:
        centroids: Array dei centroidi K-means [n_clusters, dimensione]
        vector: Vettore da classificare [dimensione]
        
    Returns:
        cluster_id: ID del cluster più vicino (intero da 0 a n_clusters-1)
    """
    vector_reshaped = vector.reshape(1, -1)  # Da [dim] a [1, dim] per broadcasting
    
    # Calcola distanza euclidea da ogni centroide
    distances = np.linalg.norm(centroids - vector_reshaped, axis=1)
    
    # Restituisce l'indice del centroide con distanza minima
    return int(np.argmin(distances))