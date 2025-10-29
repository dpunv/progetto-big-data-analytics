# quantizer.py
import faiss
import numpy as np
import joblib
import os

def build_quantizer(embeddings_sample: np.ndarray, n_clusters: int, save_path: str):
    """
    Costruisce un quantizzatore K-means usando Faiss e lo salva su disco.
    
    ═══════════════════════════════════════════════════════════════════════
    SPIEGAZIONE DETTAGLIATA: COS'È LA CLUSTERIZZAZIONE K-MEANS
    ═══════════════════════════════════════════════════════════════════════
    
    1. OBIETTIVO:
    - Dividere N vettori in K gruppi (cluster) omogenei
    - Ogni cluster contiene vettori "simili tra loro"
    - Ogni cluster è rappresentato da un centroide (punto centrale)
    
    2. ALGORITMO (Lloyd's Algorithm):
    
    Passo 0 - Inizializzazione:
    ```
    Scegli K centroidi iniziali casualmente tra i vettori
    Es. con 10K vettori e K=10 cluster:
    - Centroide 0: vettore random #234
    - Centroide 1: vettore random #1567
    - ...
    - Centroide 9: vettore random #8921
    ```
    
    Passo 1 - Assignment (assegnazione):
    ```
    Per ogni vettore:
        Calcola distanza da ogni centroide
        Assegna al cluster con centroide più vicino
    
    Esempio:
    Vettore X = [0.5, 0.3, -0.2, ...]
    
    Distanze:
    - da centroide 0: 1.2
    - da centroide 1: 0.4  ← minima!
    - da centroide 2: 2.1
    - ...
    
    → X assegnato a cluster 1
    ```
    
    Passo 2 - Update (ricalcolo centroidi):
    ```
    Per ogni cluster:
        Calcola media di tutti i vettori nel cluster
        Questa media diventa il nuovo centroide
    
    Esempio cluster 1:
    Vettori: [v1, v2, v3, ..., v_n]
    Nuovo centroide = (v1 + v2 + v3 + ... + v_n) / n
    ```
    
    Passo 3 - Iterazione:
    ```
    Ripeti Passo 1 e 2 finché:
    - Centroidi non cambiano più (convergenza)
    - Oppure raggiunto numero massimo iterazioni (es. 20)
    ```
    
    3. ESEMPIO VISIVO (2D per semplicità):
    
    Iterazione 0 (iniziale):
    ```
    Vettori:  •  •    •
              • •   •  •
                •  •   •
    
    Centroidi iniziali: ⊕  ⊕  ⊕  (random)
    ```
    
    Iterazione 1 (dopo assignment):
    ```
    Cluster 0:  •  •       (rosso)
                • •
    
    Cluster 1:      •  •   (blu)
                   •  •
    
    Cluster 2:        •    (verde)
    
    Nuovi centroidi: ⊕ (centro massa rossi)
                       ⊕ (centro massa blu)
                         ⊕ (centro massa verde)
    ```
    
    Iterazione finale (convergenza):
    ```
    Cluster 0:  • •        Centroide: ⊕ (stabile)
                • •
    
    Cluster 1:      • •    Centroide:   ⊕ (stabile)
                    • •
    
    Cluster 2:        •    Centroide:     ⊕ (stabile)
    ```
    
    4. MATEMATICA (con vettori reali 128-dim):
    
    Distanza euclidea:
    ```
    d(v, c) = √[(v₁-c₁)² + (v₂-c₂)² + ... + (v₁₂₈-c₁₂₈)²]
    ```
    
    Centroide (media):
    ```
    c = (v₁ + v₂ + ... + vₙ) / n
    
    Per ogni dimensione i:
    cᵢ = (v₁ᵢ + v₂ᵢ + ... + vₙᵢ) / n
    ```
    
    5. PERCHÉ FAISS:
    - Ottimizzato per vettori ad alta dimensione (128, 768, etc.)
    - Usa GPU se disponibile (100x più veloce)
    - Implementazione efficiente con SIMD, cache-friendly
    
    6. PARAMETRI:
    - niter=20: massimo 20 iterazioni (di solito converge prima)
    - verbose=True: stampa progresso iterazioni
    
    7. OUTPUT:
    - kmeans.centroids: array [K, dim] con i K centroidi finali
    - Es. [10, 128] = 10 centroidi di 128 dimensioni ciascuno
    
    8. INTERPRETAZIONE SEMANTICA:
    
    Se i vettori sono embeddings di documenti:
    ```
    Centroide 0: [0.8, 0.1, -0.3, ...] → area "tecnologia"
    Centroide 1: [0.2, 0.9, 0.1, ...]  → area "sport"
    Centroide 2: [-0.1, 0.3, 0.7, ...] → area "cucina"
    ...
    ```
    
    Ogni centroide rappresenta il "tema medio" del suo cluster.
    
    9. USO NEL SISTEMA:
    - Training: fatto UNA VOLTA su sample rappresentativo
    - Inference: usa predict_cluster() per assegnare nuovi vettori
    - Routing: cluster_id determina quale nodo Qdrant
    
    ═══════════════════════════════════════════════════════════════════════
    
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