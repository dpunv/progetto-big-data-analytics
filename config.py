# config.py
"""
File di configurazione centralizzato per il sistema di sharding semantico distribuito.

SCOPO:
- Mantiene tutte le configurazioni in un unico posto
- Facilita il tuning dei parametri senza modificare il codice
- Permette di cambiare facilmente dimensioni, soglie, nodi

ORGANIZZAZIONE:
1. Configurazione Qdrant (nodi e collections)
2. Parametri del Quantizer (K-means)
3. Parametri di Ingestion (dati e hotspot)
4. Parametri di Rebalancing (soglie e split)
"""

# ============================================================================
# 1. CONFIGURAZIONE QDRANT - Definisce l'infrastruttura dei nodi
# ============================================================================

QDRANT_NODES = {
    "node-1": "http://localhost:6333",  # Primo nodo Qdrant
    "node-2": "http://localhost:7333",  # Secondo nodo Qdrant
    "node-3": "http://localhost:8333",  # Terzo nodo Qdrant
}
"""
Dizionario dei nodi Qdrant disponibili.

STRUTTURA: {nome_nodo: url_endpoint}

PERCHÉ QUESTA STRUTTURA:
- Facile aggiungere/rimuovere nodi
- Nomi leggibili invece di URL ovunque nel codice
- Permette di cambiare porte senza modificare logica

ESEMPIO DI ESPANSIONE:
Per aggiungere un quarto nodo:
"node-4": "http://localhost:9333"
"""

COLLECTION_NAME = "semantic_vectors"
"""
Nome della collection Qdrant su tutti i nodi.

COSA È UNA COLLECTION:
- È come una "tabella" in un database
- Contiene tutti i vettori con i loro metadati
- Ogni nodo ha la stessa collection (stessa struttura, dati diversi)

PERCHÉ LO STESSO NOME:
- Uniformità: stesso schema su tutti i nodi
- Semplifica il codice: non dobbiamo tracciare nomi diversi
"""

# ============================================================================
# 2. PARAMETRI DEL QUANTIZER - Configurazione K-means per clustering
# ============================================================================

VECTOR_DIMENSION = 128
"""
Dimensionalità dei vettori embeddings.

COSA SIGNIFICA:
- Ogni vettore ha 128 numeri float
- Rappresenta features semantiche (es. significato di un testo/immagine)

ESEMPI COMUNI:
- 128: Embeddings leggeri (face recognition)
- 384: Sentence-BERT mini
- 768: BERT base
- 1536: OpenAI ada-002

IMPATTO:
- Dimensioni maggiori → più accuratezza ma più memoria/tempo
- Dimensioni minori → più veloce ma meno espressivo
"""

N_CLUSTERS = 10
"""
Numero di cluster K-means per il quantizer.

COSA RAPPRESENTA:
- Divide lo spazio vettoriale in 10 regioni semantiche
- Ogni cluster raggruppa vettori simili

COME SCEGLIERE:
- Troppo pochi (es. 3): cluster troppo grandi, rebalancing inefficace
- Troppi (es. 1000): overhead di gestione, split inutili
- Regola pratica: sqrt(numero_totale_vettori) / 100

ESEMPIO:
- 10 cluster con 3 nodi → ~3-4 cluster per nodo
- Dopo split: 1 cluster → 3 sub-cluster
"""

QUANTIZER_PATH = "quantizer_centroids.pkl"
"""
Percorso dove salvare/caricare i centroidi K-means.

PERCHÉ SALVARLO:
- Training K-means è costoso (minuti su milioni di vettori)
- Riutilizzo: carichiamo invece di ri-addestrare
- Consistenza: stessi cluster tra esecuzioni

FORMATO:
- File pickle con numpy array [N_CLUSTERS, VECTOR_DIMENSION]
- Es. [10, 128] = 10 centroidi di 128 dimensioni ciascuno
"""

SAMPLE_DATA_SIZE_FOR_TRAINING = 10000
"""
Numero di vettori campione per addestrare il K-means.

COSA FA:
- Genera 10K vettori random per training
- K-means trova i 10 centroidi ottimali analizzando questo campione

PERCHÉ NON TUTTI I DATI:
- K-means è O(n*k*i) dove n=vettori, k=cluster, i=iterazioni
- 10K vettori → training veloce (~secondi)
- 1M vettori → training lento (~minuti)
- Campione rappresentativo è sufficiente

IN PRODUZIONE:
- Useresti veri embeddings (es. da documento corpus)
- Sample stratificato per rappresentare bene i dati
"""

# ============================================================================
# 3. PARAMETRI DI INGESTION - Configurazione inserimento dati e hotspot
# ============================================================================

TOTAL_VECTORS_TO_INSERT = 50000
"""
Numero totale di vettori da inserire nel sistema.

SCOPO:
- Simula un carico di dati realistico
- Abbastanza grande da creare hotspot significativo
- Abbastanza piccolo da eseguire velocemente in demo

DISTRIBUZIONE:
- Con HOTSPOT_BIAS_FACTOR=0.7 e HOTSPOT_CLUSTER_ID=5:
  - 35K vettori → cluster 5 (hotspot)
  - 15K vettori → altri 9 cluster (~1.6K ciascuno)

SCALA REALE:
- Produzione: milioni o miliardi di vettori
- Demo/test: 10K-100K vettori
"""

HOTSPOT_CLUSTER_ID = 5
"""
ID del cluster dove creare artificialmente l'hotspot.

PERCHÉ CLUSTER 5:
- Arbitrario, ma mid-range (non estremi 0 o 9)
- Round-robin iniziale: cluster 5 va su node-2 (5 % 3 = 2)

COSA SIGNIFICA:
- La maggior parte dei vettori verrà forzata in questo cluster
- Simula scenario reale: es. trending topic ("COVID-19", "ChatGPT")
"""

HOTSPOT_BIAS_FACTOR = 0.7
"""
Probabilità che un vettore vada al cluster hotspot.

COME FUNZIONA:
- 70% dei vettori → forzati al cluster HOTSPOT_CLUSTER_ID
- 30% dei vettori → distribuzione normale K-means

ESEMPIO:
- Su 50K vettori totali:
  - 35K → cluster 5 (hotspot)
  - 15K → distribuiti sui 9 cluster rimanenti

PERCHÉ 0.7:
- Crea sbilanciamento significativo ma non estremo
- Abbastanza per triggerare rebalancing
- Realista: in trending topic, ~60-80% traffico va su pochi cluster
"""

# ============================================================================
# 4. PARAMETRI DI REBALANCING - Soglie e configurazione split
# ============================================================================

REBALANCER_THRESHOLD = 5000
"""
Soglia di vettori per considerare un cluster come "hotspot".

LOGICA:
- Se cluster ha > 5000 vettori → è un hotspot → split
- Se cluster ha ≤ 5000 vettori → OK, non serve azione

CALCOLO:
- Con 50K vettori e 10 cluster:
  - Distribuzione ideale: 5K per cluster
  - Cluster con 35K → 7x la media → definito hotspot

COME SCEGLIERE:
- Troppo alto (es. 100K): split solo in casi estremi
- Troppo basso (es. 500): split troppo frequenti, overhead
- Regola: 2-3x la dimensione media cluster
"""

N_SUB_CLUSTERS = 3
"""
Numero di sub-cluster in cui splittare un cluster hotspot.

COSA FA:
- Quando cluster 5 è hotspot (35K vettori):
  - Split in 3 parti: 5.0, 5.1, 5.2
  - Ogni sub-cluster ha ~11.6K vettori

PERCHÉ 3:
- Bilanciato: non troppo pochi (2), non troppi (10)
- Con 3 nodi, possiamo distribuire 1 sub-cluster per nodo
- Riduce carico del 66% sul nodo originale

ALTERNATIVE:
- 2 sub-cluster: split binario semplice
- 4-5 sub-cluster: granularità maggiore ma più overhead
"""

# ============================================================================
# 5. PARAMETRI LOAD-AWARE ASSIGNMENT - Bilanciamento semantica vs carico
# ============================================================================

ALPHA_SEMANTIC = 0.7
"""
Peso della similarità semantica nell'assegnazione sub-cluster.

SCOPO:
- Controlla quanto è importante la località semantica
- Valore alto (0.7-0.9): privilegia semantica, tollera sbilanciamento
- Valore basso (0.3-0.5): privilegia bilanciamento carico

RANGE: [0.0, 1.0]

ESEMPIO:
- α=0.9: Semantica domina, sub-cluster vanno quasi sempre sul nodo LSH-preferito
- α=0.5: Bilanciato, semantica e carico hanno peso uguale
- α=0.3: Carico domina, sub-cluster vanno sui nodi più vuoti

RACCOMANDAZIONE:
- Per query semantiche intensive: 0.7-0.8
- Per carico uniforme: 0.4-0.6
"""

BETA_LOAD = 0.3
"""
Peso della capacità del nodo nell'assegnazione sub-cluster.

RELAZIONE: ALPHA_SEMANTIC + BETA_LOAD = 1.0

SCOPO:
- Controlla quanto è importante evitare nodi sovraccarichi
- Valore alto (0.5-0.7): forza redistribuzione uniforme
- Valore basso (0.1-0.3): accetta sbilanciamento per località

INTERPRETAZIONE:
- β=0.3: "Considera il carico, ma non sacrificare troppo la semantica"
- β=0.5: "Carico e semantica ugualmente importanti"
- β=0.7: "Evita hotspot a tutti i costi, anche perdendo località"

NOTA:
- Se modifichi ALPHA_SEMANTIC, aggiorna BETA_LOAD = 1.0 - ALPHA_SEMANTIC
"""

# Verifica che somma sia 1.0
assert abs(ALPHA_SEMANTIC + BETA_LOAD - 1.0) < 0.001, \
    f"ALPHA_SEMANTIC ({ALPHA_SEMANTIC}) + BETA_LOAD ({BETA_LOAD}) must equal 1.0"

# ============================================================================
# 6. PARAMETRI NODE-LEVEL BALANCING - Soglie per bilanciamento nodi
# ============================================================================

NODE_BALANCE_TOLERANCE = 0.3
"""
Tolleranza percentuale per considerare il sistema bilanciato.

SCOPO:
- Definisce quanto sbilanciamento è accettabile prima di triggerare rebalancing
- Evita rebalancing continuo per piccole fluttuazioni

INTERPRETAZIONE:
- 0.3 = ±30% dalla media è accettabile
- Se nodo ha carico tra 70% e 130% della media → OK
- Se nodo ha carico < 70% o > 130% → rebalancing

ESEMPI:
Tolerance 0.3 (30%):
- Media: 16,666 vettori
- Range OK: 11,666 - 21,666
- node-1: 20,000 → OK (20% over)
- node-2: 25,000 → OVERLOADED (50% over)

Tolerance 0.5 (50%):
- Range OK: 8,333 - 25,000
- node-2: 25,000 → OK (più permissivo)

Tolerance 0.1 (10%):
- Range OK: 15,000 - 18,333
- node-1: 20,000 → OVERLOADED (più rigoroso)

RACCOMANDAZIONE:
- Produzione: 0.2-0.3 (bilanciamento moderato)
- Testing: 0.3-0.5 (permissivo per vedere effetti)
- Strict: 0.1-0.15 (bilanciamento rigoroso)
"""

# ============================================================================
# RIEPILOGO FLUSSO CON QUESTI PARAMETRI
# ============================================================================
"""
SETUP INIZIALE:
1. 3 nodi Qdrant (node-1, node-2, node-3)
2. 10 cluster semantici (0-9)
3. Round-robin: cluster 5 → node-2

INGESTION:
1. Inserimento 50K vettori
2. 70% (35K) → cluster 5 (node-2 sovraccarico)
3. 30% (15K) → altri 9 cluster (~1.6K ciascuno)

MONITORING:
1. Conta vettori per cluster
2. Cluster 5: 35K vettori > 5K threshold → HOTSPOT

REBALANCING:
1. Split cluster 5 in 3 sub-cluster:
   - 5.0: ~11.6K vettori → resta su node-2
   - 5.1: ~11.6K vettori → migrato su node-1
   - 5.2: ~11.6K vettori → migrato su node-3
2. Update routing table
3. Carico distribuito equamente

RISULTATO:
- Prima: node-2 aveva 35K vettori (sovraccarico)
- Dopo: ogni nodo ha ~11-12K vettori (bilanciato)
- Latency ridotta, throughput aumentato
"""