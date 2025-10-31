# config.py
"""Configurazione sistema di semantic clustering."""

# ============================================================================
# NODI QDRANT (DINAMICI)
# ============================================================================

def generate_qdrant_nodes(num_nodes: int = 10) -> dict[str, str]:
    """
    Genera dinamicamente la configurazione dei nodi Qdrant.
    
    MODIFICATO: Salta porta 7333 (node-2 problematico) e usa porte alternative.
    
    Args:
        num_nodes: Numero di nodi da generare (default: 10)
        
    Returns:
        Dictionary con mapping node_name → URL
        
    Example:
        generate_qdrant_nodes(3) → {
            "node-1": "http://localhost:6333",
            "node-2": "http://localhost:7333",
            "node-3": "http://localhost:8333"
        }
    """
    nodes = {}
    base_port = 6333
    
    for i in range(1, num_nodes + 1):
        # Porte normali, MA node-2 usa 7444 invece di 7333
        if i == 1:
            port = 6333  # node-1
        elif i == 2:
            port = 7444  # PORTA CUSTOM per node-2
        else:
            port = base_port + (i * 1000)  # 8333, 9333, 10333, ...
        
        node_name = f"node-{i}"
        node_url = f"http://localhost:{port}"
        nodes[node_name] = node_url
    
    return nodes

# Default: 10 nodi (può essere sovrascritto)
QDRANT_NODES = generate_qdrant_nodes(10)

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

TOTAL_VECTORS_TO_INSERT = 10000  # Numero di vettori da caricare

WIKIPEDIA_LANGUAGE = 'en'  # Lingua: 'en', 'it', 'es', etc.

WIKIPEDIA_CACHE_PATH = 'wikipedia_embeddings_cache.pkl'

# ============================================================================
# META-HNSW ROUTING
# ============================================================================

META_HNSW_PATH = "meta_hnsw_index.pkl"

# Parametri HNSW
META_HNSW_EF_CONSTRUCTION = 200
META_HNSW_M = 16
META_HNSW_EF_SEARCH = 50

# Routing: numero di nodi candidati per query/insert
TOP_K_NODES_FOR_ROUTING = 3

# Metodo calcolo centroidi: 'mean', 'median', 'weighted'
CENTROID_CALCULATION_METHOD = 'mean'

# ════════════════════════════════════════════════════════════════════════
# CENTROID UPDATE POLICY
# ════════════════════════════════════════════════════════════════════════

# Strategia update: 'incremental', 'batch', 'periodic', 'none'
CENTROID_UPDATE_STRATEGY = 'batch'

# Batch update: ogni quanti inserimenti aggiornare centroidi
CENTROID_UPDATE_BATCH_SIZE = 1000

# Periodic update: intervallo in secondi (solo se strategy='periodic')
CENTROID_UPDATE_INTERVAL_SECONDS = 60

# Full recalculation: ogni quanti update fare ricalcolo completo (corregge drift)
CENTROID_FULL_RECALC_EVERY = 100000  # Ogni 100K inserimenti

# ════════════════════════════════════════════════════════════════════════
# HNSW REBUILD POLICY
# ════════════════════════════════════════════════════════════════════════

# Rebuild quando drift centroidi > threshold (cosine distance)
HNSW_REBUILD_DRIFT_THRESHOLD = 0.1

# Rebuild ogni N batch update
HNSW_REBUILD_EVERY_N_UPDATES = 10

"""
HNSW REBUILD POLICY:

PROBLEMA:
- hnswlib non supporta update in-place sicuro
- add_items() con stesso ID può creare duplicati
- Soluzione: rebuild periodico dell'indice

STRATEGIE:
1. Drift-based (raccomandato):
   - Rebuild se centroidi si spostano > threshold
   - Threshold 0.1 = buon compromesso (1-2% drift)
   - Pro: rebuild solo quando serve
   - Contro: calcolo drift ha overhead

2. Interval-based:
   - Rebuild ogni N batch update (es. 10)
   - Pro: predicibile, semplice
   - Contro: potrebbe rebuild quando non serve

3. Lazy rebuild:
   - Rebuild solo prima di query
   - Pro: zero overhead durante ingestion
   - Contro: prima query può essere lenta

COSTO REBUILD:
- 10 nodi: ~10ms
- 100 nodi: ~100ms
- 1000 nodi: ~1-5s

TUNING:
- Drift threshold basso (0.05): rebuild frequente, più accurato
- Drift threshold alto (0.2): rebuild raro, meno accurato
- Interval basso (5): rebuild spesso, overhead alto
- Interval alto (50): rebuild raro, drift maggiore
"""

"""
CENTROID UPDATE STRATEGIES:

1. 'incremental' (Real-time):
   - Update centroide ad ogni inserimento
   - Overhead: ~0.1ms per insert
   - Pro: centroidi sempre aggiornati
   - Contro: overhead 10-20% su ingestion ad alta velocità

2. 'batch' (RACCOMANDATO):
   - Update ogni N inserimenti (es. 1000)
   - Overhead: ~10ms ogni 1000 insert = 0.01ms/insert
   - Pro: overhead trascurabile (1%)
   - Contro: drift temporaneo (max N vettori non considerati)

3. 'periodic':
   - Background thread che aggiorna ogni X secondi
   - Overhead: 0% su ingestion (thread separato)
   - Pro: nessun impatto su latenza insert
   - Contro: richiede fetch da Qdrant (più lento)

4. 'none':
   - Nessun update (centroidi statici)
   - Pro: zero overhead
   - Contro: degrada nel tempo con nuovi inserimenti
"""

"""
META-HNSW ROUTING:
- Ogni nodo ha un centroide rappresentativo (media dei suoi vettori)
- Centroidi indicizzati in HNSW globale (piccolo grafo)
- Query: cerca top-k nodi simili nel meta-HNSW
- Riduce latenza: O(log N) invece di O(N) nodi
- Scalabile: funziona con 100+ nodi
"""

"""
WIKIPEDIA EMBEDDINGS:
- Scarica frasi REALI da Wikipedia
- Genera embeddings con sentence-transformers (384-dim)
- Inserisce in Qdrant con metadata (testo originale)
- Prima volta: ~5-10 min download
- Successivamente: carica da cache
"""

# ════════════════════════════════════════════════════════════════════════
# PARALLEL INGESTION
# ════════════════════════════════════════════════════════════════════════

# Numero worker threads per ingestion parallela
INGESTION_PARALLEL_WORKERS = 8  # 4-8 ottimale per 10 nodi

# Dimensione batch per upsert Qdrant
QDRANT_UPSERT_BATCH_SIZE = 512

# Wait su upsert: False = async (più veloce), True = sync (più sicuro)
QDRANT_UPSERT_WAIT = False

"""
PARALLEL INGESTION:

PROBLEMA:
- Upsert sincroni: ogni nodo attende il precedente
- Con 10 nodi × 50ms latency = 500ms per ciclo batch
- Bottleneck: I/O network diventa dominante

SOLUZIONE:
- ThreadPoolExecutor: 8 thread paralleli
- Upsert su nodi diversi in contemporanea
- Riduzione latenza: da 500ms → 50ms (10x speedup)

WORKERS:
- 4 workers: conservativo, poco overhead
- 8 workers: raccomandato per 10 nodi
- 16 workers: utile solo con 20+ nodi

BATCH SIZE:
- 512: buon compromesso tra throughput e memoria
- 1024: più veloce ma più RAM
- 256: più sicuro ma più lento

WAIT:
- False (async): più veloce, risultati eventualmente consistenti
- True (sync): più lento, garanzia scrittura immediata
"""

# ════════════════════════════════════════════════════════════════════════
# CAPACITY CONSTRAINTS & REBALANCING
# ════════════════════════════════════════════════════════════════════════

# CALCOLO DINAMICO CAPACITÀ (basato su dataset e nodi)
# Queste sono solo DEFAULT - verranno ricalcolate automaticamente

# Margine di sicurezza per capacità (es. 1.5 = 150% della media)
CAPACITY_SAFETY_MARGIN = 1.5  # Cap = media × 1.5

# Soglia per trigger rebalancing (% del cap dinamico)
REBALANCING_THRESHOLD = 0.85  # 85% del cap dinamico

# Strategia rebalancing: 'distance', 'random', 'round_robin'
REBALANCING_STRATEGY = 'distance'

# Enable/disable automatic rebalancing
ENABLE_AUTO_REBALANCING = True

# Enable rebalancing POST-ingestion
ENABLE_POST_INGESTION_REBALANCING = True

"""
CAPACITY-CONSTRAINED REBALANCING (DINAMICO):

CALCOLO AUTOMATICO CAPACITÀ:
1. Calcola carico medio: avg_load = total_vectors / num_nodes
2. Capacità dinamica: cap = avg_load × CAPACITY_SAFETY_MARGIN
3. Threshold: trigger = cap × REBALANCING_THRESHOLD

ESEMPIO 100K vettori, 10 nodi:
- avg_load = 100K / 10 = 10K
- cap = 10K × 1.5 = 15K
- threshold = 15K × 0.85 = 12.75K
→ Rebalancing se nodo > 12.75K vettori

ESEMPIO 500K vettori, 20 nodi:
- avg_load = 500K / 20 = 25K
- cap = 25K × 1.5 = 37.5K
- threshold = 37.5K × 0.85 = 31.875K
→ Rebalancing se nodo > 31.875K vettori

TUNING SAFETY_MARGIN:
- 1.2 (20%):  aggressivo, rebalancing frequente, overhead alto
- 1.5 (50%):  bilanciato, rebalancing moderato (DEFAULT)
- 2.0 (100%): conservativo, tollera più sbilanciamento
- 3.0 (200%): molto permissivo, rebalancing raro

TUNING THRESHOLD:
- 0.70 (70%): preventivo, rebalancing anticipato
- 0.85 (85%): bilanciato (DEFAULT)
- 0.95 (95%): reattivo, rebalancing last-resort

PRE-INGESTION vs POST-INGESTION:

PRE-INGESTION (Step 4b):
- Usa predizione K-means per simulare distribuzione
- Corregge overflow PRIMA di inserire
- Cap stimato: total_vectors / num_nodes × SAFETY_MARGIN
- Pro: nessun overhead durante ingestion
- Contro: predizione può differire dalla realtà

POST-INGESTION (Step 6c):
- Usa distribuzione REALE da Qdrant
- Cap preciso: calcolato su dati effettivi
- Pro: correzione accurata basata su carico reale
- Contro: richiede fetch + delete + re-insert

RACCOMANDAZIONI:
- Dataset piccolo (<50K):    SAFETY_MARGIN = 1.3, THRESHOLD = 0.80
- Dataset medio (100-500K):  SAFETY_MARGIN = 1.5, THRESHOLD = 0.85 (DEFAULT)
- Dataset grande (>1M):      SAFETY_MARGIN = 1.2, THRESHOLD = 0.90

ESEMPIO OUTPUT:
  📊 Dynamic Capacity Calculation:
      Total vectors: 100,000
      Number of nodes: 10
      Average load: 10,000 vectors/node
      Capacity (1.5× avg): 15,000 vectors/node
      Rebalancing threshold (85%): 12,750 vectors/node
"""

# ════════════════════════════════════════════════════════════════════════
# REPLICATION POLICY
# ════════════════════════════════════════════════════════════════════════

# Enable/disable replication system
ENABLE_REPLICATION = True

# Replication factor (1=no replica, 2=1 replica, 3=2 repliche)
REPLICATION_FACTOR = 2

# ═══════════════════════════════════════════════════════════════
# NUOVO: Minimo repliche per cluster
# ═══════════════════════════════════════════════════════════════

# Minimo copie per cluster (None=disabled, 2=almeno 2 copie, 3=almeno 3 copie)
MIN_REPLICAS_PER_CLUSTER = 2  # NUOVO: garantisce minimo 2 copie per TUTTI i cluster

# ═══════════════════════════════════════════════════════════════
# NUOVO: Limite repliche per nodo target
# ═══════════════════════════════════════════════════════════════

# Massimo numero di repliche che un singolo nodo può ricevere
MAX_REPLICAS_PER_NODE = 3  # NUOVO: evita sovraccarico su pochi nodi

"""
MAX_REPLICAS_PER_NODE:

PROBLEMA:
- 10 cluster × 2 copie = 20 copie totali
- Se distribuisci male: node-10 potrebbe ricevere 5+ repliche
- Risultato: node-10 sovraccarico, errore 500

SOLUZIONE:
- Limita quante repliche un singolo nodo può ricevere
- Forza distribuzione uniforme delle repliche

CALCOLO:
- 10 cluster × 2 copie = 20 copie totali
- 10 nodi disponibili
- Ogni nodo ha già 1 cluster primary = 10 primary
- Repliche da distribuire: 10 (1 replica per cluster)
- Distribuzione ideale: 10 repliche / 10 nodi = 1 replica per nodo
- Con MAX_REPLICAS_PER_NODE=3: tollera disuniformità ma previene overflow

ESEMPI:
- MAX_REPLICAS_PER_NODE=1: distribuzione perfetta (difficile raggiungere)
- MAX_REPLICAS_PER_NODE=2: bilanciato
- MAX_REPLICAS_PER_NODE=3: permissivo (DEFAULT)
- MAX_REPLICAS_PER_NODE=None: illimitato (può causare overflow)
"""

# ════════════════════════════════════════════════════════════════════════
# MIN_REPLICAS_PER_CLUSTER
# ════════════════════════════════════════════════════════════════════════

"""
MIN_REPLICAS_PER_CLUSTER:

OPZIONI:
- None (default):        Replica solo hot clusters (top 20%)
- 2 (raccomandato):      Ogni cluster ha primary + 1 replica (fault tolerance base)
- 3 (alta affidabilità): Ogni cluster ha primary + 2 repliche (tolleranza 2 nodi down)

CALCOLO STORAGE:
- 10 cluster × 10K vettori/cluster = 100K vettori totali
- MIN_REPLICAS=2: 200K vettori totali (+100% storage)
- MIN_REPLICAS=3: 300K vettori totali (+200% storage)

VANTAGGI MIN_REPLICAS=2:
✅ Fault tolerance: sistema funziona anche con 1 nodo down
✅ Read scalability: ogni query può scegliere tra 2 copie
✅ Zero downtime maintenance: puoi riavviare nodi senza interrompere servizio

SVANTAGGI:
⚠️ Storage: +100% (accettabile per sistemi production)
⚠️ Write cost: ogni write va su 2 nodi (+100% latenza write)

QUANDO USARE:
- Production systems: MIN_REPLICAS=2 (minimo)
- Mission-critical: MIN_REPLICAS=3 
- Testing/dev: None o 2
"""