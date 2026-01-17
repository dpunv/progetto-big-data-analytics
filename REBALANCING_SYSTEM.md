# Sistema di Rebalancing del Database Vettoriale Distribuito

## Indice
1. [Overview](#overview)
2. [Configurazione](#configurazione)
3. [Raccolta Metriche](#raccolta-metriche)
4. [Rilevamento Sbilanciamento](#rilevamento-sbilanciamento)
5. [Calcolo del Piano di Split](#calcolo-del-piano-di-split)
6. [Esecuzione dello Split](#esecuzione-dello-split)
7. [Distribuzione ai Peer](#distribuzione-ai-peer)
8. [Gestione Repliche](#gestione-repliche)
9. [Queue dei Vettori Pendenti](#queue-dei-vettori-pendenti)
10. [Flowchart Completo](#flowchart-completo)

---

## Overview

Il sistema di rebalancing mantiene una distribuzione equilibrata dei vettori tra i nodi del cluster. Quando un server accumula troppi vettori rispetto alla media, il cluster più pesante viene diviso in subcluster che vengono redistribuiti ai nodi meno carichi.

### Principi Chiave
- **Decentralizzato**: Ogni nodo può rilevare sbilanciamenti
- **Coordinamento**: Solo un nodo (ID minimo) esegue lo split per evitare conflitti
- **Lazy Migration**: I vettori vengono spostati solo durante lo split, non continuamente
- **Fault Tolerant**: Buffer per i vettori che arrivano durante lo split

---

## Configurazione

```python
# server.py - Linee 18-20
REBALANCE_THRESHOLD = 0.5      # 50% sopra la media = sbilanciato
REBALANCE_SPLIT_FACTOR = 3    # Dividi in 3 subcluster
REBALANCE_CHECK_INTERVAL = 86400  # Controlla ogni 24 ore (in secondi)
```

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `REBALANCE_THRESHOLD` | 0.5 | Un nodo è sovraccarico se ha >50% vettori rispetto alla media |
| `REBALANCE_SPLIT_FACTOR` | 3 | Numero di subcluster creati dallo split |
| `REBALANCE_CHECK_INTERVAL` | 86400 | Intervallo tra i controlli periodici (24 ore) |

---

## Raccolta Metriche

### Chi Raccoglie
Ogni server ha un **thread dedicato** che si avvia all'inizializzazione:

```python
# server.py - Linee 790-793
self.rebalance_thread = threading.Thread(
    target=self._rebalance_monitor_loop, daemon=True
)
self.rebalance_thread.start()
```

### Quando Avviene
```python
# server.py - Linee 795-815
def _rebalance_monitor_loop(self):
    time.sleep(REBALANCE_CHECK_INTERVAL)  # Prima attesa
    
    while self.running:
        if self.status == "clustered" and self.clusters:
            self.trigger_rebalance()
        time.sleep(REBALANCE_CHECK_INTERVAL)  # Ogni 24 ore
```

**Timeline:**
```
T=0          T=24h        T=48h        T=72h
│            │            │            │
▼            ▼            ▼            ▼
[Start]──────[Check]──────[Check]──────[Check]───►
```

### Come Vengono Raccolte: `get_load_stats()`

```python
# server.py - Linee 2247-2274
def get_load_stats(self) -> dict:
    stats = {"self": self.store.count(), "peers": {}}
    total = stats["self"]
    peer_count = 1
    
    for peer in self.get_reachable_peers():
        if peer.get_id() == self.id:
            continue
        count = peer.count()  # Chiamata remota via gRPC/QUIC/HTTP
        stats["peers"][peer.get_id()] = count
        total += count
        peer_count += 1
    
    stats["total"] = total
    stats["average"] = total / peer_count
    return stats
```

**Esempio Output:**
```python
{
    "self": 1500,           # Questo server
    "peers": {
        2: 800,             # Server 2
        3: 700              # Server 3
    },
    "total": 3000,
    "average": 1000         # 3000 / 3 server
}
```

---

## Rilevamento Sbilanciamento

### Funzione: `detect_imbalance()`

```python
# server.py - Linee 2276-2310
def detect_imbalance(self, threshold: float = None) -> Optional[int]:
    if threshold is None:
        threshold = REBALANCE_THRESHOLD  # 0.5
        
    stats = self.get_load_stats()
    avg = stats["average"]
    
    # Soglia = media * (1 + 0.5) = 1.5 * media
    overload_threshold = avg * (1 + threshold)
    
    # Controlla questo server
    if stats["self"] > overload_threshold:
        return self.id
        
    # Controlla i peer
    for peer_id, count in stats["peers"].items():
        if count > overload_threshold:
            return peer_id
            
    return None  # Bilanciato
```

**Esempio Calcolo:**
```
Server 1: 1500 vettori
Server 2: 800 vettori
Server 3: 700 vettori
─────────────────────
Totale: 3000
Media: 1000
Soglia: 1000 * 1.5 = 1500

Server 1 (1500) >= 1500 → SOVRACCARICO! ✗
Server 2 (800) < 1500 → OK ✓
Server 3 (700) < 1500 → OK ✓

Risultato: detect_imbalance() ritorna 1
```

### Chi Può Triggerare: `should_trigger_rebalance_for_cluster()`

Per evitare che più server eseguano lo stesso split, solo il server con **ID minimo** tra quelli che possiedono il cluster può triggerare:

```python
# server.py - Linee 2331-2350
def should_trigger_rebalance_for_cluster(self, cluster_id: int) -> bool:
    destinations = self.cluster_to_destinations_cache.get(cluster_id, set())
    
    if not destinations:
        return False
        
    min_id = min(destinations)
    return self.id == min_id  # Solo se sono l'ID più basso
```

**Esempio:**
```
Cluster 5 → Destinations: {1, 2, 3}
min(1, 2, 3) = 1

Server 1: should_trigger? → True ✓
Server 2: should_trigger? → False
Server 3: should_trigger? → False
```

---

## Calcolo del Piano di Split

### Funzione: `_calculate_split_parameters()`

Questa funzione calcola **come** dividere un cluster in subcluster bilanciati.

```python
# server.py - Linee 2352-2510
def _calculate_split_parameters(self, cluster_id: int, n_subclusters: int = None) -> dict:
```

### Step 1: Recupera Vettori del Cluster
```python
cluster_vectors = self.store.get_by_cluster(cluster_id)
n_vectors = len(cluster_vectors)
```

### Step 2: Calcola Dimensione Massima per Subcluster
```python
avg_size = n_vectors / n_subclusters  # es: 900 / 3 = 300
max_size = int(math.ceil(avg_size * 1.2))  # 300 * 1.2 = 360
```

### Step 3: Esegui Constrained K-Means
```python
X = np.array([v[0] for v in cluster_vectors])  # Matrice vettori
centers, labels = self._constrained_kmeans(X, n_subclusters, max_size)
```

Il **Constrained K-Means** garantisce che nessun subcluster abbia più di `max_size` vettori.

### Step 4: Trova Peer Underloaded (con Projected Load)

```python
# Ottieni le repliche attuali del cluster
cluster_replicas = self.cluster_to_destinations_cache.get(cluster_id, set())

target_peers, sender_should_keep_one = self._get_underloaded_peers(
    n_subclusters * 2,
    sender_current_load=self.store.count(),
    vectors_being_sent=n_vectors,
    n_subclusters=n_subclusters,
    cluster_being_split=cluster_id,
    cluster_replicas=cluster_replicas
)
```

#### Calcolo del Projected Load

Problema: Le repliche del cluster hanno vettori che **verranno eliminati**. Il loro `count()` attuale è fuorviante.

```python
# server.py - Linee 2865-2890
def _get_underloaded_peers(...):
    for peer in self.get_reachable_peers():
        load = peer.count()
        
        # Se questo peer è una replica del cluster in split
        if peer.get_id() in cluster_replicas:
            # Sottrai i vettori che verranno eliminati
            replica_cluster_count = peer.get_cluster_vector_count(cluster_being_split)
            load = max(0, load - replica_cluster_count)  # Projected load
```

**Esempio:**
```
Server 3 (replica di Cluster 2):
  count() = 1000 vettori totali
  get_cluster_vector_count(2) = 900 vettori del Cluster 2
  
  Projected load = 1000 - 900 = 100 vettori

Senza questa fix: Server 3 appare "carico" (1000)
Con questa fix: Server 3 appare "scarico" (100) → candidato ideale!
```

### Step 5: Assegna Destinazioni ai Subcluster

```python
for i in range(n_subclusters):
    new_id = self._generate_cluster_id()
    
    # Subcluster 0 → sempre al sender (me stesso)
    if i == 0:
        primary = self.id
    else:
        # Prendi il peer più scarico
        primary = target_peers.pop(0) if target_peers else self.id
    
    # Aggiungi repliche
    destinations = [primary]
    replicas = random.sample(other_peers, replication_factor - 1)
    destinations.extend(replicas)
    
    result_plan[new_id] = {
        "center": centers[i],
        "destinations": destinations,
        "label_idx": i
    }
```

**Esempio Output:**
```python
{
    101: {
        "center": [0.5, 0.3, ...],
        "destinations": [1, 2, 3],  # Primary=1, Replicas=2,3
        "label_idx": 0
    },
    102: {
        "center": [0.2, 0.8, ...],
        "destinations": [2, 1, 3],  # Primary=2, Replicas=1,3
        "label_idx": 1
    },
    103: {
        "center": [0.9, 0.1, ...],
        "destinations": [3, 1, 2],  # Primary=3, Replicas=1,2
        "label_idx": 2
    },
    "_labels_by_vector_id": {
        "vec_001": 0,  # → Cluster 101
        "vec_002": 1,  # → Cluster 102
        ...
    }
}
```

---

## Esecuzione dello Split

### Funzione: `split_and_distribute_cluster()`

#### Step 1: Marca il Cluster come "In Splitting"

```python
# server.py - Linee 2558-2563
with self.split_queue_lock:
    self.splitting_clusters.add(cluster_id)
    self.pending_vectors[cluster_id] = []
```

Da questo momento, **qualsiasi vettore** che arriva per questo cluster viene bufferizzato invece di essere salvato.

#### Step 2: Recupera e Riassegna Vettori

```python
cluster_vectors = self.store.get_by_cluster(cluster_id)

# Assegna ogni vettore al subcluster corrispondente
for vec_idx, label in enumerate(labels):
    vec = cluster_vectors[vec_idx]
    new_id = new_ids_list[label]
    
    # Aggiorna cluster_id nel vettore
    new_vec = (vec[0], vec[1], vec[2], new_id, vec[4], vec[5])
    result_full[new_id]["members"].append(new_vec)
```

#### Step 3: Elimina Vecchi Vettori

```python
for vec in cluster_vectors:
    self.store.remove_by_id(vec[1])
```

#### Step 4: Distribuisci ai Destinatari

```python
return self._distribute_vectors_from_split(result_full, is_coordinator=True)
```

#### Step 5: Processa Vettori Pendenti (Finally)

```python
finally:
    self._process_pending_vectors_after_split(cluster_id)
```

---

## Distribuzione ai Peer

### Funzione: `_distribute_vectors_from_split()`

Per ogni subcluster, invia i vettori alle destinazioni:

```python
for new_cluster_id, data in split_result.items():
    destinations = data["destinations"]
    
    for destination in destinations:
        if destination == self.id:
            # Salva localmente
            for vec in data["members"]:
                self.store.insert(vec)
        else:
            # Invia al peer remoto
            peer = self._get_peer_by_id(destination)
            peer.receive(data["members"], "clustered")
```

### Protezione Anti-Underload

Se il server sender perderebbe troppi vettori, tiene almeno un subcluster:

```python
# Calcola load proiettato
projected_load = self.store.count()  # Dopo la rimozione

if projected_load < avg_load * 0.7:
    # Tieni il subcluster più piccolo localmente
    force_keep_one = True
```

---

## Gestione Repliche

### Come le Repliche Vengono Notificate

Il coordinator chiama `_broadcast_cluster_update()`:

```python
# server.py - Linee 3095-3105
for peer in self.get_reachable_peers():
    if peer.get_id() != self.id:
        # Invia lo stesso split_plan alle repliche
        peer.split_and_distribute_cluster(
            old_cluster_id, 
            split_plan, 
            is_coordinator=False  # ← Replica mode!
        )
```

### Comportamento delle Repliche (`is_coordinator=False`)

```python
if not is_coordinator:
    # Solo tieni i vettori localmente se sei una destinazione
    if self.id in destinations:
        for vec in data["members"]:
            self.store.insert(vec)
    # NON replicare ad altri peer - lo fa già il coordinator
    continue
```

**Differenze Coordinator vs Replica:**

| Azione | Coordinator | Replica |
|--------|-------------|---------|
| Marca cluster come splitting | ✓ | ✓ |
| Calcola K-means | Già fatto | Usa piano ricevuto |
| Elimina vecchi vettori | ✓ | ✓ |
| Tiene vettori locali (se destinazione) | ✓ | ✓ |
| Invia a peer remoti | ✓ | ✗ |
| Processa pending vectors | ✓ | ✓ |

---

## Queue dei Vettori Pendenti

### Strutture Dati

```python
# server.py - Linee 767-770
self.splitting_clusters: Set[int] = set()     # Cluster attualmente in split
self.pending_vectors: Dict[int, list] = {}    # cluster_id → [vettori]
self.split_queue_lock = threading.Lock()       # Lock dedicato
```

### Quando i Vettori Vengono Bufferizzati

In `save_vectors()` (quando un vettore arriva al server responsabile):

```python
# server.py - Linee 1627-1645
def save_vectors(self, vectors):
    vectors_to_save = []
    
    with self.split_queue_lock:
        for vector in vectors:
            cluster_id = vector[3]
            
            if cluster_id in self.splitting_clusters:
                # BUFFER invece di salvare
                self.pending_vectors[cluster_id].append(vector)
            else:
                vectors_to_save.append(vector)
    
    # Salva solo quelli non bufferizzati
    if vectors_to_save:
        self.store.insert_batch(vectors_to_save)
```

### Quando i Vettori Vengono Processati

Dopo che lo split è completato (nel `finally` block):

```python
# server.py - Linee 2666-2700
def _process_pending_vectors_after_split(self, old_cluster_id: int):
    with self.split_queue_lock:
        pending = self.pending_vectors.pop(old_cluster_id, [])
        self.splitting_clusters.discard(old_cluster_id)
    
    if not pending:
        return
    
    # Reset cluster_id = -1 per forzare ri-routing
    re_routed_vectors = []
    for v in pending:
        new_v = (v[0], v[1], v[2], -1, v[4], v[5])  # cluster_id = -1
        re_routed_vectors.append(new_v)
    
    # Ri-routa verso i NUOVI subcluster
    self.send_to_peers(re_routed_vectors)
```

### Perché `cluster_id = -1`?

Quando un vettore ha `cluster_id = -1`, il sistema ricalcola la destinazione:

```python
# In _route_vectors_to_cluster():
destinations, cluster_id = self._calculate_destinations(vector)
# ↑ Trova il centroide più vicino tra i NUOVI cluster (101, 102, 103)
```

**Esempio:**
```
Vettore [0.52, 0.31, ...] arriva per Cluster 2 (in splitting)
  → Bufferizzato

Split completa:
  Cluster 2 → Cluster 101, 102, 103

Ri-routing:
  cluster_id = -1 → ricalcola
  Distanza a 101: 0.35
  Distanza a 102: 0.02 ← MINIMA
  Distanza a 103: 0.45
  
  → Vettore va nel Cluster 102
```

---

## Flowchart Completo

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         REBALANCING SYSTEM FLOWCHART                            │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Server Start   │
                              └────────┬─────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │  rebalance_thread.start()  │
                         └─────────────┬───────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────┐
                         │  sleep(86400)  // 24 ore   │◄────────────────────┐
                         └─────────────┬───────────────┘                    │
                                       │                                    │
                                       ▼                                    │
                    ┌──────────────────────────────────────┐                │
                    │  status == "clustered" && clusters?  │                │
                    └──────────────────┬───────────────────┘                │
                                       │                                    │
                            ┌──────────┴──────────┐                         │
                            │ Yes                 │ No                      │
                            ▼                     └──────────────────────────┤
               ┌────────────────────────┐                                   │
               │   trigger_rebalance()  │                                   │
               └───────────┬────────────┘                                   │
                           │                                                │
                           ▼                                                │
              ┌─────────────────────────────┐                               │
              │     get_load_stats()        │                               │
              │  ┌─────────────────────┐    │                               │
              │  │ Chiama peer.count() │    │                               │
              │  │ per ogni peer       │    │                               │
              │  │ raggiungibile       │    │                               │
              │  └─────────────────────┘    │                               │
              └───────────┬─────────────────┘                               │
                          │                                                 │
                          ▼                                                 │
              ┌─────────────────────────────┐                               │
              │    detect_imbalance()       │                               │
              │  ┌─────────────────────┐    │                               │
              │  │ avg = total / peers │    │                               │
              │  │ threshold = avg*1.5 │    │                               │
              │  │ if any > threshold  │    │                               │
              │  │   return overloaded │    │                               │
              │  └─────────────────────┘    │                               │
              └───────────┬─────────────────┘                               │
                          │                                                 │
                 ┌────────┴────────┐                                        │
                 │ Sbilanciato?    │                                        │
                 ▼                 ▼                                        │
              ┌──────┐         ┌──────┐                                     │
              │  Sì  │         │  No  │─────────────────────────────────────┤
              └──┬───┘         └──────┘                                     │
                 │                                                          │
                 ▼                                                          │
    ┌──────────────────────────────┐                                        │
    │  get_heaviest_cluster()      │                                        │
    │  (trova cluster più grande)  │                                        │
    └──────────────┬───────────────┘                                        │
                   │                                                        │
                   ▼                                                        │
    ┌──────────────────────────────────────┐                                │
    │  should_trigger_rebalance_for_cluster│                                │
    │  ┌────────────────────────────────┐  │                                │
    │  │ if self.id == min(destinations)│  │                                │
    │  │   return True                  │  │                                │
    │  │ else                           │  │                                │
    │  │   return False                 │  │                                │
    │  └────────────────────────────────┘  │                                │
    └──────────────────┬───────────────────┘                                │
                       │                                                    │
              ┌────────┴────────┐                                           │
              │ Sono io?        │                                           │
              ▼                 ▼                                           │
           ┌──────┐         ┌──────┐                                        │
           │  Sì  │         │  No  │────────────────────────────────────────┤
           └──┬───┘         └──────┘                                        │
              │                                                             │
              ▼                                                             │
┌──────────────────────────────────────────────────────────────────────────┐│
│                    _calculate_split_parameters()                          ││
│  ┌────────────────────────────────────────────────────────────────────┐  ││
│  │ 1. cluster_vectors = store.get_by_cluster(cluster_id)              │  ││
│  │ 2. X = np.array([v[0] for v in cluster_vectors])                   │  ││
│  │ 3. centers, labels = _constrained_kmeans(X, 3, max_size)           │  ││
│  │ 4. cluster_replicas = cluster_to_destinations_cache[cluster_id]    │  ││
│  │ 5. target_peers = _get_underloaded_peers(                          │  ││
│  │        ...,                                                         │  ││
│  │        cluster_being_split=cluster_id,                             │  ││
│  │        cluster_replicas=cluster_replicas  ← PROJECTED LOAD         │  ││
│  │    )                                                                │  ││
│  │ 6. Per ogni subcluster:                                             │  ││
│  │      - Assegna primary (self o peer scarico)                       │  ││
│  │      - Assegna repliche                                             │  ││
│  │      - result_plan[new_id] = {center, destinations}                │  ││
│  └────────────────────────────────────────────────────────────────────┘  ││
└────────────────────────────────────┬─────────────────────────────────────┘│
                                     │                                      │
                                     ▼                                      │
┌──────────────────────────────────────────────────────────────────────────┐│
│               split_and_distribute_cluster(cluster_id, split_plan)        ││
│  ┌────────────────────────────────────────────────────────────────────┐  ││
│  │                                                                    │  ││
│  │  ┌─────────────────────────────────────────────────────────────┐  │  ││
│  │  │ 1. MARK SPLITTING                                           │  │  ││
│  │  │    splitting_clusters.add(cluster_id)                       │  │  ││
│  │  │    pending_vectors[cluster_id] = []                         │  │  ││
│  │  └─────────────────────────────────────────────────────────────┘  │  ││
│  │                           │                                        │  ││
│  │                           ▼                                        │  ││
│  │  ┌─────────────────────────────────────────────────────────────┐  │  ││
│  │  │ 2. RIASSEGNA VETTORI                                        │  │  ││
│  │  │    for vec in cluster_vectors:                              │  │  ││
│  │  │      label = labels[vec_id]                                 │  │  ││
│  │  │      new_cluster_id = new_ids_list[label]                   │  │  ││
│  │  │      vec.cluster_id = new_cluster_id                        │  │  ││
│  │  └─────────────────────────────────────────────────────────────┘  │  ││
│  │                           │                                        │  ││
│  │                           ▼                                        │  ││
│  │  ┌─────────────────────────────────────────────────────────────┐  │  ││
│  │  │ 3. ELIMINA VECCHI VETTORI                                   │  │  ││
│  │  │    for vec in cluster_vectors:                              │  │  ││
│  │  │      store.remove_by_id(vec[1])                             │  │  ││
│  │  └─────────────────────────────────────────────────────────────┘  │  ││
│  │                           │                                        │  ││
│  │                           ▼                                        │  ││
│  │  ┌─────────────────────────────────────────────────────────────┐  │  ││
│  │  │ 4. DISTRIBUISCI (_distribute_vectors_from_split)            │  │  ││
│  │  │    for new_cluster, data in result:                         │  │  ││
│  │  │      for dest in data.destinations:                         │  │  ││
│  │  │        if dest == self.id:                                  │  │  ││
│  │  │          store.insert(vec)  # Tieni locale                  │  │  ││
│  │  │        else:                                                │  │  ││
│  │  │          peer.receive(vecs, "clustered")  # Invia           │  │  ││
│  │  └─────────────────────────────────────────────────────────────┘  │  ││
│  │                           │                                        │  ││
│  │                           ▼                                        │  ││
│  │  ┌─────────────────────────────────────────────────────────────┐  │  ││
│  │  │ 5. FINALLY: PROCESSA PENDING VECTORS                        │  │  ││
│  │  │    pending = pending_vectors.pop(cluster_id)                │  │  ││
│  │  │    splitting_clusters.discard(cluster_id)                   │  │  ││
│  │  │    for v in pending:                                        │  │  ││
│  │  │      v.cluster_id = -1  # Forza ricalcolo                   │  │  ││
│  │  │    send_to_peers(pending)  # Ri-routing                     │  │  ││
│  │  └─────────────────────────────────────────────────────────────┘  │  ││
│  │                                                                    │  ││
│  └────────────────────────────────────────────────────────────────────┘  ││
└────────────────────────────────────┬─────────────────────────────────────┘│
                                     │                                      │
                                     ▼                                      │
┌──────────────────────────────────────────────────────────────────────────┐│
│                    _broadcast_cluster_update()                            ││
│  ┌────────────────────────────────────────────────────────────────────┐  ││
│  │ for peer in reachable_peers:                                       │  ││
│  │   peer.split_and_distribute_cluster(                               │  ││
│  │     cluster_id,                                                    │  ││
│  │     split_plan,                                                    │  ││
│  │     is_coordinator=False  ← REPLICA MODE                           │  ││
│  │   )                                                                │  ││
│  └────────────────────────────────────────────────────────────────────┘  ││
└────────────────────────────────────┬─────────────────────────────────────┘│
                                     │                                      │
                                     └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                           PARALLEL: VECTOR ARRIVAL DURING SPLIT
═══════════════════════════════════════════════════════════════════════════════

    Client                    Server A (Router)               Server B (Owner)
       │                            │                               │
       │  insert vector             │                               │
       │  (cluster_id = 2)          │                               │
       │ ──────────────────────────►│                               │
       │                            │                               │
       │                            │  route to Server B            │
       │                            │  (responsible for cluster 2)  │
       │                            │ ─────────────────────────────►│
       │                            │                               │
       │                            │                    ┌──────────┴──────────┐
       │                            │                    │    save_vectors()   │
       │                            │                    │                     │
       │                            │                    │  if cluster_id in   │
       │                            │                    │  splitting_clusters:│
       │                            │                    │    BUFFER IT        │
       │                            │                    │  else:              │
       │                            │                    │    store.insert()   │
       │                            │                    └──────────┬──────────┘
       │                            │                               │
       │                            │                               │
       │                            │                    [After split completes]
       │                            │                               │
       │                            │                    ┌──────────┴──────────┐
       │                            │                    │ _process_pending_   │
       │                            │                    │ vectors_after_split │
       │                            │                    │                     │
       │                            │                    │ v.cluster_id = -1   │
       │                            │                    │ send_to_peers(v)    │
       │                            │                    │   → recalculates    │
       │                            │                    │   → finds new       │
       │                            │                    │     cluster 101/102 │
       │                            │                    └─────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              STATE DIAGRAM: CLUSTER LIFECYCLE
═══════════════════════════════════════════════════════════════════════════════

                                    ┌─────────────┐
                                    │   NORMAL    │
                                    │  (cluster   │
                                    │   exists)   │
                                    └──────┬──────┘
                                           │
                                           │ detect_imbalance() returns this cluster
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  SPLITTING  │
                   ┌───────────────►│             │◄───────────────┐
                   │                │  splitting_ │                │
                   │                │  clusters   │                │
                   │                │  .add(id)   │                │
                   │                └──────┬──────┘                │
                   │                       │                       │
     Vectors       │                       │ split completes       │     Vectors
     for this      │                       │                       │     for this
     cluster       │                       ▼                       │     cluster
     arrive        │                ┌─────────────┐                │     arrive
         │         │                │   DELETED   │                │         │
         │         │                │  (old id    │                │         │
         │         │                │   removed)  │                │         │
         │         │                └──────┬──────┘                │         │
         │         │                       │                       │         │
         │         │                       │                       │         │
         ▼         │                       ▼                       │         ▼
    ┌──────────┐   │           ┌─────────────────────┐             │   ┌──────────┐
    │ BUFFERED │───┘           │    NEW SUBCLUSTERS  │             └───│ BUFFERED │
    │ in       │               │    101, 102, 103    │                 │ in       │
    │ pending_ │               │    (normal state)   │                 │ pending_ │
    │ vectors  │               └─────────────────────┘                 │ vectors  │
    └──────────┘                         ▲                             └──────────┘
         │                               │
         │      _process_pending_        │
         │      vectors_after_split()    │
         │                               │
         └───────────────────────────────┘
              Ri-routing: cluster_id = -1
              → _calculate_destinations()
              → nearest centroid (101/102/103)
```

---

## Riepilogo Temporale

```
T = 0:00:00    Server avviato
               └─► rebalance_thread.start()
               └─► sleep(86400)

T = 24:00:00   Primo check
               └─► get_load_stats()
               └─► detect_imbalance()
                   └─► Nessuno sbilanciato → skip

T = 48:00:00   Secondo check
               └─► get_load_stats()
                   Server 1: 1500, Server 2: 800, Server 3: 700
                   avg = 1000, threshold = 1500
               └─► detect_imbalance() → Server 1 sovraccarico
               └─► get_heaviest_cluster(1) → Cluster 2 (900 vettori)
               └─► should_trigger...() → Server 1 è il min ID, procedi
               └─► _calculate_split_parameters(2)
                   └─► Constrained K-Means → 3 centri
                   └─► _get_underloaded_peers() con PROJECTED LOAD
                       Server 3: 700 - 700 = 0 (replica di Cluster 2)
                   └─► Destinazioni: 101→S1, 102→S2, 103→S3
               └─► split_and_distribute_cluster(2, plan)
                   └─► splitting_clusters.add(2)
                   └─► Riassegna vettori ai nuovi cluster
                   └─► Elimina vecchi vettori
                   └─► Distribuisci a S1, S2, S3
                   └─► _process_pending_vectors_after_split(2)
               └─► _broadcast_cluster_update()
                   └─► S2.split_and_distribute(is_coordinator=False)
                   └─► S3.split_and_distribute(is_coordinator=False)

T = 72:00:00   Terzo check
               └─► Bilanciato → skip
```

---

## Conclusione

Il sistema di rebalancing garantisce:

1. **Bilanciamento automatico** - Check periodici ogni 24 ore
2. **Coordinamento sicuro** - Solo il nodo con ID minimo trigga
3. **Divisione bilanciata** - Constrained K-Means evita subcluster sbilanciati
4. **Distribuzione intelligente** - Projected load considera vettori in eliminazione
5. **Nessuna perdita dati** - Queue per vettori che arrivano durante lo split
6. **Replica consistency** - Coordinator + replica mode per sincronizzazione
