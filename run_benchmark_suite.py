import argparse
import time
import sys
import json
import logging
import requests
import threading
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Ogni scenario testa il sistema con diversi carichi di lavoro

SCENARIOS = {
    # Verifica rapida (5s, 2 thread) per confermare connettività e operatività base 
    # (insert/query) senza stressare il sistema, prima di eseguire i test reali.
    "sanity_check": {
        "duration": 5,        # Durata del test in secondi
        "concurrency": 2,     # Numero di thread paralleli (client simultanei)
        "mix": {"insert": 0.5, "query": 0.5},  # Probabilità: 50% insert, 50% query
        "desc": "Quick check to verify system stability"
    },
    
    "balanced": {
        "duration": 30,       # Test più lungo per risultati statisticamente significativi
        "concurrency": 20,    # 10 client simulati che lavorano in parallelo
        "mix": {"insert": 0.5, "query": 0.5},
        "desc": "Balanced read/write workload"
    },
    
    "read_heavy": {
        "duration": 30,
        "concurrency": 20,
        "mix": {"insert": 0.2, "query": 0.8},  
        "desc": "Retrieval-heavy (80% queries)"
    },
    
    
    "write_heavy": {
        "duration": 30,
        "concurrency": 20,
        "mix": {"insert": 0.8, "query": 0.2},  
        "desc": "Ingestion-heavy (80% inserts)"
    },
    
    "stress_test": {
        "duration": 60,       # Test prolungato per vedere degrado performance
        "concurrency": 60,    # Alta concorrenza per stressare il sistema
        "mix": {"insert": 0.4, "query": 0.6},
        "desc": "High concurrency stress test"
    }
}

# --- BENCHMARK CLIENT LOGIC ---
#
# IMPORTANTE: Come funziona il load balancing
# ============================================
# Ogni operazione (insert o query) viene inviata a UN server scelto CASUALMENTE
# tra tutti quelli disponibili (es. 8001, 8002, 8003).
#
# Questo simula un load balancer reale che distribuisce il traffico.
# Le metriche AGGREGATE (prima sezione del report) sommano i risultati di TUTTI i server.
# Le metriche PER-SERVER (sezione finale) mostrano le performance di ciascun server.

class BenchmarkClient:
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.request_id = 0
        self.lock = threading.Lock()

    def get_id(self):
        with self.lock:
            self.request_id += 1
            return self.request_id

    def get_target_url(self):
        # LOAD BALANCING: sceglie casualmente un server tra quelli disponibili
        # Ogni richiesta puo' andare a un server diverso!
        return random.choice(self.urls)

    def insert(self, vector: List[float], payload_text: str) -> Tuple[float, bool, str]:
        """Sends an insert request and returns (latency, success, url_used)."""
        url = self.get_target_url()
        req_id = self.get_id()
        
        # Format: List[Tuple[Vector, Payload]]
        content = [(vector, payload_text)]
        
        payload = {
            'id': req_id,
            'content': content
        }
        
        start = time.time()
        success = False
        try:
            resp = requests.post(f"{url}/add", json=payload, timeout=10)
            if resp.status_code == 200:
                success = True
            else:
                logger.error(f"Insert failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            
        return time.time() - start, success, url

    def query(self, vector: List[float]) -> Tuple[float, bool, str]:
        """Sends a query request and returns (latency, success, url_used)."""
        url = self.get_target_url()
        req_id = self.get_id()
        
        payload = {
            'id': req_id,
            'query': [vector],
            'topk': 5
        }
        
        start = time.time()
        success = False
        try:
            resp = requests.post(f"{url}/query", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'success':
                    success = True
                else:
                    logger.error(f"Query returned failure status: {data}")
            else:
                logger.error(f"Query failed with status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
        return time.time() - start, success, url

def load_data(filepath: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading vectors from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} vectors.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []

def get_stats_summary(name: str, latencies: List[float], errors: int = 0) -> str:
    """
    Genera un riepilogo statistico delle metriche per un tipo di operazione.
    
    METRICHE CALCOLATE:
    - Error Rate: percentuale di operazioni fallite
    - Latenze: avg, P50 (mediana), P95, P99
    - Jitter (Std Dev): misura la variabilità/prevedibilità delle latenze
    - Tail Latency Ratio (P99/P50): misura la consistenza del sistema
    """
    n = len(latencies)
    total_attempts = n + errors
    
    if total_attempts == 0:
        return f"No {name} operations recorded."
    
    error_rate = (errors / total_attempts) * 100
    
    if n == 0:
        return (
            f"\n--- {name.upper()} Metrics (0 success, {errors} errors) ---\n"
            f"Error Rate:      {error_rate:.2f}%"
        )

    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if n >= 20 else max(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98] if n >= 100 else max(latencies)
    
    # NUOVA METRICA 1: Jitter (Standard Deviation)
    # Misura quanto le latenze variano dalla media.
    # - Jitter basso = sistema prevedibile e stabile
    # - Jitter alto = latenze molto variabili, sistema meno affidabile
    # 
    #    Valore	Significato
    #    < 40ms	 ECCELLENTE - Sistema molto stabile
    #    40-90ms BUONO - Variabilità normale
    #    > 90ms	 ALTO - Sistema imprevedibile
    jitter = statistics.stdev(latencies) if n >= 2 else 0.0
    
    # NUOVA METRICA 2: Tail Latency Ratio (P99/P50)
    # Misura quanto le richieste più lente (outlier) sono peggiori della mediana.
    # - Ratio ~1-2x: ECCELLENTE, sistema molto consistente, pochi outlier
    # - Ratio 3-5x : NORMALE sotto carico moderato
    # - Ratio >5x  : INSTABILE, molte richieste "sfortunate" sono molto lente
    # Importante per garantire SLA (Service Level Agreement) in produzione.
    tail_ratio = p99 / p50 if p50 > 0 else 0.0
    
    return (
        f"\n--- {name.upper()} Metrics ({n} success, {errors} errors) ---\n"
        f"Error Rate:      {error_rate:.2f}%\n"
        f"Average Latency: {avg*1000:.2f} ms\n"
        f"P50 Latency:     {p50*1000:.2f} ms\n"
        f"P95 Latency:     {p95*1000:.2f} ms\n"
        f"P99 Latency:     {p99*1000:.2f} ms\n"
        f"Jitter (StdDev): {jitter*1000:.2f} ms\n"
        f"Tail Ratio:      {tail_ratio:.2f}x (P99/P50)"
    )

def run_benchmark(
    client: BenchmarkClient,
    data: List[Dict[str, Any]],
    duration: int,         # Durata totale del test (es. 30 secondi)
    concurrency: int,      # Numero di thread paralleli (es. 10 worker)
    mix: Dict[str, float]  # Probabilità operazioni (es. 50% insert, 50% query)
):
    """
    COME FUNZIONA IL BENCHMARK:
    
    1. Crea N thread worker (concurrency), ognuno lavora in parallelo
    2. Ogni worker esegue un loop continuo per 'duration' secondi
    3. In ogni iterazione, il worker decide casualmente se fare insert o query
       basandosi sul 'mix' (es. 50% probabilità insert, 50% query)
    4. Ogni operazione misura la latenza e traccia successo/errore
    
    ESEMPIO PRATICO (scenario balanced):
    - concurrency=10 significa 10 thread che lavorano simultaneamente
    - duration=30 significa che ogni thread lavora per 30 secondi
    - mix=50/50 significa che ogni thread, a ogni iterazione, ha 50% probabilità 
      di fare insert e 50% di fare query
    - Se ogni operazione impiega ~150ms, ogni thread fa ~200 operazioni in 30s
    - Totale: 10 thread × 200 ops = ~2000 operazioni totali
    - Di cui ~1000 insert e ~1000 query (per il mix 50/50)
    """
    stop_event = threading.Event()  # Segnale per fermare tutti i worker
    
    # Struttura per raccogliere tutte le latenze e gli errori
    results = {
        'insert': [],   # Lista di tutte le latenze delle insert riuscite
        'query': [],    # Lista di tutte le latenze delle query riuscite
        'errors': {
            'insert': 0,  # Contatore insert fallite
            'query': 0    # Contatore query fallite
        },
        # NUOVO: Tracking per-server per identificare bottleneck
        'per_server': {}  # Dict: {url: {'insert': [latenze], 'query': [latenze], 'errors': count}}
    }
    results_lock = threading.Lock()  # Per accesso thread-safe ai risultati
    
    insert_ratio = mix.get('insert', 0.0)
    query_ratio = mix.get('query', 0.0)
    total_weight = insert_ratio + query_ratio
    
    if total_weight == 0:
        logger.error("Invalid mix configuration: weights sum to 0")
        return results

    # Calcola la soglia per decidere insert vs query
    # Es: se insert=0.5, query=0.5 → threshold=0.5
    # random.random() < 0.5 ha 50% probabilità di essere True
    normalized_insert_threshold = insert_ratio / total_weight

    def worker():
        """
        Funzione eseguita da ogni thread worker.
        Ogni worker fa un loop continuo finché stop_event non viene settato.
        
        COMPORTAMENTO DI UN SINGOLO WORKER:
        - Loop infinito fino allo stop_event (dopo 'duration' secondi)
        - Ogni iterazione: sceglie casualmente insert o query
        - Esegue l'operazione e registra latenza/errore
        - Continua il più velocemente possibile (no sleep)
        
        Con concurrency=10, ci sono 10 worker che fanno questo in parallelo!
        """
        local_results = {'insert': [], 'query': []}
        local_errors = {'insert': 0, 'query': 0}
        # NUOVO: Tracking locale per-server
        local_per_server = {}  # {url: {'insert': [], 'query': [], 'errors': 0}}
        
        while not stop_event.is_set():  # Continua finché il tempo non scade
            # Decisione probabilistica: insert o query?
            # random.random() genera un numero tra 0 e 1
            op_type = 'insert' if random.random() < normalized_insert_threshold else 'query'
            
            # Sceglie un vettore casuale dal dataset
            item = random.choice(data)
            vector = item['embedding']
            text = item.get('text', '')

            # Esegue l'operazione scelta e misura latenza
            if op_type == 'insert':
                latency, success, url = client.insert(vector, text)
                # Inizializza struttura per-server se necessario
                if url not in local_per_server:
                    local_per_server[url] = {'insert': [], 'query': [], 'errors': 0}
                if success:
                    local_results['insert'].append(latency)
                    local_per_server[url]['insert'].append(latency)
                else:
                    local_errors['insert'] += 1
                    local_per_server[url]['errors'] += 1
            else:
                latency, success, url = client.query(vector)
                # Inizializza struttura per-server se necessario
                if url not in local_per_server:
                    local_per_server[url] = {'insert': [], 'query': [], 'errors': 0}
                if success:
                    local_results['query'].append(latency)
                    local_per_server[url]['query'].append(latency)
                else:
                    local_errors['query'] += 1
                    local_per_server[url]['errors'] += 1
        
        # Al termine, aggrega i risultati locali in quelli globali
        with results_lock:
            results['insert'].extend(local_results['insert'])
            results['query'].extend(local_results['query'])
            results['errors']['insert'] += local_errors['insert']
            results['errors']['query'] += local_errors['query']
            # Aggrega dati per-server
            for url, data_url in local_per_server.items():
                if url not in results['per_server']:
                    results['per_server'][url] = {'insert': [], 'query': [], 'errors': 0}
                results['per_server'][url]['insert'].extend(data_url['insert'])
                results['per_server'][url]['query'].extend(data_url['query'])
                results['per_server'][url]['errors'] += data_url['errors']

    logger.info(f"Starting benchmark with {concurrency} threads for {duration} seconds...")
    logger.info(f"Workload: {insert_ratio*100:.1f}% Insert, {query_ratio*100:.1f}% Query")

    # ORCHESTRAZIONE DEI WORKER:
    # 1. Crea un pool di thread
    # 2. Lancia 'concurrency' worker in parallelo (es. 10 thread)
    # 3. Aspetta 'duration' secondi (es. 30s)
    # 4. Segnala stop a tutti i worker
    # 5. Aspetta che tutti i worker terminino e raccoglie i risultati
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Lancia tutti i worker contemporaneamente
        futures = [executor.submit(worker) for _ in range(concurrency)]
        
        # Lascia lavorare i worker per 'duration' secondi
        time.sleep(duration)
        
        # Segnala a tutti i worker di fermarsi
        stop_event.set()
        
        # Aspetta che tutti i worker terminino
        for f in futures:
            f.result()

    return results

# --- SUITE ORCHESTRATION ---

def get_vector_count(client_url):
    try:
        resp = requests.get(f"{client_url}/count", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # If data is a dict (per-node count), sum values. If int, return it.
            if isinstance(data, dict):
                 return sum([v for v in data.values() if isinstance(v, int)])
            return int(data)
    except Exception as e:
        logger.error(f"Failed to get count from {client_url}: {e}")
    return -1

def get_per_server_summary(per_server_data: Dict[str, Dict]) -> str:
    """
    Genera un report delle performance per ogni singolo server.
    Utile per identificare bottleneck e server lenti.
    
    OUTPUT:
    - Latenza media per server
    - Numero di operazioni per server
    - Confronto tra server (differenza % dalla media)
    """
    if not per_server_data:
        return "\nNo per-server data available (single server mode).\n"
    
    lines = []
    lines.append("\n" + "="*60)
    lines.append("FINAL SUMMARY: PER-SERVER PERFORMANCE ANALYSIS")
    lines.append("(Aggregated across ALL scenarios above)")
    lines.append("="*60)
    lines.append("")
    lines.append("This section shows the TOTAL performance of each server")
    lines.append("across all benchmark scenarios combined.")
    
    # Calcola metriche per ogni server
    server_stats = {}
    all_latencies = []  # Per calcolare media globale
    
    for url, data in per_server_data.items():
        all_lats = data['insert'] + data['query']
        if all_lats:
            avg_lat = statistics.mean(all_lats)
            total_ops = len(all_lats)
            errors = data['errors']
            server_stats[url] = {
                'avg_latency': avg_lat,
                'total_ops': total_ops,
                'insert_count': len(data['insert']),
                'query_count': len(data['query']),
                'errors': errors,
                'insert_avg': statistics.mean(data['insert']) if data['insert'] else 0,
                'query_avg': statistics.mean(data['query']) if data['query'] else 0
            }
            all_latencies.extend(all_lats)
    
    if not server_stats:
        return "\nNo successful operations recorded per server.\n"
    
    # Media globale per confronto
    global_avg = statistics.mean(all_latencies) if all_latencies else 0
    
    lines.append(f"\nGlobal Average Latency: {global_avg*1000:.2f} ms")
    lines.append("-"*60)
    
    # Report per ogni server
    for url, stats in sorted(server_stats.items()):
        # Calcola differenza % dalla media globale
        diff_pct = ((stats['avg_latency'] - global_avg) / global_avg * 100) if global_avg > 0 else 0
        # Indicatore di performance (senza emoji per compatibilita' Windows)
        status = "[OK]" if diff_pct <= 10 else ("[WARN]" if diff_pct <= 30 else "[SLOW]")
        
        lines.append(f"\nServer: {url}")
        lines.append(f"  Total Operations: {stats['total_ops']} ({stats['insert_count']} insert, {stats['query_count']} query)")
        lines.append(f"  Errors: {stats['errors']}")
        lines.append(f"  Avg Latency (all): {stats['avg_latency']*1000:.2f} ms")
        lines.append(f"  Avg Insert Latency: {stats['insert_avg']*1000:.2f} ms")
        lines.append(f"  Avg Query Latency: {stats['query_avg']*1000:.2f} ms")
        lines.append(f"  Diff from Global: {diff_pct:+.1f}% {status}")
    
    # Analisi bottleneck
    lines.append("\n" + "-"*60)
    lines.append("BOTTLENECK ANALYSIS:")
    
    # Trova server piu' veloce e piu' lento
    sorted_by_lat = sorted(server_stats.items(), key=lambda x: x[1]['avg_latency'])
    fastest = sorted_by_lat[0]
    slowest = sorted_by_lat[-1]
    
    speed_diff = ((slowest[1]['avg_latency'] - fastest[1]['avg_latency']) / fastest[1]['avg_latency'] * 100) if fastest[1]['avg_latency'] > 0 else 0
    
    lines.append(f"  Fastest: {fastest[0]} ({fastest[1]['avg_latency']*1000:.2f} ms)")
    lines.append(f"  Slowest: {slowest[0]} ({slowest[1]['avg_latency']*1000:.2f} ms)")
    lines.append(f"  Speed Difference: {speed_diff:.1f}%")
    
    if speed_diff > 50:
        lines.append(f"  WARNING: Server {slowest[0]} is significantly slower!")
        lines.append(f"     Consider investigating network, load, or resource issues.")
    elif speed_diff > 20:
        lines.append(f"  NOTICE: Moderate performance difference between servers.")
    else:
        lines.append(f"  OK: All servers performing similarly. No bottleneck detected.")
    
    lines.append("="*60 + "\n")
    
    return "\n".join(lines)

def populate_and_wait_clustering(client, data, target_vectors=2500):
    """
    FASE DI WARMUP: Prepara il sistema prima dei benchmark
    
    SCOPO:
    - Inserisce un numero significativo di vettori (default 2500)
    - Attiva il clustering automatico del sistema
    - Porta il sistema in uno stato "caldo" e stabile
    - Evita che il primo benchmark misuri anche il tempo di inizializzazione
    
    COME FUNZIONA:
    - Usa 20 thread concorrenti per velocizzare l'inserimento
    - Ogni thread continua a inserire finché non raggiungiamo target_vectors
    - Gli errori di timeout sono normali (sistema sotto carico) e vengono ignorati
    - Conta solo gli inserimenti riusciti
    """
    logger.info(f"WARMUP: Populating {target_vectors} vectors to trigger clustering...")
    logger.info("This ensures the system is in a clustered state before benchmarking.")
    
    concurrency = 20  # 20 thread paralleli per warmup veloce
    successful_inserts = 0
    lock = threading.Lock()
    
    def warmup_worker():
        nonlocal successful_inserts
        while True:
            # Controlla se abbiamo raggiunto l'obiettivo
            with lock:
                if successful_inserts >= target_vectors:
                    return
                current_count = successful_inserts
            
            # Sceglie un vettore casuale e prova a inserirlo
            item = random.choice(data)
            vector = item['embedding']
            text = item.get('text', '')
            
            try:
                _, success, _ = client.insert(vector, text)
                if success:
                    with lock:
                        successful_inserts += 1
                        # Progress indicator ogni 100 inserimenti
                        if successful_inserts % 100 == 0:
                            sys.stdout.write(f"\rInserted {successful_inserts}/{target_vectors}")
                            sys.stdout.flush()
            except Exception:
                # Ignora gli errori durante warmup (timeout normali sotto carico)
                pass

    logger.info(f"Starting concurrent warmup with {concurrency} threads...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(warmup_worker) for _ in range(concurrency)]
        for f in futures:
            f.result()
            
    print() # Newline
    logger.info(f"Warmup complete. {successful_inserts} vectors inserted.")
    
    logger.info("Waiting 5s to allow for clustering/stabilization...")
    time.sleep(5) 

def run_suite(urls_str, data_path, output_file):
    logger.info(f"Loading data from {data_path}...")
    data = load_data(data_path)
    if not data:
        logger.error("Failed to load data. Aborting.")
        sys.exit(1)

    urls = [u.strip() for u in urls_str.split(',')]
    client = BenchmarkClient(urls)
    
    # Initial Warmup
    initial_count = get_vector_count(urls[0])
    logger.info(f"Initial System Vector Count: {initial_count}")
    
    populate_and_wait_clustering(client, data)
    
    post_warmup_count = get_vector_count(urls[0])
    logger.info(f"Post-Warmup Vector Count: {post_warmup_count} (+{post_warmup_count - initial_count})")
    
    with open(output_file, "w") as f:
        f.write("========================================================\n")
        f.write(f"DISTRIBUTED VECTOR DB BENCHMARK SUITE REPORT\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Target URLs: {urls_str}\n")
        f.write(f"Initial Count: {initial_count}\n")
        f.write(f"Post-Warmup Count: {post_warmup_count}\n")
        f.write("========================================================\n\n")
    
    # Struttura per accumulare dati per-server da tutti gli scenari
    all_per_server_data = {}

    for name, config in SCENARIOS.items():
        logger.info(f"--- Running Scenario: {name.upper()} ---")
        logger.info(f"Description: {config['desc']}")
        
        pre_test_count = get_vector_count(urls[0])
        logger.info(f"Pre-Test Count: {pre_test_count}")

        start_time = time.time()
        results = run_benchmark(
            client, 
            data, 
            config['duration'], 
            config['concurrency'], 
            config['mix']
        )
        total_time = time.time() - start_time
        
        post_test_count = get_vector_count(urls[0])
        count_delta = post_test_count - pre_test_count
        logger.info(f"Post-Test Count: {post_test_count} (Delta: +{count_delta})")
        
        # Calculate summary metrics
        total_ops = len(results['insert']) + len(results['query']) + results['errors']['insert'] + results['errors']['query']
        throughput = total_ops / total_time if total_time > 0 else 0
        
        # ============================================================
        # GENERA REPORT - METRICHE AGGREGATE (tutti i server insieme)
        # ============================================================
        # Le metriche qui sotto sono la SOMMA/MEDIA di tutti i server.
        # Le richieste sono state distribuite casualmente tra i server.
        
        report_section = [
            f"--------------------------------------------------------",
            f"SCENARIO: {name.upper()}",
            f"Description: {config['desc']}",
            f"Duration: {config['duration']}s, Concurrency: {config['concurrency']}, Mix: {config['mix']}",
            f"",
            f"[AGGREGATE METRICS - All servers combined]",
            f"(Requests distributed randomly across: {', '.join(urls)})",
            f"",
            f"Total Requests: {total_ops}",
            f"Throughput: {throughput:.2f} req/s",
            f"Vectors Stored Delta: +{count_delta} (Validation)",
            get_stats_summary("Insert", results['insert'], results['errors']['insert']),
            get_stats_summary("Query", results['query'], results['errors']['query']),
        ]
        
        # ============================================================
        # GENERA REPORT - METRICHE PER-SERVER (dettaglio per scenario)
        # ============================================================
        # Qui mostriamo come ogni singolo server ha performato in questo scenario
        
        report_section.append("")
        report_section.append("[PER-SERVER BREAKDOWN - This scenario only]")
        
        for url in sorted(results['per_server'].keys()):
            srv_data = results['per_server'][url]
            srv_insert_count = len(srv_data['insert'])
            srv_query_count = len(srv_data['query'])
            srv_total = srv_insert_count + srv_query_count
            srv_errors = srv_data['errors']
            srv_insert_avg = statistics.mean(srv_data['insert'])*1000 if srv_data['insert'] else 0
            srv_query_avg = statistics.mean(srv_data['query'])*1000 if srv_data['query'] else 0
            
            report_section.append(f"  {url}: {srv_total} ops ({srv_insert_count} ins/{srv_query_count} qry), "
                                  f"Insert: {srv_insert_avg:.0f}ms, Query: {srv_query_avg:.0f}ms, Errors: {srv_errors}")
        
        report_section.append(f"--------------------------------------------------------\n")
        
        # STREAMING: scrive subito nel file dopo ogni scenario (salta il sanity_check)
        if name != "sanity_check":
            with open(output_file, "a") as f:
                f.write("\n".join(report_section) + "\n")
                f.flush()  # Forza la scrittura su disco
        
        # Accumula dati per-server per il SUMMARY FINALE (tutti gli scenari insieme, tranne sanity_check)
        if name != "sanity_check":
            for url, server_data in results['per_server'].items():
                if url not in all_per_server_data:
                    all_per_server_data[url] = {'insert': [], 'query': [], 'errors': 0}
                all_per_server_data[url]['insert'].extend(server_data['insert'])
                all_per_server_data[url]['query'].extend(server_data['query'])
                all_per_server_data[url]['errors'] += server_data['errors']
        
        logger.info(f"Finished {name}. Throughput: {throughput:.2f} req/s")
        # Cool down between tests
        time.sleep(2)
    
    # Scrivi il report per-server alla fine del file
    with open(output_file, "a") as f:
        f.write(get_per_server_summary(all_per_server_data))

    logger.info(f"Benchmark suite completed. Results saved to {output_file}")

def load_urls_from_config(config_path: str = "config.json") -> str:
    """
    Legge gli URL dei server dal file config.json.
    Ritorna una stringa con gli URL separati da virgola.
    NOTA: Sostituisce 'localhost' con '127.0.0.1' per evitare problemi di timeout IPv6.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Estrae gli URL dalla lista servers
        urls = [server['url'] for server in config.get('servers', [])]
        if urls:
            # IMPORTANTE: Sostituisce localhost con 127.0.0.1 per evitare timeout IPv6
            urls = [url.replace('localhost', '127.0.0.1') for url in urls]
            urls_str = ','.join(urls)
            logger.info(f"Loaded {len(urls)} server URLs from {config_path}: {urls_str}")
            return urls_str
        else:
            logger.warning(f"No servers found in {config_path}")
            return None
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found")
        return None
    except Exception as e:
        logger.warning(f"Error reading config file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Full Benchmark Suite")
    parser.add_argument("--urls", type=str, default=None, help="Comma-separated server URLs (optional, reads from config.json if not specified)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file (default: config.json)")
    parser.add_argument("--data", type=str, default="embeddings.json", help="Path to data file")
    parser.add_argument("--output", type=str, default="benchmark_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    # Se --urls non è specificato, leggi dal config.json
    urls = args.urls
    if urls is None:
        logger.info("No --urls specified, reading from config file...")
        urls = load_urls_from_config(args.config)
        if urls is None:
            logger.error("Could not load URLs from config. Please specify --urls or check config.json")
            sys.exit(1)
    
    run_suite(urls, args.data, args.output)

if __name__ == "__main__":
    main()
