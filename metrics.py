from prometheus_client import Counter, Histogram, start_http_server
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
# Buckets for latency (in seconds). 
# This covers a range from very fast (5ms) to very slow (10s) to help calculate accurate P99.
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
)

# --- Metric Definitions ---

# 1. End-to-End Request Latency (Client -> Server -> Response)
# Used to calculate Average Latency and Variance (via quantiles)
REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Total time spent processing a client request',
    ['operation'], # e.g., 'global_query', 'add_vectors'
    buckets=LATENCY_BUCKETS
)

# 2. Peer Communication Latency
# Tracks how long we wait for other nodes. Critical for finding slow peers.
PEER_LATENCY = Histogram(
    'app_peer_latency_seconds',
    'Time spent waiting for a peer response',
    ['peer_id', 'operation'], # e.g., 'node_01', 'query_peer'
    buckets=LATENCY_BUCKETS
)

# 3. Database (Qdrant) Latency
# Tracks raw DB performance excluding network overhead.
DB_LATENCY = Histogram(
    'app_db_latency_seconds',
    'Time spent waiting for local Qdrant operations',
    ['operation'], # e.g., 'search_batch', 'upload_points'
    buckets=LATENCY_BUCKETS
)

# 4. Request Throughput & Status
REQUEST_COUNT = Counter(
    'app_requests_total',
    'Total number of requests processed',
    ['operation', 'status'] # e.g. 'global_query', 'success'
)

# 5. Peer Failures
# Tracks network stability.
PEER_FAILURES = Counter(
    'app_peer_failures_total',
    'Total number of failed communications with peers',
    ['peer_id', 'operation']
)

def start_metrics_server(port):
    """Starts the Prometheus metrics server on a separate thread."""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")