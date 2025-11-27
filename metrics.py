from prometheus_client import Counter, Histogram, start_http_server
import logging

logger = logging.getLogger(__name__)

LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Total time spent processing a client request',
    ['operation'],
    buckets=LATENCY_BUCKETS
)

# Per-query latency tracking
QUERY_LATENCY = Histogram(
    'app_query_latency_seconds',
    'Latency for individual queries',
    ['query_type'],  # 'global', 'local', 'peer'
    buckets=LATENCY_BUCKETS
)

PEER_LATENCY = Histogram(
    'app_peer_latency_seconds',
    'Time spent waiting for a peer response',
    ['peer_id', 'operation'],
    buckets=LATENCY_BUCKETS
)

DB_LATENCY = Histogram(
    'app_db_latency_seconds',
    'Time spent waiting for local Qdrant operations',
    ['operation'], # e.g., 'search_batch', 'upload_points'
    buckets=LATENCY_BUCKETS
)

REQUEST_COUNT = Counter(
    'app_requests_total',
    'Total number of requests processed',
    ['operation', 'status'] # e.g. 'global_query', 'success'
)

# Query distribution counter
QUERY_RECEIVED = Counter(
    'app_queries_received_total',
    'Total number of queries received by this node',
    ['query_type', 'source']  # query_type: 'global'/'local'/'peer', source: 'client'/'peer'
)

# Query routing counter
QUERY_ROUTED = Counter(
    'app_queries_routed_total',
    'Total number of queries routed to other nodes',
    ['target_node']
)

# Vectors routed (more granular than queries)
VECTORS_ROUTED = Counter(
    'app_vectors_routed_total',
    'Total number of individual vectors routed to nodes',
    ['target_node']
)

# Cluster hit rate
CLUSTER_HITS = Counter(
    'app_cluster_hits_total',
    'Number of times a cluster was selected for routing',
    ['cluster_id']
)

PEER_FAILURES = Counter(
    'app_peer_failures_total',
    'Total number of failed communications with peers',
    ['peer_id', 'operation']
)

# Rebalancing metrics
REBALANCING_EVENTS = Counter(
    'app_rebalancing_events_total',
    'Total number of rebalancing events',
    ['event_type']  # 'triggered', 'completed', 'failed'
)

def start_metrics_server(port):
    """Starts the Prometheus metrics server on a separate thread."""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")