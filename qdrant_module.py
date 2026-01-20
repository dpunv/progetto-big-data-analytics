import contextlib
import logging
import threading
from itertools import groupby
from operator import itemgetter

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# Lock only for client cache access (QdrantClient itself is thread-safe)
_cache_lock = threading.Lock()


def get_collection_name(name, cluster):
    return f"{name}_{cluster}"


# Helper to create a client instance.
_client_cache = {}
_io_locks = {}
_io_locks_lock = threading.Lock()


def get_lock(url: str):
    """
    Get a lock for the specific Qdrant URL/Path.

    QdrantClient with local storage (path or :memory:) is NOT thread-safe for
    concurrent writes and reads (causing numpy broadcast errors).
    We must serialize access to local instances.
    HTTP clients are thread-safe (server handles concurrency).
    """
    if url.startswith("http"):
        return contextlib.nullcontext()

    with _io_locks_lock:
        if url not in _io_locks:
            _io_locks[url] = threading.RLock()
        return _io_locks[url]


def get_client(url: str) -> QdrantClient:
    """Get or create a QdrantClient for the given URL. Thread-safe."""
    # Fast path: check without lock
    if url in _client_cache:
        return _client_cache[url]

    # Slow path: acquire lock and create client
    with _cache_lock:
        # Double-check after acquiring lock
        if url in _client_cache:
            return _client_cache[url]

        # Handle :memory: or local path
        if url == ":memory:":
            client = QdrantClient(location=url)
        elif not url.startswith("http"):
            client = QdrantClient(path=url)
        else:
            # prefer_grpc=True forces the client to use the gRPC port (usually 6334)
            try:
                grpc_port = int(url.split(":")[-1]) + 1
            except ValueError:
                grpc_port = 6334  # Default

            client = QdrantClient(
                url=url,
                grpc_port=grpc_port,
                prefer_grpc=True,
                timeout=60,
                grpc_options={
                    "grpc.max_send_message_length": 512 * 1024 * 1024,
                    "grpc.max_receive_message_length": 512 * 1024 * 1024,
                },
            )
        _client_cache[url] = client
        return client


def close_all_clients():
    """Close all cached QdrantClient instances."""
    with _cache_lock:
        for url, client in list(_client_cache.items()):
            try:
                client.close()
                logger.info(f"Closed QdrantClient for {url}")
            except Exception as e:
                logger.error(f"Error closing QdrantClient for {url}: {e}")
        _client_cache.clear()


def create_collection(url, collection_name, vector_size: int, distance: str = "Cosine"):
    """Create a collection. Thread-safe (QdrantClient handles concurrency)."""
    client = get_client(url)
    try:
        if client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists on {url}")
            return True

        # Map string distance to Qdrant model
        dist_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }

        with get_lock(url):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=dist_map.get(distance, models.Distance.COSINE),
                ),
            )
        logger.info(f"Collection '{collection_name}' created successfully on {url}")
        return True
    except Exception as e:
        logger.error(f"Error creating collection on {url}: {e}")
        return False


def delete_collection(url: str, collection_name: str) -> bool:
    """Delete a collection."""
    try:
        client = get_client(url)
        with get_lock(url):
            client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted on {url}")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name} on {url}: {e}")
        return False


def query_vectors_generic(url, collection, query, topk):
    """Query vectors across multiple collections."""
    keyfunc = itemgetter(1)
    vectors_sorted = sorted(query, key=keyfunc)
    grouped = [
        query_vectors(url, get_collection_name(collection, key), list(group), topk)
        for key, group in groupby(vectors_sorted, keyfunc)
    ]
    return [item for sublist in grouped for item in sublist]


def query_vectors(url, collection, query, topk):
    """
    Query vectors using query_batch_points (gRPC).
    Returns a list of dicts with id, score, payload.
    Thread-safe (QdrantClient handles concurrency).
    """
    client = get_client(url)
    try:
        # Create search requests
        search_queries = [
            models.QueryRequest(
                query=(
                    query_vector[0] if isinstance(query_vector, tuple) else query_vector
                ),
                limit=topk,
                with_payload=True,
                with_vector=True,
            )
            for query_vector in query
        ]

        # Execute batch search
        with get_lock(url):
            results = client.query_batch_points(
                collection_name=collection, requests=search_queries
            )

        # Convert ScoredPoint objects to dictionaries
        final_results = []
        for response in results:
            for point in response.points:
                final_results.append(
                    {
                        "id": point.id,
                        "score": point.score,
                        "payload": {
                            "string": point.payload.get("string"),
                            "vector": point.vector,
                        },
                    }
                )

        # logger.info(f"[Qdrant] Success: found {len(final_results)} results")
        return final_results
    except Exception as e:
        logger.error(f"[Qdrant] QUERY ERROR on {url}: {e}")
        return []


def insert_vectors_generic(url, collection, vectors, batch_size_retry, batch_size=256):
    """Insert vectors into multiple collections based on cluster_id."""
    keyfunc = itemgetter(3)
    vectors_sorted = sorted(vectors, key=keyfunc)
    [
        insert_vectors(
            url,
            get_collection_name(collection, key),
            list(group),
            batch_size_retry,
            batch_size,
        )
        for key, group in groupby(vectors_sorted, keyfunc)
    ]


def insert_vectors(url, collection, vectors, batch_size_retry, batch_size=256):
    """
    Insert vectors using upload_points.
    Thread-safe (QdrantClient handles concurrency).

    Args:
        vectors: A list of tuples/lists in the format:
                 (vector_content, vector_id, vector_payload, cluster_id)
    """
    client = get_client(url)
    # logger.info(
    #    f"[Qdrant] Attempting to insert {len(vectors)} vectors into {collection} on {url}..."
    # )

    # Convert your input list to PointStruct objects
    points = [
        models.PointStruct(
            id=vector_id,
            vector=vector_content,
            payload={"string": vector_payload, "cluster_id": cluster_id},
        )
        for vector_content, vector_id, vector_payload, cluster_id in vectors
    ]

    if batch_size is None or batch_size > len(vectors):
        effective_batch_size = len(vectors)
    else:
        effective_batch_size = batch_size

    try:
        # logger.info(f"[Qdrant] Trying upload with batch_size= {batch_size}")
        with get_lock(url):
            client.upload_points(
                collection_name=collection,
                points=points,
                batch_size=effective_batch_size,
                wait=True,
            )
        # logger.info(
        #    f"[Qdrant] Success: Inserted {len(points)} vectors with batch_size={batch_size}."
        # )
        return True

    except Exception as e:
        logger.warning(
            f"[Qdrant] Upload with batch_size={batch_size} failed: {e}. Retrying with batch_size={batch_size_retry}..."
        )
        import time

        time.sleep(1)
        time.sleep(1)
        try:
            with get_lock(url):
                client.upload_points(
                    collection_name=collection,
                    points=points,
                    batch_size=batch_size_retry,
                    wait=True,
                )
            logger.info(
                f"[Qdrant] Success: Inserted {len(points)} vectors with batch_size={batch_size_retry}."
            )
            return True
        except Exception as e2:
            logger.error(f"[Qdrant] INSERT ERROR on {url} (fallback failed): {e2}")
            return False


def count(url, collection):
    """Count vectors in a collection."""
    client = get_client(url)
    try:
        with get_lock(url):
            count_result = client.count(collection_name=collection, exact=True)
        return count_result.count
    except Exception as e:
        logger.error(f"Error counting vectors on {url}: {e}")
        return 0


def retrieve_vector(url, collection, vector_id):
    """Retrieve a single vector by ID."""
    client = get_client(url)
    try:
        with get_lock(url):
            points = client.retrieve(
                collection_name=collection,
                ids=[vector_id],
                with_vectors=True,
                with_payload=True,
            )
        if points:
            return points[0]
        return None
    except Exception as e:
        logger.error(f"Error retrieving vector {vector_id} on {url}: {e}")
        return None


def delete_vector(url, collection, vector_id):
    """Delete a single vector by ID."""
    client = get_client(url)
    try:
        with get_lock(url):
            client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[vector_id]),
                wait=True,
            )
        return True
    except Exception as e:
        logger.error(f"Error deleting vector {vector_id} on {url}: {e}")
        return False


def delete_vectors_by_payload(url, collection, key, value):
    """Delete vectors where payload[key] == value."""
    client = get_client(url)
    try:
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            ]
        )

        with get_lock(url):
            client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=filter_condition),
                wait=True,
            )
        logger.info(f"Deleted vectors with {key}={value} on {url}")
        return True
    except Exception as e:
        logger.error(f"Error deleting vectors by payload {key}={value} on {url}: {e}")
        return False


def get_all_vectors(url, collection):
    """Scroll through all vectors in a collection."""
    client = get_client(url)
    all_points = []
    offset = None
    try:
        while True:
            with get_lock(url):
                points, offset = client.scroll(
                    collection_name=collection,
                    offset=offset,
                    limit=1000,
                    with_payload=True,
                    with_vectors=True,
                )
            all_points.extend(points)
            if offset is None:
                break
        return all_points
    except Exception as e:
        logger.error(f"Error scrolling vectors on {url}: {e}")
        return []
