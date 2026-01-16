import logging
import threading
from itertools import groupby
from operator import itemgetter

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)
GLOBAL_LOCK = threading.RLock()


def get_collection_name(name, cluster):
    return f"{name}_{cluster}"


# Helper to create a client instance.
_client_cache = {}


def get_client(url: str) -> QdrantClient:
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


def create_collection(url, collection_name, vector_size: int, distance: str = "Cosine"):
    with GLOBAL_LOCK:
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
    with GLOBAL_LOCK:
        try:
            client = get_client(url)
            client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted on {url}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name} on {url}: {e}")
            return False


def query_vectors_generic(url, collection, query, topk):
    with GLOBAL_LOCK:
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
    Returns a list of lists of ScoredPoint objects.
    """
    with GLOBAL_LOCK:
        client = get_client(url)
        logger.info(
            f"[Qdrant] Executing batch search for {len(query)} vectors on {url}..."
        )
        try:
            # Create search requests
            search_queries = [
                models.QueryRequest(
                    query=(
                        query_vector[0]
                        if isinstance(query_vector, tuple)
                        else query_vector
                    ),
                    limit=topk,
                    with_payload=True,
                    with_vector=True,
                )
                for query_vector in query
            ]

            # Execute batch search
            results = client.query_batch_points(
                collection_name=collection, requests=search_queries
            )

            # Convert ScoredPoint objects to dictionaries
            # query_batch_points returns a list of QueryResponse objects
            # Each QueryResponse has a 'points' attribute which is a list of ScoredPoint
            final_results = []
            for response in results:
                # response.points is the list of ScoredPoint for that query
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

            logger.info(f"[Qdrant] Success: found {len(final_results)} results")
            return final_results
        except Exception as e:
            logger.error(f"[Qdrant] QUERY ERROR on {url}: {e}")
            return []


def insert_vectors_generic(url, collection, vectors, batch_size_retry, batch_size=256):
    with GLOBAL_LOCK:
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

    Args:
        vectors: A list of tuples/lists in the format:
                 (vector_content, vector_id, vector_payload)
    """
    with GLOBAL_LOCK:
        client = get_client(url)
        logger.info(
            f"[Qdrant] Attempting to insert {len(vectors)} vectors into {collection} on {url}..."
        )

        # Convert your input list to PointStruct objects
        points = [
            models.PointStruct(
                id=vector_id,
                vector=vector_content,
                payload={"string": vector_payload, "cluster_id": cluster_id},
            )
            for vector_content, vector_id, vector_payload, cluster_id in vectors
        ]
        # Se batch_size non è specificato (o se vogliamo ottimizzare),
        # diciamo a Qdrant di inviare tutto in una volta sola.
        # Se vectors contiene 1800 elementi, effective_batch_size sarà 1800.
        if batch_size is None or batch_size > len(vectors):
            effective_batch_size = len(
                vectors
            )  # Invia tutto in un colpo solo (massima velocità)
        else:
            effective_batch_size = batch_size

        try:
            # Try with larger batch size first
            logger.info(f"[Qdrant] Trying upload with batch_size= {batch_size}")
            client.upload_points(
                collection_name=collection,
                points=points,
                batch_size=effective_batch_size,
                wait=True,
            )
            logger.info(
                f"[Qdrant] Success: Inserted {len(points)} vectors with batch_size={batch_size}."
            )
            return True

        except Exception as e:
            logger.warning(
                f"[Qdrant] Upload with batch_size={batch_size} failed: {e}. Retrying with batch_size={batch_size_retry}..."
            )
            import time

            time.sleep(1)
            try:
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
    with GLOBAL_LOCK:
        client = get_client(url)
        try:
            count_result = client.count(collection_name=collection, exact=True)
            return count_result.count
        except Exception as e:
            logger.error(f"Error counting vectors on {url}: {e}")
            return 0


def retrieve_vector(url, collection, vector_id):
    with GLOBAL_LOCK:
        client = get_client(url)
        try:
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
    with GLOBAL_LOCK:
        client = get_client(url)
        try:
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
    with GLOBAL_LOCK:
        client = get_client(url)
        try:
            # Construct filter
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                ]
            )
            
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
    with GLOBAL_LOCK:
        client = get_client(url)
        all_points = []
        offset = None
        try:
            while True:
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
