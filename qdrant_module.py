import metrics
from qdrant_client import QdrantClient, models
import logging

logger = logging.getLogger(__name__)

# Helper to create a client instance.
_client_cache = {}

def get_client(url: str) -> QdrantClient:
    if url in _client_cache:
        return _client_cache[url]
        
    # prefer_grpc=True forces the client to use the gRPC port (usually 6334)
    client = QdrantClient(
        url=url, 
        grpc_port=(int(url.split(':')[-1])+1), 
        prefer_grpc=True,
        grpc_options={
            'grpc.max_send_message_length': 100 * 1024 * 1024,
            'grpc.max_receive_message_length': 100 * 1024 * 1024
        }
    )
    _client_cache[url] = client
    return client

def create_collection(url, collection_name, vector_size: int, distance: str = "Cosine"):
    client = get_client(url)
    try:
        if client.collection_exists(collection_name):
            # logger.info(f"Collection '{collection_name}' already exists on {url}")
            return True

        # Map string distance to Qdrant model
        dist_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT
        }

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=dist_map.get(distance, models.Distance.COSINE)
            )
        )
        logger.info(f"Collection '{collection_name}' created successfully on {url}")
        return True
    except Exception as e:
        logger.error(f"Error creating collection on {url}: {e}")
        return False

def query_vectors(url, collection, query, topk):
    """
    Query vectors using query_batch_points (gRPC).
    Returns a list of lists of ScoredPoint objects.
    """
    client = get_client(url)
    logger.info(f"[Qdrant] Executing batch search for {len(query)} vectors on {url} (Collection: {collection})...")
    try:
        with metrics.DB_LATENCY.labels(operation='search_batch').time():
        # Create search requests
            search_queries = [
                models.QueryRequest(
                    query=query_vector,
                    limit=topk,
                    with_payload=True,
                    with_vector=True
                ) for query_vector in query
            ]

            # Execute batch search
            results = client.query_batch_points(
                collection_name=collection,
                requests=search_queries
            )

            # Convert ScoredPoint objects to dictionaries
            # query_batch_points returns a list of QueryResponse objects
            # Each QueryResponse has a 'points' attribute which is a list of ScoredPoint
            final_results = []
            for response in results:
                # response.points is the list of ScoredPoint for that query
                for point in response.points:
                    final_results.append({
                        "id": point.id,
                        "score": point.score,
                        'payload':{
                            "string": point.payload.get("string"),
                            "vector": point.vector
                        }
                    })
            
            logger.info(f"[Qdrant] Success: found {len(final_results)} results")
            return final_results
    except Exception as e:
        logger.error(f"[Qdrant] QUERY ERROR on {url}: {e}")
        return []

def insert_vectors(url, collection, vectors, batch_size_retry, batch_size=256):
    """
    Insert vectors using upload_points.
    
    Args:
        vectors: A list of tuples/lists in the format: 
                 (vector_content, vector_id, vector_payload)
    """
    client = get_client(url)
    logger.info(f"[Qdrant] Attempting to insert {len(vectors)} vectors into {collection} on {url}...")
    
    # Ensure collection exists
    if vectors:
        dim = len(vectors[0][0])
        create_collection(url, collection, dim)

    with metrics.DB_LATENCY.labels(operation='upload_points').time(): 
        # Convert your input list to PointStruct objects
        points = [
            models.PointStruct(
                id=vector_id,
                vector=vector_content,
                payload={"string": vector_payload}
            )
            for vector_content, vector_id, vector_payload in vectors
        ]
        # Se batch_size non è specificato (o se vogliamo ottimizzare), 
        # diciamo a Qdrant di inviare tutto in una volta sola.
        # Se vectors contiene 1800 elementi, effective_batch_size sarà 1800.
        if batch_size is None or batch_size > len(vectors):
            effective_batch_size = len(vectors) # Invia tutto in un colpo solo (massima velocità)
        else:
            effective_batch_size = batch_size
            
        try:
            # Try with larger batch size first
            logger.info(f"[Qdrant] Trying upload with batch_size= {batch_size}")
            client.upload_points(
                collection_name=collection,
                points=points,
                batch_size=effective_batch_size, 
                wait=False #CAMBIATO QUESTOOO
            )
            logger.info(f"[Qdrant] Success: Inserted {len(points)} vectors with batch_size={batch_size}.")
            return True

        except Exception as e:
            logger.warning(f"[Qdrant] Upload with batch_size={batch_size} failed: {e}. Retrying with batch_size={batch_size_retry}...")
            try:
                client.upload_points(
                    collection_name=collection,
                    points=points,
                    batch_size=batch_size_retry, 
                    wait=True
                )
                logger.info(f"[Qdrant] Success: Inserted {len(points)} vectors with batch_size={batch_size_retry}.")
                return True
            except Exception as e2:
                logger.error(f"[Qdrant] INSERT ERROR on {url} (fallback failed): {e2}")
                return False

def count(url, collection):
    client = get_client(url)
    try:
        count_result = client.count(
            collection_name=collection,
            exact=True
        )
        return count_result.count
    except Exception as e:
        logger.error(f"Error counting vectors on {url}: {e}")
        return -1