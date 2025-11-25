from qdrant_client import QdrantClient, models
import sys
import logging

logger = logging.getLogger(__name__)

# Helper to create a client instance.
def get_client(url: str) -> QdrantClient:
    # prefer_grpc=True forces the client to use the gRPC port (usually 6334)
    return QdrantClient(url=url, grpc_port=(int(url.split(':')[-1])+1), prefer_grpc=True)

def create_collection(url, collection_name, vector_size: int, distance: str = "Cosine"):
    client = get_client(url)
    try:
        if client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists on {url}")
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
    logger.info(f"[Qdrant] Executing batch search for {len(query)} vectors on {url}...")
    try:
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

def insert_vectors(url, collection, vectors, batch_size=256):
    """
    Insert vectors using upload_points.
    
    Args:
        vectors: A list of tuples/lists in the format: 
                 (vector_content, vector_id, vector_payload)
    """
    client = get_client(url)
    logger.info(f"[Qdrant] Attempting to insert {len(vectors)} vectors into {collection} on {url}...")
    try:
        # Convert your input list to PointStruct objects
        points = [
            models.PointStruct(
                id=vector_id,
                vector=vector_content,
                payload={"string": vector_payload}
            )
            for vector_content, vector_id, vector_payload in vectors
        ]

        # upload_points automatically handles batching and retries
        client.upload_points(
            collection_name=collection,
            points=points,
            batch_size=batch_size, 
            wait=True
        )

        logger.info(f"[Qdrant] Success: Inserted {len(points)} vectors.")
        return True

    except Exception as e:
        logger.error(f"[Qdrant] INSERT ERROR on {url}: {e}")
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