from qdrant_client import QdrantClient, models
import sys

# Helper to create a client instance.
# Since creating a connection has overhead, it is better to instantiate this once
# and pass the 'client' object around, but to keep your function signatures 
# similar to your original code, I will instantiate it inside functions.
def get_client(url: str) -> QdrantClient:
    # prefer_grpc=True forces the client to use the gRPC port (usually 6334)
    return QdrantClient(url=url, grpc_port=(int(url.split(':')[-1])+1), prefer_grpc=True)

def create_collection(url, collection_name, vector_size: int, distance: str = "Cosine"):
    client = get_client(url)
    try:
        if client.collection_exists(collection_name):
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
        return True
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

def query_vectors(url, collection, query, topk):
    """
    Query vectors using search_batch (gRPC).
    Returns a list of lists of ScoredPoint objects.
    """
    client = get_client(url)
    try:
        # Create search requests
        search_queries = [
            models.SearchRequest(
                vector=query_vector,
                limit=topk,
                with_payload=True,
                with_vector=True
            ) for query_vector in query
        ]

        # Execute batch search
        results = client.search_batch(
            collection_name=collection,
            requests=search_queries
        )

        # Convert ScoredPoint objects to dictionaries
        results = [
            {
                "id": point.id,
                "score": point.score,
                'payload':{
                    "string": point.payload,
                    "vector": point.vector
                }
            }
            for batch in results
            for point in batch
        ]
        
        print(f"[Qdrant Query] Success: processed {len(results)} query results")
        return results

    except Exception as e:
        print(f"[Qdrant Query] ERROR: {e}")
        return None

def insert_vectors(url, collection, vectors, batch_size=256):
    """
    Insert vectors using upload_points.
    
    Args:
        vectors: A list of tuples/lists in the format: 
                 (vector_content, vector_id, vector_payload)
    """
    client = get_client(url)
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
        # It is much faster than manual requests loops
        client.upload_points(
            collection_name=collection,
            points=points,
            batch_size=batch_size, # Client handles the splitting internally
            wait=True
        )

        print(f"[Qdrant Insert] Success: {len(points)} vectors inserted/uploaded")
        return True

    except Exception as e:
        print(f"[Qdrant Insert] ERROR: {e}")
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
        print(f"Error counting vectors: {e}")
        return -1