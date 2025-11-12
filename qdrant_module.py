import requests


def query_vectors(url, collection, query, topk):
    """Query vectors from Qdrant collection."""
    payload = [{
        "vector": query_vector,
        "limit": topk,
        "with_payload": True,
        "with_vector": True
    } for query_vector in query]
    response = requests.post(
        f'{url}/collections/{collection}/points/search/batch',
        json={"searches": payload},
        timeout=10
    )
    results = response.json()['result']
    return results


def insert_vectors(url, collection, vectors):
    """Insert vectors into Qdrant collection."""
    points = [
        {
            "id": vector_id,
            "vector": vector_content,
            "payload": {
                "string": vector_payload
            }
        }
        for vector_id, vector_content, vector_payload in vectors
    ]
    response = requests.put(
        f'{url}/collections/{collection}/points',
        params={"wait": "true"},
        json={"points": points},
        timeout=10
    )

    if response.status_code == 200:
        return True
    else:
        return False

    