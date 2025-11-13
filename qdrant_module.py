import requests

"""
- Prende la tua lista di query (sì, puoi fare più ricerche in un colpo solo!).
- Per ogni query, chiede i topk risultati più vicini.
- Invia la richiesta con POST (che significa "ehi, dammi questi dati").
- Ti restituisce la lista dei risultati trovati.
"""
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


"""
Ogni dato che inserisci è un "punto" e deve avere:
    - Un ID (un nome unico, es. "documento_abc").
    - Un vettore (i numeri che ne rappresentano il significato, es. [0.1, 0.2, 0.3]).
    - Un payload (dati extra che vuoi salvare insieme, es. il testo originale, un titolo, un link).
La funzione prende la tua lista di dati, la formatta nel modo corretto per Qdrant e la invia con un comando PUT
"""
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

    