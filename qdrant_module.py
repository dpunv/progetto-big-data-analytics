import requests
#import traceback

def create_collection(url, collection_name, vector_size: int, distance: str = "Cosine"):
    try:
        response = requests.get(
            f"{url}/collections/{collection_name}",
            timeout=10
        )
        if response.status_code == 200:
            return True
        
        payload = {
            "vectors": {
                "size": vector_size,
                "distance": distance
            }
        }
        response = requests.put(
            f"{url}/collections/{collection_name}",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error creating collection: {e} - {e.response.text if e.response else 'No response'}")
        return False

"""
- Prende la tua lista di query (sì, puoi fare più ricerche in un colpo solo!).
- Per ogni query, chiede i topk risultati più vicini.
- Invia la richiesta con POST (che significa "ehi, dammi questi dati").
- Ti restituisce la lista dei risultati trovati.
"""
def query_vectors(url, collection, query, topk):
    """Query vectors from Qdrant collection."""
    try:
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
        
        print(f"[Qdrant Query] Status code: {response.status_code}")

        if response.status_code != 200:
            print(f"[Qdrant Query] ERRORE: {response.text}")
            return None

        results = response.json()['result']
        print(f"[Qdrant Query] Successo: trovati {len(results)} risultati")
        #print(results)
        return results

    except requests.exceptions.Timeout:
        print("[Qdrant Query] ERRORE: Timeout della richiesta")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[Qdrant Query] ERRORE di connessione: {e}")
        return None
    except Exception as e:
        print(f"[Qdrant Query] ERRORE generico: {e}")
        return None

def insert_vectors_batch(url, collection, vectors):
    """Insert vectors into Qdrant collection."""
    try:
        points = [
            {
                "id": vector_id,
                "vector": vector_content,
                "payload": {
                    "string": vector_payload
                }
            }
            for vector_content, vector_id, vector_payload in vectors
        ]
        response = requests.put(
            f'{url}/collections/{collection}/points',
            params={"wait": "true"},
            json={"points": points},
            timeout=10
        )

        print(response)
        print(response.status_code)

        print(f"[Qdrant Insert] Status code: {response.status_code}")

        if response.status_code == 200:
            print(f"[Qdrant Insert] Successo: {len(points)} vettori inseriti")
            return True
        else:
            print(f"[Qdrant Insert] ERRORE: {response.text}")
            #traceback.print_stack()
            return False

    except requests.exceptions.Timeout:
        print("[Qdrant Insert] ERRORE: Timeout della richiesta")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[Qdrant Insert] ERRORE di connessione: {e}")
        return False
    except Exception as e:
        print(f"[Qdrant Insert] ERRORE generico: {e}")
        return False


"""
Ogni dato che inserisci è un "punto" e deve avere:
    - Un ID (un nome unico, es. "documento_abc").
    - Un vettore (i numeri che ne rappresentano il significato, es. [0.1, 0.2, 0.3]).
    - Un payload (dati extra che vuoi salvare insieme, es. il testo originale, un titolo, un link).
La funzione prende la tua lista di dati, la formatta nel modo corretto per Qdrant e la invia con un comando PUT
"""
def insert_vectors(url, collection, vectors, batch_size=256):
    num_batches = len(vectors) // batch_size
    for batch_id in range(num_batches):
        insert_vectors_batch(url, collection, vectors[batch_size * batch_id:batch_size * (batch_id+1)])
    if len(vectors)% num_batches != 0:
        insert_vectors_batch(url, collection, vectors[batch_size * num_batches:])