from typing import List, Tuple, Dict
import requests
import clustering_module
import threading
import qdrant_module
from compound_types import *

class Peer:
    def __init__(self, id, url):
        self.id = id
        self.url = url
        self.status = 'active'
        self.clusters = []

    def set_status(self, status):
        self.status = status

    def set_clusters(self, clusters: ListOfVectorsWithId):
        self.clusters = clusters
        
    def add_clusters(self, clusters: ListOfVectorsWithId):
        self.clusters.extend(clusters)
    
    def contains(self, cluster: VectorWithId):
        if cluster in [c[0] for c in self.clusters]:
            return True
        else:
            return False
    
    def send(self, vectors: ListOfVectorsComplete, id):
        print(f"\tsend function start: sending to peer: {self.id} with url {self.url}")
        payload = {
            'id': id,
            'content': vectors
        }
        print("\tpayload created")
        requests.post(
            f'{self.url}/receive_vectors_peer',
            json=payload
        )
        print("\trequest sent; return")
    
    def notify_clustering(self):
        print(f"notifying node {self.id}")
        requests.get(
            f'{self.url}/notify_clustering'
        )
        print(f"node {self.id} notified")
    
    def send_clusters(self, assignment: Dict[str, ListOfVectorsWithId], clusters: Dict[VectorId, Tuple[Vector, List[VectorId]]], meta_hnsw: clustering_module.MetaHNSW, request_id):
        to_send = {
            'id': request_id,
            'content': {
                'my_vectors': [vector_id for cluster_id, _ in assignment[self.id] for vector_id in clusters[cluster_id][1]],
                'peers_clusters': assignment,
                'meta_hnsw': meta_hnsw.to_serializable_dict()
            }
        }
        self.set_clusters(assignment[self.id])
        requests.post(f'{self.url}/set_clusters',
                    json=to_send)
        
    def query_peer(self, query: ListOfVectors, topk: int, request_id: int):
        payload = {
            'id': request_id,
            'query': query,
            'topk': topk
        }
        response = requests.post(
            f'{self.url}/query_peer',
            json=payload
        )
        return {}

class ServerApp:
    def __init__(self, id, url, qdrant_url, coordinator_url, replicas=3, collection_name="vectors", num_vectors_before_clustering=10_000, dimension=384):
        self.node_id = id
        self.url = url
        self.qdrant_url = qdrant_url
        self.coordinator_url = coordinator_url
        self.collection_name = collection_name
        self.dimension = dimension
        self.node_clusters = [] # ListOfVectorsWithId: Tuples of cluster_id, cluster_centroid
        self.status = 'bootstrap'
        self.num_vectors = 0
        self.replicas = replicas
        self.peers = []
        self.meta_hnsw = None
        self.vector_buffer = []
        self.additional_buffer = []
        self.num_vectors_before_clustering = num_vectors_before_clustering
        self.vector_buffer_lock = threading.RLock()
        self.additional_buffer_lock = threading.Lock()
        self.id_lock = threading.Lock()
        self.id_count = 0
    
    def i_am_coord(self):
        return self.coordinator_url == self.url
    
    def coordinator(self) -> Peer:
        for peer in self.peers:
            if peer.url == self.coordinator_url:
                return peer
    
    def clustering_notified(self):
        if self.status == 'bootstrap':
            self.status = 'clustering'
        else:
            raise("Error: status Undefined")
    
    def add_peers(self, peers: List[Tuple[str, str]]):
        '''
        Add peers to the node.
        
        Args:
            peers: List of Tuple(id, url)
        '''
        for id, url in peers:
            self.peers.append(Peer(id, url))
    
    """def add_peers_clusters(self, peers_vectors: Dict[str, ListOfVectorsWithId]):
        for id, vector_list in peers_vectors.items():
            for peer in self.peers:
                if peer.id == id:
                    peer.add_clusters(vector_list)"""
    
    def add_vectors(self, vectors: ListOfVectorsComplete, request_id):
        print("entering add_vectors function")
        with self.vector_buffer_lock:
            print(f"node {self.node_id}: {len(self.vector_buffer)}")
        if self.status == 'bootstrap':
            with self.vector_buffer_lock:
                print("entered first if branch: bootstrap")
                self.vector_buffer.extend(vectors)
                if self.i_am_coord() and len(self.vector_buffer) >= self.num_vectors_before_clustering:
                    print("start clustering")
                    self.status = 'clustering'
                    for peer in self.peers:
                        peer.notify_clustering()
                    print("all peers clustering notified")
                    clusters = clustering_module.get_clusters(self.vector_buffer) # Dict{VectorId: Tuple[Vector, List[VectorId]]}
                    print(f"called clustering, got clusters:{len(clusters.keys())}")
                    peers_with_me = self.peers[:]
                    peers_with_me.append(Peer(self.node_id, self.url))
                    print(f"type of peers_with_me {type(peers_with_me)}")
                    assignment = clustering_module.get_node_assignment(clusters, peers_with_me, self.replicas) # Dict{str: ListOfVectorsWithId}
                    print(f"called get_node_assignment, got assignment:{len(clusters.keys())}")
                    self.meta_hnsw = clustering_module.build_meta_hnsw(clusters, self.dimension)
                    print(f"TYPE META HNSW{type(self.meta_hnsw)}")
                    for peer in self.peers:
                        peer.send_clusters(assignment, clusters, self.meta_hnsw, request_id)
                    my_vectors = []
                    self.node_clusters = assignment[self.node_id]
                    for cluster_id, (centroid, vector_ids) in clusters.items():
                        if (cluster_id, centroid) in self.node_clusters:
                            my_vectors.extend(vector_ids)
                    self.adjust_after_clustering(my_vectors, request_id)
                    with self.additional_buffer_lock:
                        self.route_vectors_send(self.additional_buffer, request_id)
                        self.additional_buffer = []
                    #self.vector_buffer = []
        elif self.status == 'clustering':
            with self.additional_buffer_lock:
                self.additional_buffer.extend(vectors)
            if not self.i_am_coord():
                self.coordinator().send(vectors, request_id)
        elif self.status == 'clustered':
            qdrant_module.insert_vectors(self.qdrant_url, self.collection_name, vectors)
            self.num_vectors += len(vectors)
        else:
            raise("ERROR: status Undefined")
    
    def get_id(self):
        with self.id_lock:
            self.id_count += 1
            n = len(str(abs(len(self.peers)+1)))
            numeric_id = int(self.node_id.split("node")[-1])
            return int(f'{self.id_count}{numeric_id:0{n}d}')
        
    """
    È la porta d'ingresso pubblica per i client. Quando un utente vuole aggiungere un nuovo "libro" (vettore), chiama questa funzione.
    """
    def add_vectors_client(self, vectors: ListOfVectorsWithPayload, request_id):
        print("add_vectors_client_start")
        vectors_with_id = [(vector_content, self.get_id(), vector_payload) for vector_content, vector_payload  in vectors]
        if self.status == 'bootstrap':
            print("entered first if branch: bootstrap")
            for peer in self.peers:
                print(f"sending peer {peer.id} the vectors")
                peer.send(vectors_with_id, request_id)
            print("vectors sent to all peers")
            self.add_vectors(vectors_with_id, request_id)
            print("vectors added to buffer")
        elif self.status == 'clustered':
            self.route_vectors_send(vectors_with_id, request_id)
        elif self.status == 'clustering':
            with self.additional_buffer_lock:
                self.additional_buffer.extend(vectors_with_id)
            if not self.i_am_coord():
                self.coordinator().send(vectors_with_id, request_id)
        else:
            raise("Error: Undefined status")
    
    def set_clusters(self, assignment, request_id): # assignment is of type Dict['my_vectors': List[VectorId], 'peers_clusters': Dict[str, ListOfVectorsWithId], 'meta_hnsw': MetaHNSW]
        for peer in self.peers:
            peer.set_clusters(assignment['peers_clusters'][peer.id])
        self.node_clusters = assignment['peers_clusters'][self.node_id]
        self.meta_hnsw = assignment['meta_hnsw']
        self.adjust_after_clustering(assignment['my_vectors'], request_id)

    """
    Fa pulizia. Dopo a clustering finito, il peer guarda nella suo vector_buffer e salva solo i libri che gli sono stati assegnati, buttando il resto.
    """
    def adjust_after_clustering(self, my_vector_ids: List[VectorId], request_id):
        with self.vector_buffer_lock:
            self.status = 'clustered'
            to_save = []
            for vector in self.vector_buffer:
                if vector[1] in my_vector_ids:
                    to_save.append(vector)
            print(f"len of to_save = {len(to_save)} - len of vector_buffer = {len(self.vector_buffer)} - len of my_vector_ids = {len(my_vector_ids)}")
            self.add_vectors(to_save, request_id)
            self.vector_buffer = []

    """
    Input: Un vettore, quanti nodi trovare (k).
    Output: Una lista dei k nodi più rilevanti per quel vettore.
    """
    def route_vector(self, vector: Vector, k: int):
        return clustering_module.find(self.meta_hnsw, vector, k)

    """
    Prende una lista intera di vettori da spedire. Per ogni singolo vettore:
     - Usa route_vector(k=1) per chiedere: "Qual è il nodo migliore per questo vettore?".
     - Mette il vettore nel batch destinato a quel nodo (Peer).
     - Se non trova un proprietario chiaro o se la replica fallisce, tiene il vettore per sé (to_me).
    """
    def route_vectors_send(self, vectors: ListOfVectorsComplete, request_id):
        assigned_vectors = [[] for _ in range(len(self.peers))]
        to_me = []
        for vector_content, vector_id, vector_payload in vectors:
            vector = (vector_content, vector_id, vector_payload)
            top_1 = self.route_vector(vector_content, 1)[0]
            found = 0
            for index, peer in enumerate(self.peers):
                if peer.contains(top_1):
                    assigned_vectors[index].append(vector)
                    found += 1
            if found != self.replicas:
                to_me.append(vector)
        for index, peer in enumerate(self.peers):
            if len(assigned_vectors[index]) > 0:
                peer.send(assigned_vectors[index], request_id)
        if len(to_me) > 0:
            self.add_vectors(to_me, request_id)

    """
    Cerca un vettore/i solo nel suo database locale (il suo Qdrant).
    """
    def query_me(self, query: ListOfVectors, topk: int):
        return qdrant_module.query_vectors(self.qdrant_url, self.collection_name, query, topk)
    
    """
    Cerca un vettore/i sia nel suo database locale che in quello dei peer.
        -usa route_vector per decidere quali peer interrogare
        -chiama query_peer su quei peer
    """
    def query(self, query: ListOfVectors, topk: int, request_id: int):
        response = []
        to_query_peer = [[] for _ in range(len(self.peers))]
        to_query_me = []
        for vector in query:
            top_3 = self.route_vector(vector, 3)
            for index, peer in enumerate(self.peers):
                for result in top_3:
                    if peer.contains(result):
                        to_query_peer[index].append(vector)
            for result in top_3:
                if result in [c[0] for c in self.node_clusters]:
                    to_query_me.append(vector)
        for index, peer in enumerate(self.peers):
            response.extend(peer.query_peer(to_query_peer[index], topk, request_id))
        response.extend(self.query_me(to_query_me, topk))
        #response = list(set(response))

        return response

    """
    Usata dal coordinatore per notificare tutti i peer che ci sono abbastanza vettori dunque è ora di fare il clustering.
    """
    def notify_clustering(self):
        self.status = 'clustering'
        #for peer in self.peers:
        #    peer.notify_clustering()