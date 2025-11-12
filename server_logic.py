from typing import List, Tuple
import requests
import clustering_module
import threading
import qdrant_module

ListOfVectors = List[Tuple[str, List[float]]]

class Peer:
    def __init__(self, id, url):
        self.id = id
        self.url = url
        self.status = 'active'
        self.clusters = []

    def set_status(self, status):
        self.status = status

    def set_clusters(self, clusters: ListOfVectors):
        self.clusters = clusters
        
    def add_clusters(self, clusters: ListOfVectors):
        self.clusters.extend(clusters)
    
    def contains(self, cluster: List[float]):
        return cluster in self.clusters
    
    def send(self, vectors: ListOfVectors, id):
        payload = {
            'id': id,
            'vectors': vectors
        }
        requests.post(
            f'{self.url}/receive_vectors_peer',
            json=payload
        )
    
    def notify_clustering(self):
        requests.get(
            f'{self.url}/notify_clustering'
        )
    
    def send_clusters(self, assignment, clusters, meta_hnsw):
        to_send = {
            'my_vectors': [vector_id for cluster_id, _ in assignment[self.id] for vector_id in clusters[cluster_id].second],
            'peers_clusters': assignment,
            'meta_hnsw': meta_hnsw
        }
        self.set_clusters(assignment[self.id])
        requests.post(f'{self.url}/set_clusters',
                    json=to_send)
        
    def query_peer(self, query: List[List[float]], topk: int, request_id: int):
        payload = {
            'id': request_id,
            'query': query,
            'topk': topk
        }
        response = requests.post(
            f'{self.url}/query_peer',
            json=payload
        )
        return response.json()['results']

class ServerApp:
    def __init__(self, id, url, qdrant_url, coordinator_url, replicas=3, collection_name="vectors", num_vectors_before_clustering=10_000):
        self.node_id = id
        self.url = url
        self.qdrant_url = qdrant_url
        self.coordinator_url = coordinator_url
        self.collection_name = collection_name
        self.node_clusters = [] # Tuples of cluster_id, cluster_centroid
        self.status = 'bootstrap'
        self.num_vectors = 0
        self.replicas = replicas
        self.peers = []
        self.meta_hnsw = None
        self.vector_buffer = []
        self.additional_buffer = []
        self.num_vectors_before_clustering = num_vectors_before_clustering
        self.additional_buffer_lock = threading.Lock()
    
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
    
    def add_peers_clusters(self, peers_vectors: ListOfVectors):
        '''
        Add clusters to peers.
        
        Args:
            peers_vectors: List of Tuple(id, List of clusters)
            
        '''
        for id, vector_list in peers_vectors:
            for peer in self.peers:
                if peer.id == id:
                    peer.add_clusters(vector_list)
    
    def add_vectors(self, vectors: ListOfVectors):
        '''
        Add vectors to the Qdrant database.
        
        Args:
            vectors: List of vectors to add
        '''
        if self.status == 'bootstrap':
            self.vector_buffer.extend(vectors)
            if self.i_am_coord() and len(self.vector_buffer) >= self.num_vectors_before_clustering:
                self.status = 'clustering'
                for peer in self.peers:
                    peer.notify_clustering()
                clusters = clustering_module.get_clusters(self.vector_buffer) # Dict{id: Tuple[List[float], List[str]]}
                assignment = clustering_module.get_assignment(clusters, self.peers[:].extend(Peer(self.node_id, self.url))) # Dict{id: List[Tuple[str, List[float]]]}
                self.meta_hnsw = clustering_module.build_meta_hnsw(clusters)
                for peer in self.peers:
                    peer.send_clusters(assignment, clusters, self.meta_hnsw)
                my_vectors = []
                self.node_clusters = assignment[self.node_id]
                for cluster_id, (centroid, vector_ids) in clusters.items():
                    if (cluster_id, centroid) in self.node_clusters:
                        my_vectors.extend(vector_ids)
                self.adjust_after_clustering(my_vectors)
                with self.additional_buffer_lock:
                    self.route_vectors_send(self.additional_buffer, -1) #TODO change request id
                    self.additional_buffer = []
                self.vector_buffer = []
        elif self.status == 'clustering':
            with self.additional_buffer_lock:
                self.additional_buffer.extend(vectors)
            if not self.i_am_coord():
                self.coordinator().send(vectors)
        elif self.status == 'clustered':
            qdrant_module.insert_vectors(self.qdrant_url, self.collection_name, vectors)
            self.num_vectors += len(vectors)
        else:
            raise("ERROR: status Undefined")

    def add_vectors_client(self, vectors: ListOfVectors, request_id):
        if self.status == 'bootstrap':
            for peer in self.peers:
                peer.send(vectors, request_id)
            self.add_vectors(vectors, True)
        elif self.status == 'clustered':
            self.route_vectors_send(vectors, request_id)
        elif self.status == 'clustering':
            with self.additional_buffer_lock:
                self.additional_buffer.extend(vectors)
            if self.i_am_coord():
                self.coordinator().send(vectors, request_id)
        else:
            raise("Error: Undefined status")
    
    def set_clusters(self, assignment): # assignment = Dict{'my_vectors': List[str], 'peers_clusters': Dict{peer_id: List[Tuple[str, List[float]]]}, 'meta_hnsw': meta_hnsw}
        for peer in self.peers:
            peer.set_clusters(assignment['peers_clusters'][peer.id])
        self.node_clusters = assignment['peers_clusters'][self.node_id]
        self.meta_hnsw = assignment['meta_hnsw']
        self.adjust_after_clustering(assignment['my_vectors'])

    def adjust_after_clustering(self, my_vector_ids):
        self.status = 'clustered'
        to_save = []
        for vector in self.vector_buffer:
            if vector.first in my_vector_ids:
                to_save.append(vector)
        self.add_vectors(to_save)
        self.vector_buffer = []

    def route_vectors(self, vector: Tuple[str, List[float]], k: int):
        return self.meta_hnsw.find(vector, k) # trovo il cluster più vicino a vector, ricerca top 1

    def route_vectors_send(self, vectors: ListOfVectors, request_id):
        assigned_vectors = [[] for _ in range(self.peers)]
        to_me = []
        for vector in vectors:
            top_1 = self.route_vector(vector, 1)[0]
            found = 0
            for index, peer in enumerate(self.peers):
                if peer.contains(top_1):
                    assigned_vectors[index].append(vector)
                    found += 1
            if found != self.replicas:
                to_me.append(vector, False)
        for index, peer in enumerate(self.peers):
            peer.send(assigned_vectors[index], request_id)
        if len(to_me) > 0:
            self.add_vectors(to_me)

    def query_me(self, query: List[List[float]], topk: int):
        return qdrant_module.query_vectors(self.qdrant_url, self.collection_name, query, topk)
    
    def query(self, query: List[List[float]], topk: int, request_id: int):
        response = []
        to_query_peer = [[] for _ in range(self.peers)]
        to_query_me = []
        for vector in query:
            top_3 = self.route_vectors(vector, 3)
            for index, peer in enumerate(self.peers):
                for result in top_3:
                    if peer.contains(result):
                        to_query_peer[index].append(vector)
            for result in top_3:
                if result in self.node_clusters:
                    to_query_me.append(vector)
        for index, peer in enumerate(self.peers):
            response.extend(peer.query_peer(to_query_peer[index], topk, request_id))
        response.extend(self.query_me(to_query_me, topk))
        response = list(set(response))
        return response

    def notify_clustering(self):
        for peer in self.peers:
            peer.notify_clustering()