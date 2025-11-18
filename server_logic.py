from typing import List, Tuple, Dict
import requests
import clustering_module
import threading
import qdrant_module
from compound_types import *
import utils
import grpc
import p2p_pb2
import p2p_pb2_grpc
import pickle

class Peer:
    def __init__(self, id, url, grpc_url):
        self.id = id
        self.url = url
        self.grpc_url = grpc_url
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
    
    def send(self, vectors: ListOfVectorsComplete, status: str, id):
        print(f"\tsend (gRPC) to peer: {self.id} at {self.grpc_url}")
        
        # Convert Python data to Proto objects
        proto_vectors = [
            p2p_pb2.VectorPoint(vector=v[0], id=v[1], payload=v[2]) 
            for v in vectors
        ]
        
        req = p2p_pb2.AddVectorsRequest(
            req_id=id,
            content=proto_vectors,
            type=status
        )

        with grpc.insecure_channel(self.grpc_url) as channel:
            stub = p2p_pb2_grpc.P2PNodeStub(channel)
            stub.ReceiveVectors(req)

    def notify_clustering(self):
        with grpc.insecure_channel(self.grpc_url) as channel:
            stub = p2p_pb2_grpc.P2PNodeStub(channel)
            stub.NotifyClustering(p2p_pb2.Empty())

    def send_clusters(self, assignment, clusters, meta_hnsw, request_id):
        # Prepare data as before
        to_send_dict = {
            'my_vectors': [vector_id for cluster_id, _ in assignment[self.id] for vector_id in clusters[cluster_id][1]],
            'peers_clusters': assignment,
            'meta_hnsw': meta_hnsw.to_serializable_dict()
        }
        
        # Serialize using Pickle for binary transfer over gRPC
        data_bytes = pickle.dumps(to_send_dict)
        
        req = p2p_pb2.SetClustersRequest(req_id=request_id, content_pickle=data_bytes)
        
        with grpc.insecure_channel(self.grpc_url) as channel:
            stub = p2p_pb2_grpc.P2PNodeStub(channel)
            stub.SetClusters(req)

    def query_peer(self, query: ListOfVectors, topk: int, request_id: int):
        proto_query = [p2p_pb2.VectorList(values=v) for v in query]
        req = p2p_pb2.QueryRequest(req_id=request_id, query_vectors=proto_query, topk=topk)
        
        with grpc.insecure_channel(self.grpc_url) as channel:
            stub = p2p_pb2_grpc.P2PNodeStub(channel)
            response = stub.QueryPeer(req)
        
        # Convert Proto response back to Python Dict structure expected by ServerApp.query
        results = []
        for sp in response.results:
            results.append({
                'id': sp.id,
                'score': sp.score,
                'payload': {
                    'string': sp.payload.payload,
                    'vector': list(sp.payload.vector)
                }
            })
        return results

    def get_count(self):
        return requests.get(
            f'{self.url}/count_peer'
        ).json()

class ServerApp:
    def __init__(self, id, url, qdrant_url, grpc_url, coordinator_url, replicas=3, collection_name="vectors", num_vectors_before_clustering=10_000, dimension=384):
        self.node_id = id
        self.url = url
        self.qdrant_url = qdrant_url
        self.grpc_url = grpc_url
        self.coordinator_url = coordinator_url
        self.collection_name = collection_name
        self.dimension = dimension
        self.node_clusters = [] # ListOfVectorsWithId: Tuples of cluster_id, cluster_centroid
        self.status = 'bootstrap'
        self.replicas = replicas
        self.peers = []
        self.meta_hnsw = None
        self.vector_buffer = []
        self.additional_buffer = []
        self.num_vectors_before_clustering = num_vectors_before_clustering
        self.vector_buffer_lock = threading.RLock()
        self.additional_buffer_lock = threading.Lock()
        self.clustering_lock = threading.Lock()
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
    
    def add_peers(self, peers: List[Tuple[str, str, str]]): # Added 3rd tuple element
        for id, http_url, grpc_url in peers:
            self.peers.append(Peer(id, http_url, grpc_url))
    
    def start_clustering_thread(self, request_id):
        print("start clustering")
        self.status = 'clustering'
        for peer in self.peers:
            print(f"notify {peer.id}")
            peer.notify_clustering()
        clusters = clustering_module.get_clusters(self.vector_buffer) # Dict{VectorId: Tuple[Vector, List[VectorId]]}
        peers_with_me = self.peers[:]
        peers_with_me.append(Peer(self.node_id, self.url, self.grpc_url))
        assignment = clustering_module.get_node_assignment(clusters, peers_with_me, self.replicas) # Dict{str: ListOfVectorsWithId}
        self.meta_hnsw = clustering_module.build_meta_hnsw(clusters, self.dimension)
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

    def add_vectors(self, vectors: ListOfVectorsComplete, status_of_sender: str, request_id):
        if status_of_sender == 'clustered':
            qdrant_module.insert_vectors(self.qdrant_url, self.collection_name, vectors)
        else:
            if self.status == 'bootstrap':
                with self.vector_buffer_lock:
                    self.vector_buffer.extend(vectors)
                if self.i_am_coord() and len(self.vector_buffer) >= self.num_vectors_before_clustering:
                    with self.clustering_lock:
                        if not self.status == 'clustered':
                            cluster_thread = threading.Thread(
                                target=self.start_clustering_thread,
                                args=(request_id,)
                            )
                            cluster_thread.start()
                            return
            elif self.status == 'clustering':
                if not self.i_am_coord():
                    with self.additional_buffer_lock:
                        self.additional_buffer.extend(vectors)
                    if status_of_sender == 'bootstrap':
                        return
                    self.coordinator().send(vectors, self.status, request_id)
            elif self.status == 'clustered':
                if self.i_am_coord() and (status_of_sender == 'clustering' or status_of_sender == 'bootstrap'):
                    self.route_vectors_send(vectors, request_id)
                else:
                    qdrant_module.insert_vectors(self.qdrant_url, self.collection_name, vectors)
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
                peer.send(vectors_with_id, self.status, request_id)
            print("vectors sent to all peers")
            self.add_vectors(vectors_with_id, self.status, request_id)
            print("vectors added to buffer")
        elif self.status == 'clustered':
            self.route_vectors_send(vectors_with_id, request_id)
        elif self.status == 'clustering':
            with self.additional_buffer_lock:
                self.additional_buffer.extend(vectors_with_id)
            if not self.i_am_coord():
                self.coordinator().send(vectors_with_id, self.status, request_id)
        else:
            raise("Error: Undefined status")
    
    def set_clusters(self, assignment, request_id): # assignment is of type Dict['my_vectors': List[VectorId], 'peers_clusters': Dict[str, ListOfVectorsWithId], 'meta_hnsw': MetaHNSW]
        for peer in self.peers:
            peer.set_clusters(assignment['peers_clusters'][peer.id])
        self.node_clusters = assignment['peers_clusters'][self.node_id]
        self.meta_hnsw = clustering_module.MetaHNSW.from_serializable_dict(assignment['meta_hnsw'])
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
            self.add_vectors(to_save, self.status, request_id)
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
                peer.send(assigned_vectors[index], self.status, request_id)
        if len(to_me) > 0:
            self.add_vectors(to_me, self.status, request_id)

    def linear_search(self, query: ListOfVectors, topk: int):
        to_return = []
        for qv in query:
            distances_calcs = [(vector, utils.cosine_similarity(vector[0], qv)) for vector in self.vector_buffer]
            distances_calcs.sort(key=lambda x: x[1], reverse=True)
            to_return.extend([{'id': v[1], 'score': d, 'payload': {'string': v[2], 'vector': v[0]}} for v, d in distances_calcs[:topk]])
        return to_return

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
        if(self.meta_hnsw is None):
            return self.linear_search(query, topk)    
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
        #res = [j for i in response for j in i]
        seen = set()
        unique_response = []
        for item in response:
            item_id = item.get('id')
            if item_id not in seen:
                seen.add(item_id)
                unique_response.append(item)
        response = sorted(unique_response, key=lambda x: x['score'])
        return response

    """
    Usata dal coordinatore per notificare tutti i peer che ci sono abbastanza vettori dunque è ora di fare il clustering.
    """
    def notify_clustering(self):
        self.status = 'clustering'
    
    def get_count(self):
        if self.status == 'bootstrap':
            return 0
        elif self.status == 'clustering':
            return 0
        elif self.status == 'clustered':
            return qdrant_module.count(self.qdrant_url, self.collection_name)
        else:
            raise("Error: status Undefined")
    
    def get_count_client(self):
        res = {}
        for peer in self.peers:
            res[peer.id] = peer.get_count()
        res[self.node_id] = self.get_count()
        return res