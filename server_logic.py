from typing import List, Tuple, Dict
import requests
import clustering_module
import threading
import metrics
import qdrant_module
from compound_types import *
import utils
import grpc
import p2p_pb2
import p2p_pb2_grpc
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

GRPC_OPTIONS = [
    ('grpc.max_send_message_length', 512 * 1024 * 1024),
    ('grpc.max_receive_message_length', 512 * 1024 * 1024)
]

class Peer:
    def __init__(self, id, url, grpc_url):
        self.id = id
        self.url = url
        self.grpc_url = grpc_url
        self.status = 'active'
        self.clusters = []
        self.cluster_ids_set = set()
        
        # Initialize gRPC channel and stub once
        self.channel = grpc.insecure_channel(self.grpc_url, options=GRPC_OPTIONS)
        self.stub = p2p_pb2_grpc.P2PNodeStub(self.channel)

    def set_status(self, status):
        self.status = status

    def set_clusters(self, clusters: ListOfVectorsWithId):
        self.clusters = clusters
        self.cluster_ids_set = {c[0] for c in clusters}
        
    def add_clusters(self, clusters: ListOfVectorsWithId):
        self.clusters.extend(clusters)
        self.cluster_ids_set.update(c[0] for c in clusters)
    
    def contains(self, cluster: VectorWithId):
        return cluster in self.cluster_ids_set
    
    def send(self, vectors: ListOfVectorsComplete, status: str, id):
        logger.debug(f"--> [Peer Send] Sending {len(vectors)} vectors to Peer {self.id} ({self.grpc_url}) | Status: {status}")
        
        proto_vectors = [
            p2p_pb2.VectorPoint(vector=v[0], id=v[1], payload=v[2]) 
            for v in vectors
        ]
        
        req = p2p_pb2.AddVectorsRequest(
            req_id=id,
            content=proto_vectors,
            type=status
        )

        try:
            self.stub.ReceiveVectors(req)
            logger.debug(f"--> [Peer Send] Success: Sent vectors to {self.id}")
        except grpc.RpcError as e:
            logger.error(f"--> [Peer Send] FAILED to {self.id}: {e}")

    def send_clusters(self, assignment, clusters, meta_hnsw, request_id):
        logger.debug(f"--> [Peer SendClusters] Sending cluster assignment to Peer {self.id}")
        to_send_dict = {
            'my_vectors': [vector_complete for cluster_id, _ in assignment[self.id] for vector_complete in clusters[cluster_id][1]],
            'peers_clusters': assignment,
            'meta_hnsw': meta_hnsw.to_serializable_dict()
        }

        self.set_clusters(assignment[self.id])
        
        data_bytes = pickle.dumps(to_send_dict)
        
        req = p2p_pb2.SetClustersRequest(req_id=request_id, content_pickle=data_bytes)
        
        try:
            self.stub.SetClusters(req)
            logger.debug(f"--> [Peer SendClusters] Success: Sent clusters to {self.id}")
        except grpc.RpcError as e:
            logger.error(f"--> [Peer SendClusters] FAILED to {self.id}: {e}")

    def query_peer(self, query: ListOfVectors, topk: int, request_id: int):
        logger.debug(f"--> [Peer Query] Querying Peer {self.id} for {len(query)} vectors")
        proto_query = [p2p_pb2.VectorList(values=v) for v in query]
        req = p2p_pb2.QueryRequest(req_id=request_id, query_vectors=proto_query, topk=topk)
        
        try:
            response = self.stub.QueryPeer(req)
            
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
            logger.debug(f"--> [Peer Query] Success: Peer {self.id} returned {len(results)} results")
            return results
        except grpc.RpcError as e:
            logger.error(f"--> [Peer Query] FAILED to {self.id}: {e}")
            return []

    def get_count(self):
        try:
            return requests.get(f'{self.url}/count_peer').json()
        except Exception as e:
            logger.error(f"Error getting count from {self.id}: {e}")
            return {}
            
    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()

class ServerApp:
    def __init__(self, id, url, qdrant_url, grpc_url, coordinator_url, replicas=3, collection_name="vectors", num_vectors_before_clustering=10_000, dimension=384, batch_size=256, batch_size_retry=64):
        self.node_id = id
        self.url = url
        self.qdrant_url = qdrant_url
        self.grpc_url = grpc_url
        self.coordinator_url = coordinator_url
        self.collection_name = collection_name
        self.dimension = dimension
        self.node_clusters = [] 
        self.status = 'bootstrap'
        self.replicas = replicas
        self.peers = []
        self.meta_hnsw = None
        self.vector_buffer = []
        self.additional_buffer = []
        self.num_vectors_before_clustering = num_vectors_before_clustering
        self.vector_buffer_lock = threading.RLock()
        self.additional_buffer_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.clustering_lock = threading.Lock()
        self.id_lock = threading.Lock()
        self.id_count = 0
        self.batch_size = batch_size
        self.batch_size_retry = batch_size_retry
        # Thread pool for parallel peer communication
        self.peer_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='PeerComm')
        # NEW: Mapping cluster_id -> collection_name for multi-collection support
        self.cluster_collections = {}  # {cluster_id: "collection_name"}
        logger.info(f"ServerApp initialized: ID={id}, URL={url}, Coords={coordinator_url}")

    def _run_parallel_tasks(self, tasks):
        """
        Executes a list of tasks in parallel using the peer executor.
        tasks: List of tuples (function, *args)
        Returns: List of results from the functions.
        """
        if not tasks:
            return []
        
        futures = [self.peer_executor.submit(func, *args) for func, args in tasks]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results
    
    def i_am_coord(self):
        return self.coordinator_url == self.url
    
    def coordinator(self) -> Peer:
        for peer in self.peers:
            if peer.url == self.coordinator_url:
                return peer
        logger.warning("Coordinator not found in peers list!")
        return None
    
    def add_peers(self, peers: List[Tuple[str, str, str]]):
        with metrics.REQUEST_LATENCY.labels(operation='add_peers').time():
            logger.info(f"[Add Peers] Adding {len(peers)} peers: {peers}")
            for id, http_url, grpc_url in peers:
                self.peers.append(Peer(id, http_url, grpc_url))
    
    def start_clustering_thread(self, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='clustering').time():
            logger.info(f"*** START CLUSTERING THREAD (ReqID: {request_id}) ***")
            with self.vector_buffer_lock:
                logger.info(f"Clustering: Converting buffer of {len(self.vector_buffer)} vectors.")
                vectors_to_cluster = self.vector_buffer[:]
                # self.vector_buffer = [] # Do not clear buffer yet, wait for status change
            
            peers_with_me = self.peers[:]
            peers_with_me.append(Peer(self.node_id, self.url, self.grpc_url))
            
            logger.info("Clustering: Calculating node assignments...")
            clusters = clustering_module.get_clusters(vectors_to_cluster)
            assignment = clustering_module.get_node_assignment(clusters, peers_with_me, self.replicas)
            self.meta_hnsw = clustering_module.build_meta_hnsw(clusters, self.dimension)
            
            [logger.info(f"Clusters and vectors count per node: {node_id}: Clusters: {len(clusters_in_node)} - Vectors: {sum([len(clusters[cluster[0]][1]) for cluster in clusters_in_node])}") for node_id, clusters_in_node in assignment.items()]

            logger.info("Clustering: Broadcasting assignments to peers...")
            
            # Parallel broadcasting of cluster assignments
            def broadcast_to_peer(peer):
                try:
                    peer.send_clusters(assignment, clusters, self.meta_hnsw, request_id)
                    return (peer.id, True)
                except Exception as e:
                    logger.error(f"[Clustering] Failed to broadcast to {peer.id}: {e}")
                    return (peer.id, False)
            
            # Submit all broadcast tasks
            tasks = [(broadcast_to_peer, (peer,)) for peer in self.peers]
            results = self._run_parallel_tasks(tasks)
            
            # Check results
            failed_broadcasts = [peer_id for peer_id, success in results if not success]
            
            if failed_broadcasts:
                logger.error(f"[Clustering] Failed to broadcast clusters to: {failed_broadcasts}")
            else:
                logger.info("[Clustering] All cluster assignments broadcasted successfully")
            
            self.node_clusters = assignment[self.node_id]
            for cluster_id, _ in assignment[self.node_id]:
                collection_name = f"{self.collection_name}_cluster_{cluster_id}"
                self.cluster_collections[cluster_id] = collection_name
                logger.info(f"Creating collection '{collection_name}' for cluster {cluster_id}")
                qdrant_module.create_collection(self.qdrant_url, collection_name, self.dimension)
            
            with self.vector_buffer_lock:
                with self.status_lock:
                    self.status = 'clustered'
                self.vector_buffer = []
                logger.info(f"*** Status changed to 'clustered'. Processing local assignment ({len(assignment[self.node_id])} clusters)...")
                
                local_vectors = [vector_complete for cluster_id, _ in assignment[self.node_id] for vector_complete in clusters[cluster_id][1]]
                logger.info(f"Clustering: Inserting {len(local_vectors)} assigned vectors to local Qdrant.")
                self.add_vectors(local_vectors, self.status, request_id)
                
            with self.additional_buffer_lock:
                logger.info(f"Clustering: Processing additional buffer ({len(self.additional_buffer)} vectors)...")
                self.route_vectors_send(self.additional_buffer, request_id)
                self.additional_buffer = []
                
            logger.info("*** CLUSTERING FINISHED ***")

    def add_vectors(self, vectors: ListOfVectorsComplete, status_of_sender: str, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='add_vectors').time():
            logger.debug(f"[Add Vectors] Processing {len(vectors)} vectors. Sender Status: {status_of_sender}. Current Node Status: {self.status}")
            
            if status_of_sender == 'clustered':
                logger.debug(f"[Add Vectors] Direct Insert: Storing {len(vectors)} vectors in local Qdrant (Sender is clustered).")
                
                # Raggruppa vettori per cluster_id
                vectors_by_cluster = {}
                for vector_content, vector_id, vector_payload in vectors:
                    if "|cluster_id:" in vector_payload:
                        parts = vector_payload.rsplit("|cluster_id:", 1)
                        original_payload = parts[0]
                        cluster_id = int(parts[1])
            
                        if cluster_id not in vectors_by_cluster:
                            vectors_by_cluster[cluster_id] = []
                            vectors_by_cluster[cluster_id].append((vector_content, vector_id, original_payload))
                    else:
                        # Fallback per vettori senza cluster_id
                        if 'default' not in vectors_by_cluster:
                            vectors_by_cluster['default'] = []
                            vectors_by_cluster['default'].append((vector_content, vector_id, vector_payload))
    
                # Inserisci ogni gruppo nella sua collection
                for cluster_id, cluster_vectors in vectors_by_cluster.items():
                    if cluster_id == 'default':
                        collection_name = self.collection_name
                    else:
                        collection_name = self.cluster_collections.get(cluster_id, f"{self.collection_name}_cluster_{cluster_id}")
        
                    logger.debug(f"[Add Vectors] Inserting {len(cluster_vectors)} vectors into '{collection_name}' (cluster {cluster_id})")
                    qdrant_module.insert_vectors(self.qdrant_url, collection_name, cluster_vectors, self.batch_size_retry, self.batch_size)

            else:
                if not self.i_am_coord():
                    logger.error("Error: Received bootstrap/clustering vectors but I am not coordinator.")
                    raise Exception('Error: status Undefined')
                else:
                    if self.status == 'bootstrap':
                        with self.vector_buffer_lock:
                            self.vector_buffer.extend(vectors)
                            buffer_len = len(self.vector_buffer)
                        
                        logger.debug(f"[Buffer Update] Bootstrap mode. Buffer size: {buffer_len}/{self.num_vectors_before_clustering}")
                        
                        if buffer_len >= self.num_vectors_before_clustering:
                            with self.clustering_lock:
                                if self.status == 'bootstrap':
                                    self.status = 'clustering'
                                    logger.info("THRESHOLD REACHED: Triggering Clustering.")
                                    cluster_thread = threading.Thread(
                                        target=self.start_clustering_thread,
                                        args=(request_id,)
                                    )
                                    cluster_thread.start()
                                    return
                    elif self.status == 'clustering':
                        with self.additional_buffer_lock:
                            self.additional_buffer.extend(vectors)
                        logger.debug(f"[Buffer Update] Clustering in progress. Added to additional buffer. Size: {len(self.additional_buffer)}")
                    elif self.status == 'clustered':
                        if status_of_sender == 'bootstrap':
                            logger.debug("[Add Vectors] Node is clustered but sender is bootstrap. Routing vectors.")
                            self.route_vectors_send(vectors, request_id)
                    else:
                        logger.error(f"Undefined status encountered: {self.status}")
                        raise Exception("ERROR: status Undefined")
    
    def get_id(self):
        with self.id_lock:
            self.id_count += 1
            n = len(str(abs(len(self.peers)+1)))
            numeric_id = int(self.node_id.split("node")[-1])
            return int(f'{self.id_count}{numeric_id:0{n}d}')
        
    def add_vectors_client(self, vectors: ListOfVectorsWithPayload, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='add_vectors_client').time():
            logger.debug(f"[Client Request] Client wants to add {len(vectors)} vectors. My Status: {self.status}")
            vectors_with_id = [(vector_content, self.get_id(), vector_payload) for vector_content, vector_payload  in vectors]
            with self.status_lock:
                status = self.status
            
            if status == 'bootstrap':
                if self.i_am_coord():
                    logger.debug("[Client Request] I am Coordinator (Bootstrap). Adding to buffer.")
                    self.add_vectors(vectors_with_id, status, request_id)
                else:
                    coord = self.coordinator()
                    if coord:
                        logger.debug(f"[Client Request] Forwarding {len(vectors)} vectors to Coordinator {coord.id}.")
                        coord.send(vectors_with_id, status, request_id)
                    else:
                        logger.error("Cannot send vectors: Coordinator undefined.")
            elif status == 'clustered':
                logger.debug("[Client Request] System Clustered. Routing vectors to appropriate peers.")
                self.route_vectors_send(vectors_with_id, request_id)
            elif status == 'clustering':
                if not self.i_am_coord():
                    raise Exception("Error: status Undefined")
                else:
                    logger.debug("[Client Request] System Clustering. Buffering at Coordinator.")
                    self.add_vectors(vectors_with_id, status, request_id)
            else:
                raise Exception("Error: Undefined status")
    
    def set_clusters(self, assignment, request_id): 
        with metrics.REQUEST_LATENCY.labels(operation='set_clusters').time():
            logger.info(f"[Set Clusters] Received assignment (ReqID: {request_id}).")
            for peer in self.peers:
                peer.set_clusters(assignment['peers_clusters'][peer.id])
            self.node_clusters = assignment['peers_clusters'][self.node_id]
            self.meta_hnsw = clustering_module.MetaHNSW.from_serializable_dict(assignment['meta_hnsw'])
            self.status = 'clustered'
            
            my_vecs = assignment['my_vectors']
            logger.info(f"[Set Clusters] Storing {len(my_vecs)} assigned vectors locally.")
            self.add_vectors(my_vecs, self.status, request_id)
            logger.debug(f"[Set Clusters] Complete. My Total Count: {self.get_count()}")

    def route_vector(self, vector: Vector, k: int):
        with metrics.REQUEST_LATENCY.labels(operation='route_vector').time():
            return clustering_module.find(self.meta_hnsw, vector, k)

    def route_vectors_send(self, vectors: ListOfVectorsComplete, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='route_vectors_send').time():
            logger.debug(f"[Router] Routing {len(vectors)} vectors among {len(self.peers) + 1} nodes (Replicas: {self.replicas}).")
            
            # --- OTTIMIZZAZIONE BATCH ---
            # 1. Estraiamo solo i vettori grezzi (liste di float) per il calcolo
            raw_vectors = [v[0] for v in vectors]
            
            # 2. Eseguiamo UNA sola chiamata batch invece di 1800 chiamate singole.
            # Ritorna una lista di liste, es: [[cluster_id_A], [cluster_id_B], ...]
            # Questo riduce il tempo di calcolo da ~4s a ~0.05s
            target_clusters_batch = clustering_module.find_batch(self.meta_hnsw, raw_vectors, 1)
            # -----------------------------

            assigned_vectors = [[] for _ in range(len(self.peers))]
            to_me = []
            
            # 3. Iteriamo sui vettori originali e sui risultati pre-calcolati
            for i, (vector_content, vector_id, vector_payload) in enumerate(vectors):
                top_1 = target_clusters_batch[i][0]
                # Aggiungi cluster_id al payload
                enhanced_payload = f"{vector_payload}|cluster_id:{top_1}"
                vector_tuple = (vector_content, vector_id, enhanced_payload)
                
                found = 0
                for index, peer in enumerate(self.peers):
                    if peer.contains(top_1):
                        assigned_vectors[index].append(vector_tuple)
                        found += 1
                        # logger.debug(f"[DEBUG] found replica for vector: {found}") # Decommentare solo se necessario per debug profondo

                # Logica di replicazione (Invariata)
                if found != self.replicas:
                    # logger.debug(f"[DEBUG] to me replica for vector: {found}")
                    to_me.append(vector_tuple)
                
                if found < self.replicas-1:
                    logger.debug(f"[DEBUG] not enough replicas found: {found} (Cluster ID: {top_1})")
            
            # Log distribution summary
            summary = [f"Peer {peer.id}: {len(assigned_vectors[i])}" for i, peer in enumerate(self.peers)]
            summary.append(f"Me ({self.node_id}): {len(to_me)}")
            logger.debug(f"[Router] Distribution: {', '.join(summary)}")

            # 4. Invio Parallelo (Invariato)
            def send_to_peer(index, peer, vectors_chunk):
                try:
                    # CHUNKING LOGIC
                    CHUNK_SIZE = 5000 # Safe size to stay under 100MB/512MB limit
                    total_sent = 0
                    
                    if len(vectors_chunk) > CHUNK_SIZE:
                        logger.debug(f"[Router] Chunking {len(vectors_chunk)} vectors for {peer.id} into blocks of {CHUNK_SIZE}")
                        for i in range(0, len(vectors_chunk), CHUNK_SIZE):
                            sub_chunk = vectors_chunk[i : i + CHUNK_SIZE]
                            peer.send(sub_chunk, self.status, request_id)
                            total_sent += len(sub_chunk)
                    else:
                        peer.send(vectors_chunk, self.status, request_id)
                        total_sent = len(vectors_chunk)
                        
                    return (peer.id, True, total_sent)
                except Exception as e:
                    logger.error(f"[Router] Failed to send to {peer.id}: {e}")
                    return (peer.id, False, len(vectors_chunk))
            
            # Submit tasks for peers with vectors
            tasks = []
            for index, peer in enumerate(self.peers):
                if len(assigned_vectors[index]) > 0:
                    tasks.append((send_to_peer, (index, peer, assigned_vectors[index])))
            # Definisci il task locale
            task_locale = None
            if len(to_me) > 0:
                # Creiamo una lambda per uniformare la firma
                task_locale = (lambda: (self.node_id, True, self.add_vectors(to_me, self.status, request_id)), ())
            # Aggiungi il task locale alla lista dei task remoti
            if task_locale:
                tasks.append(task_locale)
            # Esegui TUTTO insieme (Remoti + Locale corrono in parallelo)
            results = self._run_parallel_tasks(tasks)

            # Check results
            failed_sends = [peer_id for peer_id, success, count in results if not success]
            
            if failed_sends:
                logger.warning(f"[Router] Failed to send vectors to peers: {failed_sends}")


    def linear_search(self, query: ListOfVectors, topk: int):
        with metrics.REQUEST_LATENCY.labels(operation='linear_search').time():
            with self.vector_buffer_lock:
                current_buffer = self.vector_buffer[:]
            with self.additional_buffer_lock:
                current_additional = self.additional_buffer[:]
            
            total_vectors = current_buffer + current_additional
            logger.debug(f"[Linear Search] Searching buffer ({len(total_vectors)} vectors) for {len(query)} queries.")
            
            to_return = []
            for qv in query:
                distances_calcs = [(vector, utils.cosine_similarity(vector[0], qv)) for vector in total_vectors]
                distances_calcs.sort(key=lambda x: x[1], reverse=True)
                to_return.extend([{'id': v[1], 'score': d, 'payload': {'string': v[2], 'vector': v[0]}} for v, d in distances_calcs[:topk]])
            for i, res in enumerate(to_return):
                logger.info(f"[Linear Search] Result {i+1}: ID={res.get('id')}, Score={res.get('score')}")
            return to_return

    def query_me(self, query: ListOfVectors, topk: int):
        with metrics.REQUEST_LATENCY.labels(operation='local_query').time():
            if (self.status == 'bootstrap' or self.status == 'clustering') and self.i_am_coord():
                logger.debug(f"[Local Query] Bootstrap/Clustering mode: performing linear search on buffer.")
                return self.linear_search(query, topk)
            logger.debug(f"[Local Query] Querying local Qdrant for {len(query)} vectors across {len(self.node_clusters)} collections.")
            all_results = []
            for cluster_id, _ in self.node_clusters:
                collection_name = self.cluster_collections.get(cluster_id, f"{self.collection_name}_cluster_{cluster_id}")
                cluster_results = qdrant_module.query_vectors(self.qdrant_url, collection_name, query, topk)
                all_results.extend(cluster_results)
            all_results.sort(key=lambda x: x['score'], reverse=True)
            results = all_results[:topk * len(query)]
            for i, res in enumerate(results):
                logger.info(f"[Local Qdrant Query] Result {i+1}: ID={res.get('id')}, Score={res.get('score')}")
            return results
    
    def query(self, query: ListOfVectors, topk: int, request_id: int):
        with metrics.REQUEST_LATENCY.labels(operation='global_query').time():
            logger.info(f"[Global Query] Processing query (ReqID: {request_id}). Queries: {len(query)}, TopK: {topk}")
            response = []
            if self.meta_hnsw is None or self.status == 'clustering':
                if self.i_am_coord():
                    logger.warning("MetaHNSW not built or Clustering in progress. Coordinator falling back to Linear Search.")
                    return self.linear_search(query, topk)
                else:
                    logger.warning("MetaHNSW not built. Forwarding query to Coordinator.")
                    coord = self.coordinator()
                    if coord:
                        return coord.query_peer(query, topk, request_id)
                    else:
                        logger.error("Coordinator not found during bootstrap query.")
                        return []
            
            to_query_peer = [[] for _ in range(len(self.peers))]
            to_query_me = []
            
            # Routing queries
            for vector in query:
                top_3 = self.route_vector(vector, 3)
                for index, peer in enumerate(self.peers):
                    for result in top_3:
                        if peer.contains(result):
                            to_query_peer[index].append(vector)
                for result in top_3:
                    if result in [c[0] for c in self.node_clusters]:
                        to_query_me.append(vector)
            
            # Log Routing
            q_summary = [f"Peer {peer.id}: {len(to_query_peer[i])}" for i, peer in enumerate(self.peers)]
            q_summary.append(f"Me: {len(to_query_me)}")
            logger.debug(f"[Global Query] Routing: {', '.join(q_summary)}")

            # Parallel execution of queries
            def query_peer_task(index, peer, queries):
                try:
                    results = peer.query_peer(queries, topk, request_id)
                    return (peer.id, True, results)
                except Exception as e:
                    logger.error(f"[Global Query] Failed to query {peer.id}: {e}")
                    return (peer.id, False, [])
            
            # Submit query tasks to thread pool
            tasks = []
            for index, peer in enumerate(self.peers):
                if to_query_peer[index]:
                    tasks.append((query_peer_task, (index, peer, to_query_peer[index])))
            
            # Also query self in parallel if needed
            if to_query_me:
                tasks.append((lambda: (self.node_id, True, self.query_me(to_query_me, topk)), ()))
            
            results_list = self._run_parallel_tasks(tasks)
            
            # Collect all results
            response = []
            failed_queries = []
            
            for node_id, success, results in results_list:
                if success:
                    response.extend(results)
                else:
                    failed_queries.append(node_id)
            
            if failed_queries:
                logger.warning(f"[Global Query] Failed to query nodes: {failed_queries}")
            
            logger.debug(f"[Global Query] Aggregating {len(response)} raw results...")

            seen = set()
            unique_response = []
            for item in response:
                item_id = item.get('id')
                if item_id not in seen:
                    seen.add(item_id)
                    unique_response.append(item)
            
            # Sorting
            unique_response = sorted(unique_response, key=lambda x: x['score'], reverse=True) 
            logger.info(f"[Global Query] Final unique results: {len(unique_response)}")
            for i, res in enumerate(unique_response):
                logger.info(f"[Global Query] Result {i+1}: ID={res.get('id')}, Score={res.get('score')}")
            return unique_response
    
    def get_count(self):
        with metrics.REQUEST_LATENCY.labels(operation='get_count_function').time():
            return qdrant_module.count(self.qdrant_url, self.collection_name)
    
    def get_count_client(self):
        with metrics.REQUEST_LATENCY.labels(operation='get_count_client').time():
            res = {}
            for peer in self.peers:
                res[peer.id] = peer.get_count()
            res[self.node_id] = self.get_count()
            return res