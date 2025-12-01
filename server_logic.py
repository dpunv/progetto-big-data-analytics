from typing import List, Tuple, Dict
import time
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
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

logger = logging.getLogger(__name__)

GRPC_OPTIONS = [
    ('grpc.max_send_message_length', 512 * 1024 * 1024),
    ('grpc.max_receive_message_length', 512 * 1024 * 1024)
]

class ServerStatus(Enum):
    BOOTSTRAP = 'bootstrap'
    CLUSTERING = 'clustering'
    CLUSTERED = 'clustered'

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
    
    def send(self, vectors: ListOfVectorsComplete, status: str, id, retries=3):
        logger.debug(f"--> [Peer Send] Sending {len(vectors)} vectors to Peer {self.id} ({self.grpc_url}) | Status: {status}")
        
        proto_vectors = [
            p2p_pb2.VectorPoint(vector=v[0], id=v[1], payload=v[2], cluster=v[3]) 
            for v in vectors
        ]
        
        req = p2p_pb2.AddVectorsRequest(
            req_id=id,
            content=proto_vectors,
            type=status
        )

        for attempt in range(retries):
            try:
                self.stub.ReceiveVectors(req, timeout=10) # Add timeout
                logger.debug(f"--> [Peer Send] Success: Sent vectors to {self.id}")
                return True
            except grpc.RpcError as e:
                logger.warning(f"--> [Peer Send] Attempt {attempt+1}/{retries} FAILED to {self.id}: {e}")
                time.sleep(0.5 * (attempt + 1)) # Exponential backoff
        
        logger.error(f"--> [Peer Send] PERMANENT FAILURE to {self.id} after {retries} attempts.")
        return False

    def send_clusters(self, assignment, clusters, meta_hnsw, request_id, retries=3):
        logger.debug(f"--> [Peer SendClusters] Sending cluster assignment to Peer {self.id}")
        to_send_dict = {
            'my_vectors': [vector_complete for cluster_id, _ in assignment[self.id] for vector_complete in clusters[cluster_id][1]],
            'peers_clusters': assignment,
            'meta_hnsw': meta_hnsw.to_serializable_dict()
        }

        self.set_clusters(assignment[self.id])
        
        data_bytes = pickle.dumps(to_send_dict)
        
        req = p2p_pb2.SetClustersRequest(req_id=request_id, content_pickle=data_bytes)
        
        for attempt in range(retries):
            try:
                self.stub.SetClusters(req, timeout=30)
                logger.debug(f"--> [Peer SendClusters] Success: Sent clusters to {self.id}")
                return True
            except grpc.RpcError as e:
                logger.warning(f"--> [Peer SendClusters] Attempt {attempt+1}/{retries} FAILED to {self.id}: {e}")
                time.sleep(1)
        
        logger.error(f"--> [Peer SendClusters] PERMANENT FAILURE to {self.id}")
        return False

    def query_peer(self, query: ListOfVectors, topk: int, request_id: int, retries=3):
        logger.debug(f"--> [Peer Query] Querying Peer {self.id} for {len(query)} vectors")
        # query contains tuples (vector, cluster_id)
        proto_query = [p2p_pb2.VectorList(values=v[0], cluster_id=str(v[1])) for v in query]
        req = p2p_pb2.QueryRequest(req_id=request_id, query_vectors=proto_query, topk=topk)
        
        for attempt in range(retries):
            try:
                response = self.stub.QueryPeer(req, timeout=10)
                
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
                logger.warning(f"--> [Peer Query] Attempt {attempt+1}/{retries} FAILED to {self.id}: {e}")
                time.sleep(0.5)
        
        logger.error(f"--> [Peer Query] PERMANENT FAILURE to {self.id}")
        return []

    def get_count(self):
        try:
            return requests.get(f'{self.url}/count_peer', timeout=5).json()
        except Exception as e:
            logger.error(f"Error getting count from {self.id}: {e}")
            return -1
            
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
        self.status = ServerStatus.BOOTSTRAP
        self.replicas = replicas
        self.peers = []
        self.meta_hnsw = None
        
        # New Queue-based architecture
        self.vector_queue = queue.Queue()
        self.bootstrap_buffer = [] # Only used by Coordinator in BOOTSTRAP
        self.bootstrap_buffer_lock = threading.RLock() # For linear search access
        
        self.num_vectors_before_clustering = num_vectors_before_clustering
        self.query_received_count = 0
        self.metrics_lock = threading.Lock()
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
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, name="QueueWorker", daemon=True)
        self.worker_thread.start()
        
        logger.info(f"ServerApp initialized: ID={id}, URL={url}, Coords={coordinator_url}")
    
    def create_cluster_collections(self):
        logger.warning(f"Creating collections for {len(self.node_clusters)} clusters: {[c[0] for c in self.node_clusters]}")
        for cluster in self.node_clusters:
            coll_name = utils.get_collection_name(self.collection_name, cluster[0])
            logger.warning(f"Creating collection: {coll_name}")
            qdrant_module.create_collection(self.qdrant_url, coll_name, self.dimension)

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
    
    def _process_queue(self):
        """Main worker loop to process incoming vectors."""
        logger.info("*** Queue Worker Thread Started ***")
        while True:
            try:
                # Get vectors from queue (blocking)
                item = self.vector_queue.get()
                if item is None:
                    break # Stop signal
                
                vectors, sender_status, request_id = item
                
                if self.status == ServerStatus.BOOTSTRAP:
                    if self.i_am_coord():
                        with self.bootstrap_buffer_lock:
                            self.bootstrap_buffer.extend(vectors)
                            buffer_len = len(self.bootstrap_buffer)
                        
                        if buffer_len % 1000 == 0:
                            logger.info(f"[Worker] Bootstrap Buffer: {buffer_len}/{self.num_vectors_before_clustering}")

                        if buffer_len >= self.num_vectors_before_clustering:
                            logger.info("THRESHOLD REACHED: Triggering Clustering.")
                            self.status = ServerStatus.CLUSTERING
                            self._perform_clustering(request_id)
                    else:
                        # Non-coordinator in BOOTSTRAP shouldn't really get vectors unless forwarded?
                        # Or if they are 'my_vectors' from SetClusters (which changes status first)
                        # If we receive vectors here, it might be a race or misrouting.
                        # But if sender_status is 'clustered', we should probably insert?
                        if sender_status == 'clustered':
                             # Wait for status to become CLUSTERED
                             logger.info(f"[Worker] Received clustered vectors while in BOOTSTRAP. Waiting for SetClusters...")
                             start_wait = time.time()
                             while self.status == ServerStatus.BOOTSTRAP:
                                 if time.time() - start_wait > 30: # 30s timeout
                                     logger.error("[Worker] Timeout waiting for SetClusters. Dropping vectors.")
                                     break
                                 time.sleep(0.1)
                             
                             if self.status == ServerStatus.CLUSTERED:
                                 logger.info("[Worker] Status changed to CLUSTERED. Processing vectors.")
                                 qdrant_module.insert_vectors_generic(self.qdrant_url, self.collection_name, vectors, self.batch_size_retry, self.batch_size)
                        else:
                             logger.debug(f"[Worker] Received {len(vectors)} vectors in BOOTSTRAP (Not Coord). Dropping or waiting?")

                elif self.status == ServerStatus.CLUSTERING:
                    # If we are clustering, we just let vectors sit in the queue?
                    # No, we just popped them!
                    # We need to buffer them temporarily until clustering finishes.
                    # But wait, if _perform_clustering is blocking, we wouldn't be here popping!
                    # _perform_clustering is called FROM this thread.
                    # So we only reach here if we are NOT clustering (or just finished).
                    # Wait, if status was set to CLUSTERING by another thread?
                    # No, only THIS thread sets status to CLUSTERING (in the block above).
                    # So if self.status is CLUSTERING here, it means... wait.
                    # If we are in CLUSTERING state, it means we are currently running _perform_clustering?
                    # No, _perform_clustering blocks this thread.
                    # So we can't be popping from queue while clustering.
                    # UNLESS: status was set to CLUSTERING, and we returned from _perform_clustering?
                    # No, _perform_clustering sets status to CLUSTERED at the end.
                    
                    # So, effectively, we should never see status == CLUSTERING here 
                    # because we transition BOOTSTRAP -> CLUSTERING -> (block) -> CLUSTERED
                    # all in one go.
                    
                    # Exception: If we want to support non-blocking clustering?
                    # No, blocking is safer for consistency.
                    pass

                elif self.status == ServerStatus.CLUSTERED:
                    # Route or Insert
                    if sender_status == 'clustered':
                        # Vectors are assigned to me. Insert.
                        qdrant_module.insert_vectors_generic(self.qdrant_url, self.collection_name, vectors, self.batch_size_retry, self.batch_size)
                    else:
                        # Vectors need routing
                        self.route_vectors_send(vectors, request_id)
                
                self.vector_queue.task_done()
                
            except Exception as e:
                logger.error(f"[Worker] Error processing queue item: {e}", exc_info=True)

    def _perform_clustering(self, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='clustering').time():
            logger.info(f"*** START CLUSTERING (ReqID: {request_id}) ***")
            
            # Use the buffer
            with self.bootstrap_buffer_lock:
                vectors_to_cluster = self.bootstrap_buffer[:]
                # We don't clear buffer yet? Actually we can, since we have a local copy.
                self.bootstrap_buffer = [] 
            
            peers_with_me = self.peers[:]
            peers_with_me.append(Peer(self.node_id, self.url, self.grpc_url))
            
            logger.info("Clustering: Calculating node assignments...")
            clusters = clustering_module.get_clusters(vectors_to_cluster)
            assignment = clustering_module.get_node_assignment(clusters, peers_with_me, self.replicas)
            self.meta_hnsw = clustering_module.build_meta_hnsw(clusters, self.dimension)
            
            [logger.info(f"Clusters/Vectors per node: {nid}: {len(c_list)} clusters, {sum([len(clusters[cid][1]) for cid, _ in c_list])} vectors") for nid, c_list in assignment.items()]

            self.node_clusters = assignment[self.node_id]
            self.create_cluster_collections()

            logger.info("Clustering: Broadcasting assignments to peers...")
            
            def broadcast_to_peer(peer):
                return (peer.id, peer.send_clusters(assignment, clusters, self.meta_hnsw, request_id))
            
            tasks = [(broadcast_to_peer, (peer,)) for peer in self.peers]
            results = self._run_parallel_tasks(tasks)
            
            failed_broadcasts = [pid for pid, success in results if not success]
            if failed_broadcasts:
                logger.error(f"[Clustering] Failed to broadcast to: {failed_broadcasts}")
            else:
                logger.info("[Clustering] Broadcast complete.")

            # Update status
            self.status = ServerStatus.CLUSTERED
            logger.info("*** Status changed to CLUSTERED ***")
            
            # Process local assignment
            local_vectors = [(v, v_id, v_payload, str(cluster_id)) for cluster_id, _ in assignment[self.node_id] for v, v_id, v_payload, _ in clusters[cluster_id][1]]
            logger.info(f"Clustering: Inserting {len(local_vectors)} assigned vectors locally.")
            
            # We can insert directly or put back in queue?
            # Putting back in queue is safer to keep single writer?
            # But we want to ensure they are processed.
            # Let's insert directly since we are in the worker thread.
            if local_vectors:
                qdrant_module.insert_vectors_generic(self.qdrant_url, self.collection_name, local_vectors, self.batch_size_retry, self.batch_size)
            
            logger.info("*** CLUSTERING FINISHED ***")

    def add_vectors(self, vectors: ListOfVectorsComplete, status_of_sender: str, request_id):
        """
        Add vectors to the processing queue.
        This is non-blocking (mostly).
        """
        with metrics.REQUEST_LATENCY.labels(operation='add_vectors').time():
            self.vector_queue.put((vectors, status_of_sender, request_id))
            logger.debug(f"[AddVectors] Enqueued {len(vectors)} vectors. Queue size: {self.vector_queue.qsize()}")

    def get_id(self):
        with self.id_lock:
            self.id_count += 1
            n = len(str(abs(len(self.peers)+1)))
            numeric_id = int(self.node_id.split("node")[-1])
            return int(f'{self.id_count}{numeric_id:0{n}d}')
        
    def add_vectors_client(self, vectors: ListOfVectorsWithPayload, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='add_vectors_client').time():
            logger.debug(f"[Client Request] Adding {len(vectors)} vectors. Status: {self.status.value}")
            vectors_with_id = [(vector_content, self.get_id(), vector_payload, "-1") for vector_content, vector_payload  in vectors]
            
            if self.status == ServerStatus.BOOTSTRAP:
                if self.i_am_coord():
                    self.add_vectors(vectors_with_id, 'bootstrap', request_id)
                else:
                    coord = self.coordinator()
                    if coord:
                        # Forward to coordinator
                        # We use send with 'bootstrap' status
                        coord.send(vectors_with_id, 'bootstrap', request_id)
                    else:
                        logger.error("Cannot send vectors: Coordinator undefined.")
            
            elif self.status == ServerStatus.CLUSTERING:
                # If clustering, we forward to coordinator (who will buffer in queue)
                if self.i_am_coord():
                    self.add_vectors(vectors_with_id, 'clustering', request_id)
                else:
                    coord = self.coordinator()
                    if coord:
                        coord.send(vectors_with_id, 'clustering', request_id)
            
            elif self.status == ServerStatus.CLUSTERED:
                # Route
                self.route_vectors_send(vectors_with_id, request_id)

    def set_clusters(self, assignment, request_id): 
        with metrics.REQUEST_LATENCY.labels(operation='set_clusters').time():
            logger.info(f"[Set Clusters] Received assignment (ReqID: {request_id}).")
            for peer in self.peers:
                peer.set_clusters(assignment['peers_clusters'][peer.id])
            self.node_clusters = assignment['peers_clusters'][self.node_id]
            self.meta_hnsw = clustering_module.MetaHNSW.from_serializable_dict(assignment['meta_hnsw'])
            self.create_cluster_collections()
            
            self.status = ServerStatus.CLUSTERED
            
            my_vecs = assignment['my_vectors']
            logger.info(f"[Set Clusters] Storing {len(my_vecs)} assigned vectors locally.")
            
            # Insert directly or queue?
            # Queue is better to maintain order if other things are happening?
            # But we want to be sure it's done.
            # Let's queue them with status 'clustered' so they get inserted.
            self.add_vectors(my_vecs, 'clustered', request_id)

    def route_vector(self, vector: Vector, k: int):
        with metrics.REQUEST_LATENCY.labels(operation='route_vector').time():
            return clustering_module.find(self.meta_hnsw, vector, k)

    def route_vectors_send(self, vectors: ListOfVectorsComplete, request_id):
        with metrics.REQUEST_LATENCY.labels(operation='route_vectors_send').time():
            # logger.debug(f"[Router] Routing {len(vectors)} vectors.")
            
            raw_vectors = [v[0] for v in vectors]
            target_clusters_batch = clustering_module.find_batch(self.meta_hnsw, raw_vectors, 1)
            
            assigned_vectors = [[] for _ in range(len(self.peers))]
            to_me = []
            
            for i, (vector_content, vector_id, vector_payload, vector_cluster) in enumerate(vectors):
                top_1 = target_clusters_batch[i][0]
                vector_tuple = (vector_content, vector_id, vector_payload, str(top_1))
                
                found = 0                
                if top_1 in {c[0] for c in self.node_clusters}:
                    to_me.append(vector_tuple)
                    found += 1

                for index, peer in enumerate(self.peers):
                    if peer.contains(top_1):
                        assigned_vectors[index].append(vector_tuple)
                        found += 1
                
                if found < self.replicas:
                     logger.warning(f"[ROUTER] not enough replicas found: {found} (Cluster ID: {top_1})")
            
            # Send to peers
            def send_to_peer(peer, vectors_chunk):
                CHUNK_SIZE = 5000
                if len(vectors_chunk) > CHUNK_SIZE:
                    for i in range(0, len(vectors_chunk), CHUNK_SIZE):
                        peer.send(vectors_chunk[i : i + CHUNK_SIZE], 'clustered', request_id)
                else:
                    peer.send(vectors_chunk, 'clustered', request_id)
                return True

            tasks = []
            for index, peer in enumerate(self.peers):
                if len(assigned_vectors[index]) > 0:
                    tasks.append((send_to_peer, (peer, assigned_vectors[index])))
            
            # Send to self (queue it)
            if to_me:
                self.add_vectors(to_me, 'clustered', request_id)
            
            self._run_parallel_tasks(tasks)

    def linear_search(self, query: ListOfVectors, topk: int):
        with metrics.REQUEST_LATENCY.labels(operation='linear_search').time():
            with self.bootstrap_buffer_lock:
                total_vectors = self.bootstrap_buffer[:]
            
            # Also check queue? No, too complex. Just check what's in buffer.
            
            logger.debug(f"[Linear Search] Searching buffer ({len(total_vectors)} vectors).")
            
            to_return = []
            for qv in query:
                query_vec = qv[0] if isinstance(qv, tuple) else qv
                distances_calcs = [(vector, utils.cosine_similarity(vector[0], query_vec)) for vector in total_vectors]
                distances_calcs.sort(key=lambda x: x[1], reverse=True)
                to_return.extend([{'id': v[1], 'score': d, 'payload': {'string': v[2], 'vector': v[0]}} for v, d in distances_calcs[:topk]])
            return to_return

    def query_me(self, query, topk: int):
        with metrics.REQUEST_LATENCY.labels(operation='local_query').time():
            if self.status in [ServerStatus.BOOTSTRAP, ServerStatus.CLUSTERING] and self.i_am_coord():
                logger.debug(f"[Local Query] Bootstrap/Clustering mode: performing linear search on buffer.")
                return self.linear_search(query, topk)
            
            results = qdrant_module.query_vectors_generic(self.qdrant_url, self.collection_name, query, topk)
            return results

    def query(self, query: ListOfVectors, topk: int, request_id: int):
        with metrics.REQUEST_LATENCY.labels(operation='global_query').time():
            logger.info(f"[Global Query] Processing query (ReqID: {request_id}). Queries: {len(query)}, TopK: {topk}")
            response = []
            with self.metrics_lock:
                self.query_received_count += 1
            if self.meta_hnsw is None or self.status == 'clustering':
                if self.i_am_coord():
                    return self.linear_search(query, topk)
                else:
                    coord = self.coordinator()
                    if coord:
                        return coord.query_peer(query, topk, request_id)
                    return []
            
            to_query_peer = [[] for _ in range(len(self.peers))]
            to_query_me = []
            
            # Routing queries with cluster hit tracking
            for vector in query:
                top_3 = self.route_vector(vector, 3)
                for cluster_id in top_3:
                    metrics.CLUSTER_HITS.labels(cluster_id=str(cluster_id)).inc()
        
                for index, peer in enumerate(self.peers):
                    for result in top_3:
                        if peer.contains(result):
                            to_query_peer[index].append((vector, result))
                for result in top_3:
                    if result in [c[0] for c in self.node_clusters]:
                        to_query_me.append((vector, result))
            
            # Track routing decisions
            for index, peer in enumerate(self.peers):
                if to_query_peer[index]:
                    # Conta quanti VETTORI individuali sono stati routati
                    metrics.VECTORS_ROUTED.labels(target_node=peer.id).inc(len(to_query_peer[index]))
                    # Conta quante QUERY (chiamate uniche al peer) vengono fatte
                    metrics.QUERY_ROUTED.labels(target_node=peer.id).inc(1)
            
            # Log Routing
            q_summary = [f"Peer {peer.id}: {len(to_query_peer[i])}" for i, peer in enumerate(self.peers)]
            q_summary.append(f"Me: {len(to_query_me)}")
            logger.debug(f"[Global Query] Routing: {', '.join(q_summary)}")

            # Parallel execution of queries
            def query_peer_task(index, peer, queries):
                try:
                    with metrics.PEER_LATENCY.labels(peer_id=peer.id, operation='query').time():
                        results = peer.query_peer(queries, topk, request_id)
                    return (peer.id, True, results)
                except Exception as e:
                    logger.error(f"[Global Query] Failed to query {peer.id}: {e}")
                    metrics.PEER_FAILURES.labels(peer_id=peer.id, operation='query').inc()
                    return (peer.id, False, [])
            
            # Submit query tasks to thread pool
            tasks = []
            for index, peer in enumerate(self.peers):
                if to_query_peer[index]:
                    tasks.append((query_peer_task, (peer, to_query_peer[index])))
            
            if to_query_me:
                tasks.append((lambda: (self.node_id, True, self.query_me(to_query_me, topk)), ()))
            
            results_list = self._run_parallel_tasks(tasks)
            
            response = []
            for _, success, results in results_list:
                if success:
                    response.extend(results)
            
            # Deduplicate and sort
            seen = set()
            unique_response = []
            for item in response:
                item_id = item.get('id')
                if item_id not in seen:
                    seen.add(item_id)
                    unique_response.append(item)
            
            unique_response = sorted(unique_response, key=lambda x: x['score'], reverse=True) 
            return unique_response
    
    def get_count(self):
        with metrics.REQUEST_LATENCY.labels(operation='get_count_function').time():
            if self.status == ServerStatus.CLUSTERED:
                total_count = 0
                for cluster in self.node_clusters:
                    cluster_id = cluster[0]
                    collection_name = utils.get_collection_name(self.collection_name, cluster_id)
                    c = qdrant_module.count(self.qdrant_url, collection_name)
                    if c != -1:
                        total_count += c
                return total_count
            else:
                with self.bootstrap_buffer_lock:
                    return len(self.bootstrap_buffer)
    
    def get_count_client(self):
        with metrics.REQUEST_LATENCY.labels(operation='get_count_client').time():
            res = {}
            for peer in self.peers:
                res[peer.id] = peer.get_count()
            res[self.node_id] = self.get_count()
            return res
        
    def metrics_snapshot(self):
        return {
                'query_received_count': self.query_received_count
        }
