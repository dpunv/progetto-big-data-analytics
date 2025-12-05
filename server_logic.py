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
import socket
import struct
import random

logger = logging.getLogger(__name__)

GRPC_OPTIONS = [
    ('grpc.max_send_message_length', 512 * 1024 * 1024),
    ('grpc.max_receive_message_length', 512 * 1024 * 1024)
]

# Costanti per il carico

ALPHA_VECTOR = 1.5e-7
BETA_QPS = 4e-4
BASE_LOAD = 0.5

# --- Costanti UDP Heartbeat ---
HEARTBEAT_INTERVAL = 2.0  # Veloce per il test (in prod: 5.0)
TIMEOUT_LIMIT = 10.0      # Se non ti sento per 10s, sei morto (in prod: 30.0)
JITTER = 1.0

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
    def __init__(self, id, url, qdrant_url, grpc_url, coordinator_url, replicas=3, collection_name="vectors", num_vectors_before_clustering=10_000, dimension=384, batch_size=256, batch_size_retry=64, udp_port=None):
        self.node_id = id
        self.url = url
        self.qdrant_url = qdrant_url
        self.grpc_url = grpc_url
        self.udp_port = udp_port  # NUOVO: Porta UDP locale
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
        
        # --- NUOVO: Gestione Stato Globale e Carico ---
        self.current_load = 0.5
        self.last_qps_check_time = time.time()
        self.last_query_count = 0
        
        # Tabella Salute Peer: { int_node_id: {'load': float, 'last_seen': timestamp, 'status': 'online'|'offline'} }
        self.peer_health = {} 
        self.peer_health_lock = threading.Lock()
        
        # UDP Socket
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind(('0.0.0.0', self.udp_port))
        self.udp_sock.setblocking(False) # Non-blocking per il receive loop

        # Avvio Thread UDP (Sender e Receiver)
        self.running = True
        self.udp_sender_thread = threading.Thread(target=self._udp_heartbeat_loop, name="UDP-Sender", daemon=True)
        self.udp_receiver_thread = threading.Thread(target=self._udp_listener_loop, name="UDP-Receiver", daemon=True)
        self.udp_sender_thread.start()
        self.udp_receiver_thread.start()

        logger.info(f"ServerApp initialized: ID={id}, URL={url}, Coords={coordinator_url}, UDP={udp_port}")

    def _calculate_load(self):
        """Calcola il carico corrente basato su Vettori e QPS."""
        try:
            # 1. Calcolo QPS Istantaneo
            now = time.time()
            time_diff = now - self.last_qps_check_time
            if time_diff <= 0: time_diff = 1.0 # Evita divisione per zero
            
            with self.metrics_lock:
                current_queries = self.query_received_count
            
            qps = (current_queries - self.last_query_count) / time_diff
            
            # Aggiorna per il prossimo ciclo
            self.last_query_count = current_queries
            self.last_qps_check_time = now

            # 2. Ottieni Numero Vettori (Stimato o Reale)
            # Nota: get_count() potrebbe essere lento se chiama Qdrant via HTTP. 
            # Per l'heartbeat frequente, potremmo usare una variabile cachata o approssimata.
            # Qui usiamo una chiamata rapida.
            if self.status == ServerStatus.BOOTSTRAP:
                with self.bootstrap_buffer_lock:
                    num_vectors = len(self.bootstrap_buffer)
            else:
                # In produzione, meglio aggiornare questo valore in background ogni X secondi
                # per non rallentare l'heartbeat loop. Per ora assumiamo sia veloce o usiamo un valore salvato.
                num_vectors = self.get_count() # Assicurati che questo sia veloce o cachato
                if num_vectors == -1: num_vectors = 0

            # 3. Formula
            load = BASE_LOAD + (ALPHA_VECTOR * num_vectors) + (BETA_QPS * qps)
            return load, qps, num_vectors

        except Exception as e:
            logger.error(f"Error calculating load: {e}")
            return 0.5, 0, 0

    def _udp_heartbeat_loop(self):
        """Loop di invio: Calcola carico -> Invia a tutti -> Dorme (con Jitter)."""
        logger.info("Starting UDP Heartbeat Sender...")
        
        my_numeric_id = int(self.node_id.replace("node", "")) # Esempio: "node1" -> 1

        while self.running:
            try:
                # 1. Calcola Carico
                new_load, qps, n_vec = self._calculate_load()
                self.current_load = new_load
                
                # 2. Prepara Pacchetto Binario
                # Struct: int (4 bytes) + float (4 bytes) = 8 bytes totali
                # 'i': integer, 'f': float
                payload = struct.pack('!if', my_numeric_id, new_load)
                
                # 3. Invia a TUTTI i peer conosciuti
                # Nota: Dobbiamo conoscere l'IP e la porta UDP dei peer.
                # Assunzione: La porta UDP dei peer è calcolabile o salvata in self.peers.
                # Per semplicità, qui assumiamo che self.peers abbia un metodo o attributo per l'indirizzo UDP.
                # Visto che Peer ha `url` (http), deriveremo la porta UDP da lì (vedi run.py).
                
                peers_snapshot = list(self.peers) # Copia thread-safe
                for peer in peers_snapshot:
                    try:
                        # Logica per derivare IP e Porta UDP dal peer
                        # Assumiamo che Peer.url sia "http://localhost:8002"
                        # E che la porta UDP sia (PortaHTTP - 1000) come definito in run.py
                        peer_ip = peer.url.split("//")[1].split(":")[0]
                        peer_http_port = int(peer.url.split(":")[-1])
                        peer_udp_port = peer_http_port - 1000 # CONVENZIONE definita in run.py
                        
                        self.udp_sock.sendto(payload, (peer_ip, peer_udp_port))
                    except Exception as e:
                        # UDP fire and forget, loggiamo solo debug
                        pass

                # 4. Failure Detector (Watchdog) Locale
                self._check_peer_timeouts()


                # 5. Sleep con Jitter centralizzato
                sleep_time = HEARTBEAT_INTERVAL + random.uniform(-JITTER, JITTER)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in UDP Heartbeat loop: {e}")
                time.sleep(5)

    def _udp_listener_loop(self):
        """Loop di ricezione: Riceve pacchetti -> Aggiorna Tabella Salute."""
        logger.info("Starting UDP Listener...")
        while self.running:
            try:
                # Receive buffer size 1024 è più che sufficiente per 8 bytes
                data, addr = self.udp_sock.recvfrom(1024)
                
                if len(data) >= 8:
                    # Unpack: ID (int), Load (float)
                    peer_id, peer_load = struct.unpack('!if', data[:8])
                    
                    with self.peer_health_lock:
                        # Se il peer era offline o sconosciuto, logghiamo il ritorno
                        old_status = self.peer_health.get(peer_id, {}).get('status')
                        
                        self.peer_health[peer_id] = {
                            'load': peer_load,
                            'last_seen': time.time(),
                            'status': 'online'
                        }
                        
                        if old_status != 'online':
                            logger.info(f"UDP Monitor: Node {peer_id} detected ONLINE (Load: {peer_load:.2f})")

            except BlockingIOError:
                # Nessun dato disponibile (socket non bloccante), dormiamo un po' per non fondere la CPU
                time.sleep(0.1)

            except ConnectionResetError:
                # [FIX CRITICO PER WINDOWS]
                # Questo errore viene sollevato se inviamo un pacchetto a una porta chiusa.
                # Il sistema operativo lo riporta al listener. Dobbiamo ignorarlo.
                pass

            except OSError as e:
                # Gestione di altri errori di rete (es. buffer pieno, rete giù momentanea)
                # Se il socket è stato chiuso esplicitamente (self.running=False), usciamo puliti
                if not self.running:
                    break
                # Altrimenti ignoriamo l'errore transitorio
                pass

            except Exception as e:
                # Errori imprevisti (es. struct unpack fallito per dati corrotti)
                logger.error(f"UDP Listener unexpected error: {e}")
                # Qui dormiamo per evitare loop infiniti in caso di bug logici gravi
                time.sleep(1)

    def _check_peer_timeouts(self):
        """Controlla se qualcuno non risponde da > 30s."""
        now = time.time()
        timeout = 30.0
        
        with self.peer_health_lock:
            for pid, info in self.peer_health.items():
                if info['status'] == 'online':
                    if (now - info['last_seen']) > timeout:
                        logger.warning(f"FAILURE DETECTED: Node {pid} is now OFFLINE (Timeout).")
                        info['status'] = 'offline'
                        # Qui potresti voler rimuovere il peer dalla lista attiva self.peers 
                        # o marcarlo come non utilizzabile per le query.
                        
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
            def query_peer_task(peer, queries):
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
