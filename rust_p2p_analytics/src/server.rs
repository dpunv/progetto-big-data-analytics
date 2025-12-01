use crate::p2p::node_service_server::NodeService;
use crate::p2p::node_service_client::NodeServiceClient;
use crate::p2p::{
    AddVectorsRequest, Empty, QueryRequest, QueryResponse, RegisterPeersRequest, SetClustersRequest,
    VectorPoint, ScoredPoint, VectorList, ClusterInfo
};
use crate::models::{ServerStatus, Peer, Cluster};
use crate::clustering::{self, MetaHNSW};
use crate::qdrant::QdrantHandler;
use crate::{VectorWithPayload, VectorId, Vector};
use tonic::{Request, Response, Status};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use std::collections::HashMap;
use tracing::{info, warn, error, debug};
use std::time::Duration;
use futures::future;
use metrics::{counter, histogram, gauge};



pub struct ServerAppInner {
    pub id: String,
    pub url: String,
    pub grpc_url: String,
    pub coordinator_url: String,
    pub status: RwLock<ServerStatus>,
    pub peers: RwLock<Vec<Peer>>,
    pub qdrant: QdrantHandler,
    pub vector_queue: tokio::sync::mpsc::Sender<(Vec<VectorWithPayload>, String, i64)>,
    pub bootstrap_buffer: Mutex<Vec<VectorWithPayload>>,
    pub meta_hnsw: RwLock<Option<MetaHNSW>>,
    pub node_clusters: RwLock<Vec<Cluster>>,
    pub replicas: usize,
    pub num_before_clustering: usize,
    pub dimension: usize,
    pub peer_clients: RwLock<HashMap<String, NodeServiceClient<tonic::transport::Channel>>>,
    pub global_assignment: RwLock<HashMap<String, Vec<String>>>,
}

use std::ops::Deref;

#[derive(Clone)]
pub struct ServerApp {
    inner: Arc<ServerAppInner>,
}

impl Deref for ServerApp {
    type Target = ServerAppInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ServerApp {
    pub async fn new(
        id: String,
        url: String,
        grpc_url: String,
        qdrant_url: String,
        coordinator_url: String,
        replicas: usize,
        num_before_clustering: usize,
        dimension: usize,
    ) -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1000);
        let qdrant = QdrantHandler::new(qdrant_url).unwrap();
        
        let inner = Arc::new(ServerAppInner {
            id,
            url,
            grpc_url,
            coordinator_url,
            status: RwLock::new(ServerStatus::Bootstrap),
            peers: RwLock::new(Vec::new()),
            qdrant,
            vector_queue: tx,
            bootstrap_buffer: Mutex::new(Vec::new()),
            meta_hnsw: RwLock::new(None),
            node_clusters: RwLock::new(Vec::new()),
            replicas,
            num_before_clustering,
            dimension,
            peer_clients: RwLock::new(HashMap::new()),
            global_assignment: RwLock::new(HashMap::new()),
        });

        let app = Self { inner };
        let app_clone = app.clone();
        tokio::spawn(async move {
            app_clone.process_queue(&mut rx).await;
        });

        app
    }

    pub fn i_am_coord(&self) -> bool {
        self.inner.url == self.inner.coordinator_url
    }

    async fn process_queue(&self, rx: &mut tokio::sync::mpsc::Receiver<(Vec<VectorWithPayload>, String, i64)>) {
        info!("Queue Worker Started");
        while let Some((vectors, sender_status, req_id)) = rx.recv().await {
            let start = std::time::Instant::now();
            let status = self.inner.status.read().await.clone();
            match status {
                ServerStatus::Bootstrap => {
                    if self.i_am_coord() {
                        let mut buffer = self.inner.bootstrap_buffer.lock().await;
                        buffer.extend(vectors);
                        let len = buffer.len();
                        gauge!("app_bootstrap_buffer_size", len as f64);
                        if len % 1000 == 0 {
                            info!("Bootstrap Buffer: {}/{}", len, self.inner.num_before_clustering);
                        }
                        if len >= self.inner.num_before_clustering {
                            drop(buffer); // Release lock before clustering
                            self.trigger_clustering(req_id).await;
                        }
                    } else {
                        if sender_status == "clustered" {
                             let _ = self.inner.qdrant.insert_vectors("vectors", vectors).await;
                        } else {
                            self.forward_to_coordinator(vectors, req_id).await;
                        }
                    }
                }
                ServerStatus::Clustering => {
                    // Buffer or forward?
                    if self.i_am_coord() {
                         // Buffer
                         let mut buffer = self.inner.bootstrap_buffer.lock().await;
                         buffer.extend(vectors);
                    } else {
                        self.forward_to_coordinator(vectors, req_id).await;
                    }
                }
                ServerStatus::Clustered => {
                    if sender_status == "clustered" {
                        let _ = self.inner.qdrant.insert_vectors("vectors", vectors).await;
                    } else {
                        self.route_vectors(vectors, req_id).await;
                    }
                }
                }

            histogram!("app_request_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "process_queue");
        }
    }
    async fn forward_to_coordinator(&self, vectors: Vec<VectorWithPayload>, req_id: i64) {
        let peers = self.inner.peers.read().await;
        // Find coordinator by URL. Note: coordinator_url in ServerAppInner is http url.
        // We need to match it with peer.http_url
        let coord_peer = peers.iter().find(|p| p.http_url == self.inner.coordinator_url);
        
        if let Some(peer) = coord_peer {
            // Connect to gRPC
            // Assuming grpc_url is like "localhost:9001" or "127.0.0.1:9001"
            // We need to prepend http:// if missing, but tonic might need it.
            // run-rust.py passes "http://127.0.0.1:9001" now?
            // Let's check run-rust.py. It passes "http://127.0.0.1:..." for grpc_url in config?
            // In run-rust.py: 'grpc_url': f'http://127.0.0.1:{GRPC_START_PORT + i + 1}'
            // So it has http://.
            
            let endpoint = peer.grpc_url.clone();
            match crate::p2p::node_service_client::NodeServiceClient::connect(endpoint).await {
                Ok(mut client) => {
                    let proto_vectors: Vec<VectorPoint> = vectors.into_iter().map(|v| v.into()).collect();
                    let req = AddVectorsRequest {
                        req_id,
                        content: proto_vectors,
                        request_type: "bootstrap".to_string(),
                    };
                    if let Err(e) = client.receive_vectors(req).await {
                        error!("Failed to forward vectors to coordinator: {}", e);
                        counter!("app_peer_failures_total", 1, "peer_id" => "coordinator", "operation" => "forward");
                    }
                },
                Err(e) => {
                    error!("Failed to connect to coordinator: {}", e);
                    counter!("app_peer_failures_total", 1, "peer_id" => "coordinator", "operation" => "connect");
                },
            }
        } else {
            error!("Coordinator not found in peers list! URL: {}", self.inner.coordinator_url);
        }
    }

    async fn trigger_clustering(&self, req_id: i64) {
        let mut status = self.inner.status.write().await;
        *status = ServerStatus::Clustering;
        drop(status);
        
        let start = std::time::Instant::now();
        info!("*** START CLUSTERING (ReqID: {}) ***", req_id);
        let mut buffer = self.inner.bootstrap_buffer.lock().await;
        let vectors = buffer.clone();
        buffer.clear();
        drop(buffer);

        info!("Clustering: Copied {} vectors from buffer. Starting blocking task.", vectors.len());

        // Run clustering (CPU heavy, maybe spawn blocking)
        let peers = self.inner.peers.read().await.clone();
        let replicas = self.inner.replicas;
        let dimension = self.inner.dimension;
        let my_id = self.inner.id.clone();
        let my_grpc_url = self.inner.grpc_url.clone();
        
        let vectors_clone = vectors.clone();
        let (clusters, meta_hnsw, assignment) = tokio::task::spawn_blocking(move || {
            info!("Clustering [Blocking]: Starting perform_kmeans_auto_k with {} vectors...", vectors_clone.len());
            let clusters = clustering::perform_kmeans_auto_k(&vectors_clone, 2, 30); // Try k from 2 to 30
            info!("Clustering [Blocking]: K-Means done. Clusters found: {}", clusters.len());
            gauge!("app_clusters_created_total", clusters.len() as f64);
            
            let mut meta = MetaHNSW::new(dimension);
            info!("Clustering [Blocking]: Building MetaHNSW...");
            meta.build(&clusters);
            info!("Clustering [Blocking]: MetaHNSW built.");
            
            // Add self to peers for assignment
            let mut all_peers = peers.clone();
            all_peers.push(Peer { id: my_id, http_url: "".to_string(), grpc_url: my_grpc_url }); // URL doesn't matter for ID check
            
            info!("Clustering [Blocking]: Calculating node assignment for {} peers...", all_peers.len());
            let assignment = clustering::get_node_assignment(&clusters, &all_peers, replicas);
            info!("Clustering [Blocking]: Node assignment calculated.");
            
            (clusters, meta, assignment)
        }).await.unwrap();

        info!("Clustering: Blocking task finished. Updating local state.");

        // Broadcast assignment to all peers
        let peers = self.inner.peers.read().await;
        let meta_bytes = meta_hnsw.to_bytes().unwrap_or_default();
        
        info!("Clustering: Peers list: {:?}", peers.iter().map(|p| p.id.clone()).collect::<Vec<_>>());
        info!("Clustering: Assignment keys: {:?}", assignment.keys().collect::<Vec<_>>());

        // Prepare global assignment for broadcast
        let mut global_assignment_proto = Vec::new();
        for (node_id, clusters) in assignment.iter() {
            let cluster_ids: Vec<String> = clusters.iter().map(|c| c.id.clone()).collect();
            global_assignment_proto.push(crate::p2p::ClusterAssignment {
                node_id: node_id.clone(),
                cluster_ids,
            });
        }
        
        // Update local global assignment
        let mut assignment_map: HashMap<String, Vec<String>> = HashMap::new();
        for (node_id, clusters) in assignment.iter() {
            assignment_map.insert(node_id.clone(), clusters.iter().map(|c| c.id.clone()).collect());
        }
        let mut ga_guard = self.inner.global_assignment.write().await;
        *ga_guard = assignment_map;
        drop(ga_guard);

        // Collect tasks to wait for all set_clusters to complete
        let mut set_cluster_tasks = Vec::new();
        
        for peer in peers.iter() {
            if let Some(peer_clusters) = assignment.get(&peer.id) {
                let endpoint = peer.grpc_url.clone();
                let peer_clusters_proto: Vec<ClusterInfo> = peer_clusters.iter().map(|c| ClusterInfo {
                    id: c.id.clone(),
                    centroid: c.centroid.clone(),
                    vector_ids: vec![],
                }).collect();
                
                let req = SetClustersRequest {
                    clusters: peer_clusters_proto,
                    meta_hnsw: meta_bytes.clone(),
                    global_assignment: global_assignment_proto.clone(),
                };
                
                let peer_id = peer.id.clone();
                let task = tokio::spawn(async move {
                    match crate::p2p::node_service_client::NodeServiceClient::connect(endpoint).await {
                        Ok(mut client) => {
                            if let Err(e) = client.set_clusters(req).await {
                                error!("Failed to set clusters for peer {}: {}", peer_id, e);
                                counter!("app_peer_failures_total", 1, "peer_id" => peer_id.clone(), "operation" => "set_clusters");
                            } else {
                                info!("Successfully set clusters for peer {}", peer_id);
                            }
                        },
                        Err(e) => {
                            error!("Failed to connect to peer {} for set_clusters: {}", peer_id, e);
                            counter!("app_peer_failures_total", 1, "peer_id" => peer_id.clone(), "operation" => "set_clusters_connect");
                        },
                    }
                });
                set_cluster_tasks.push(task);
            } else {
                warn!("Clustering: No assignment for peer {}", peer.id);
            }
        }
        
        // Wait for ALL peers to create their collections before routing
        info!("Clustering: Waiting for {} peers to create collections...", set_cluster_tasks.len());
        futures::future::join_all(set_cluster_tasks).await;
        info!("Clustering: All peers ready. Proceeding with vector routing.");
        
        // Update local state
        let mut meta_guard = self.inner.meta_hnsw.write().await;
        *meta_guard = Some(meta_hnsw);
        drop(meta_guard); // Explicit drop
        info!("Clustering: MetaHNSW updated.");
        
        let mut clusters_guard = self.inner.node_clusters.write().await;
        if let Some(my_clusters) = assignment.get(&self.inner.id) {
            info!("Clustering: Assigned {} clusters to this node.", my_clusters.len());
            *clusters_guard = my_clusters.clone();
            // Create collections
            for cluster in my_clusters {
                let collection_name = format!("vectors_{}", cluster.id);
                info!("Clustering: Creating collection '{}'", collection_name);
                let _ = self.inner.qdrant.create_collection(&collection_name, self.inner.dimension as u64).await;
            }
        } else {
            warn!("Clustering: No clusters assigned to this node.");
        }
        drop(clusters_guard); // Explicit drop
        
        // Route buffered vectors (the ones we used for clustering)
        // We need to re-route them because they are not in Qdrant yet.
        // But wait, we just cleared the buffer.
        // We should route `vectors` now.
        info!("Clustering: Routing {} initial vectors...", vectors.len());
        self.route_vectors(vectors, req_id).await;
        
        // Route any vectors that arrived during clustering
        let mut buffer = self.inner.bootstrap_buffer.lock().await;
        let remaining_vectors = buffer.clone();
        buffer.clear();
        drop(buffer);
        
        if !remaining_vectors.is_empty() {
            info!("Clustering: Routing {} vectors that arrived during clustering...", remaining_vectors.len());
            self.route_vectors(remaining_vectors, req_id).await;
        }

        let mut status = self.inner.status.write().await;
        *status = ServerStatus::Clustered;
        info!("*** CLUSTERING FINISHED ***");
        histogram!("app_request_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "clustering");
    }

    // ... query and get_count ...

    async fn get_peer_client(&self, peer_id: &str, endpoint: &str) -> Result<NodeServiceClient<tonic::transport::Channel>, String> {
        {
            let clients = self.inner.peer_clients.read().await;
            if let Some(client) = clients.get(peer_id) {
                return Ok(client.clone());
            }
        }
        
        // Connect if not in cache
        let client = NodeServiceClient::connect(endpoint.to_string())
            .await
            .map_err(|e| e.to_string())?;
            
        let mut clients = self.inner.peer_clients.write().await;
        clients.insert(peer_id.to_string(), client.clone());
        
        Ok(client)
    }

    pub async fn route_vectors(&self, vectors: Vec<VectorWithPayload>, req_id: i64) {
        let start = std::time::Instant::now();
        let meta_guard = self.inner.meta_hnsw.read().await;
        if let Some(meta) = meta_guard.as_ref() {
            // Group vectors by target cluster
            let mut vectors_by_cluster: HashMap<usize, Vec<VectorWithPayload>> = HashMap::new();
            let mut dropped_count = 0;
            
            for vector in vectors {
                // Find nearest centroid
                let nearest_clusters = meta.search(&vector.vector, 1);
                if let Some(&cluster_idx) = nearest_clusters.first() {
                    vectors_by_cluster.entry(cluster_idx).or_default().push(vector);
                    counter!("app_cluster_hits_total", 1, "cluster_id" => cluster_idx.to_string());
                } else {
                    dropped_count += 1;
                }
            }
            
            if dropped_count > 0 {
                warn!("Routing: Dropped {} vectors because MetaHNSW returned no results!", dropped_count);
                counter!("app_vectors_dropped_total", dropped_count as u64, "reason" => "no_cluster_found");
            }

            let total_routed: usize = vectors_by_cluster.values().map(|v| v.len()).sum();
            info!("Routing: Routing {} vectors to {} clusters. (Dropped: {})", total_routed, vectors_by_cluster.len(), dropped_count);
            // We can't easily count vectors routed per target node here without iterating again or changing structure.
            // But we can count total routed.
            // Python counts per target node.
            // Let's try to approximate or just count total for now, as iterating again is expensive.
            // Actually, we iterate below. We can count there.
            // counter!("app_vectors_routed_total", total_routed as u64);
            
            let mut tasks = Vec::new();

            for (cluster_idx, cluster_vectors) in vectors_by_cluster {
                let cluster_id = cluster_idx.to_string(); // Assuming cluster ID is index
                let collection_name = format!("vectors_{}", cluster_id);
                
                // Check if I have this cluster
                let my_clusters = self.inner.node_clusters.read().await;
                let i_have_it = my_clusters.iter().any(|c| c.id == cluster_id);
                drop(my_clusters);
                
                if i_have_it {
                    // Retry local insertion with exponential backoff
                    let app = self.clone();
                    let collection_name = collection_name.clone();
                    let cluster_vectors = cluster_vectors.clone();
                    
                    tasks.push(tokio::spawn(async move {
                        let max_retries = 10;
                        let mut retries = 0;
                        
                        loop {
                            match app.inner.qdrant.insert_vectors(&collection_name, cluster_vectors.clone()).await {
                                Ok(_) => {
                                    if retries > 0 {
                                        info!("Successfully inserted vectors into local '{}' after {} retries", collection_name, retries);
                                    }
                                    break;
                                },
                                Err(e) if retries < max_retries => {
                                    let delay_ms = 100 * 2_u64.pow(retries as u32);
                                    warn!("Local insert into '{}' failed (retry {}/{}): {}. Retrying in {}ms...", 
                                          collection_name, retries + 1, max_retries, e, delay_ms);
                                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                                    retries += 1;
                                },
                                Err(e) => {
                                    error!("Failed to insert vectors into local collection '{}' after {} retries: {}", 
                                           collection_name, max_retries, e);
                                    break;
                                }
                            }
                        }
                    }));
                }
                
                // Send to peers
                let peers = self.inner.peers.read().await;
                let global_assignment = self.inner.global_assignment.read().await;
                
                for peer in peers.iter() {
                     let peer_id = peer.id.clone();
                     
                     // Check if peer owns this cluster
                     let should_send = if let Some(assigned_clusters) = global_assignment.get(&peer_id) {
                         assigned_clusters.contains(&cluster_id)
                     } else {
                         // Fallback to broadcast if assignment is missing (should not happen after clustering)
                         // But maybe we are in a state where we don't have assignment yet?
                         // If we are routing, we should have assignment.
                         warn!("Routing: No assignment found for peer {}. Defaulting to broadcast.", peer_id);
                         true
                     };
                     
                     if !should_send {
                         continue;
                     }

                     counter!("app_vectors_routed_total", cluster_vectors.len() as u64, "target_node" => peer_id.clone());

                     let endpoint = peer.grpc_url.clone();
                     let vectors_clone = cluster_vectors.clone();
                     let req_id_clone = req_id;
                     let request_type = format!("cluster_{}", cluster_id);
                     let endpoint = endpoint.clone();
                     let app = self.clone();
                     let cluster_id_clone = cluster_id.clone();
                     
                     tasks.push(tokio::spawn(async move {
                        let max_retries = 10;
                        let mut retries = 0;
                        
                        loop {
                            // 1. Get client from cache or connect
                            match app.get_peer_client(&peer_id, &endpoint).await {
                                Ok(mut client) => {
                                    let proto_vectors: Vec<VectorPoint> = vectors_clone.clone().into_iter().map(|v| v.into()).collect();
                                    let req = AddVectorsRequest {
                                        req_id: req_id_clone,
                                        content: proto_vectors,
                                        request_type: request_type.clone(),
                                    };
                                    
                                    // 2. Try to send
                                    match client.receive_vectors(req).await {
                                        Ok(_) => break, // Success
                                        Err(e) => {
                                            warn!("Send to peer {} failed (retry {}/{}): {}. Retrying...", 
                                                  peer_id, retries + 1, max_retries, e);
                                            counter!("app_peer_failures_total", 1, "peer_id" => peer_id.clone(), "operation" => "route_vectors");
                                        }
                                    }
                                },
                                Err(e) => {
                                    warn!("Connect to peer {} failed (retry {}/{}): {}. Retrying...", 
                                          peer_id, retries + 1, max_retries, e);
                                    counter!("app_peer_failures_total", 1, "peer_id" => peer_id.clone(), "operation" => "route_vectors_connect");
                                }
                            }
                            
                            retries += 1;
                            if retries >= max_retries {
                                error!("Failed to send vectors to peer {} after {} retries", peer_id, max_retries);
                                break;
                            }
                            
                            let delay_ms = 100 * 2_u64.pow(retries as u32);
                            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        }
                     }));
                }
            }
            
            // Wait for all routing tasks to complete
            if !tasks.is_empty() {
                // info!("Routing: Waiting for {} routing tasks to complete...", tasks.len());
                futures::future::join_all(tasks).await;
                // info!("Routing: All routing tasks completed.");
            }
        } else {
             error!("Routing: MetaHNSW is None! Dropping {} vectors.", vectors.len());
        }
        histogram!("app_request_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "route_vectors");
    }
    pub async fn query(&self, query_vec: Vector, topk: u64) -> Result<Vec<qdrant_client::qdrant::ScoredPoint>, Box<dyn std::error::Error + Send + Sync>> {
        let start = std::time::Instant::now();
        counter!("app_queries_received_total", 1, "query_type" => "global", "source" => "client");
        
        // 1. Get local results
        let local_results = self.search_local(query_vec.clone(), topk).await?;
        
        // 2. Broadcast to peers
        let peers = self.inner.peers.read().await;
        let mut peer_tasks = Vec::new();
        
        for peer in peers.iter() {
            let endpoint = peer.grpc_url.clone();
            let peer_id = peer.id.clone();
            let query_vec_clone = query_vec.clone();
            let app = self.clone();
            
            counter!("app_queries_routed_total", 1, "target_node" => peer_id.clone());

            counter!("app_queries_routed_total", 1, "target_node" => peer_id.clone());

            peer_tasks.push(tokio::spawn(async move {
                match app.get_peer_client(&peer_id, &endpoint).await {
                    Ok(mut client) => {
                        let req = QueryRequest {
                            req_id: 0, // TODO: generate ID
                            query_vectors: vec![VectorList {
                                values: query_vec_clone,
                                cluster_id: "".to_string(),
                            }],
                            topk: topk as i32,
                        };
                        
                        let start_peer = std::time::Instant::now();
                        match client.query_peer(req).await {
                            Ok(resp) => {
                                histogram!("app_peer_latency_seconds", start_peer.elapsed().as_secs_f64(), "peer_id" => peer_id.clone(), "operation" => "query");
                                Ok(resp.into_inner().results)
                            },
                            Err(e) => {
                                counter!("app_peer_failures_total", 1, "peer_id" => peer_id.clone(), "operation" => "query");
                                Err(format!("Query peer {} failed: {}", peer_id, e))
                            },
                        }
                    },
                    Err(e) => {
                        counter!("app_peer_failures_total", 1, "peer_id" => peer_id.clone(), "operation" => "query_connect");
                        Err(format!("Connect to peer {} failed: {}", peer_id, e))
                    },
                }
            }));
        }
        
        // 3. Collect and merge results
        let mut all_results = local_results;
        
        let peer_results = futures::future::join_all(peer_tasks).await;
        for res in peer_results {
            if let Ok(Ok(results)) = res {
                // Convert proto ScoredPoint back to qdrant ScoredPoint
                for sp in results {
                    let mut payload = std::collections::HashMap::new();
                    if let Some(p) = sp.payload {
                         payload.insert("payload".to_string(), qdrant_client::qdrant::Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(p.payload)),
                        });
                    }
                    
                    all_results.push(qdrant_client::qdrant::ScoredPoint {
                        id: Some(qdrant_client::qdrant::PointId {
                            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(sp.id as u64)),
                        }),
                        payload,
                        score: sp.score,
                        version: 0,
                        vectors: None,
                        order_value: None,
                        shard_key: None,
                    });
                }
            }
        }
        
        // 4. Deduplicate by ID
        let mut seen_ids = std::collections::HashSet::new();
        all_results.retain(|sp| {
            let id = sp.id.as_ref().map(|id| match id.point_id_options {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n,
                _ => 0,
            }).unwrap_or(0);
            seen_ids.insert(id)
        });

        // 5. Final sort and topk
        all_results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(topk as usize);
        
        histogram!("app_query_latency_seconds", start.elapsed().as_secs_f64(), "query_type" => "global");
        histogram!("app_request_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "global_query");
        Ok(all_results)
    }

    pub async fn get_count(&self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let status = self.inner.status.read().await.clone();
        match status {
            ServerStatus::Bootstrap | ServerStatus::Clustering => {
                let buffer = self.inner.bootstrap_buffer.lock().await;
                Ok(buffer.len() as u64)
            },
            ServerStatus::Clustered => {
                let clusters = self.inner.node_clusters.read().await;
                let mut total = 0;
                for cluster in clusters.iter() {
                    let collection_name = format!("vectors_{}", cluster.id);
                    if let Ok(count) = self.inner.qdrant.count(&collection_name).await {
                        total += count;
                    }
                }
                Ok(total)
            },
        }
    }

    pub async fn search_local(&self, query_vec: Vector, topk: u64) -> Result<Vec<qdrant_client::qdrant::ScoredPoint>, Box<dyn std::error::Error + Send + Sync>> {
        let start = std::time::Instant::now();
        let status = self.inner.status.read().await.clone();
        match status {
            ServerStatus::Bootstrap | ServerStatus::Clustering => {
                if self.i_am_coord() {
                    // Linear search in buffer
                    let buffer = self.inner.bootstrap_buffer.lock().await;
                    let mut scored_points: Vec<qdrant_client::qdrant::ScoredPoint> = buffer.iter().map(|v| {
                        let score = cosine_similarity(&v.vector, &query_vec);
                        let mut payload = std::collections::HashMap::new();
                        payload.insert("payload".to_string(), qdrant_client::qdrant::Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(v.payload.clone())),
                        });
                        
                        qdrant_client::qdrant::ScoredPoint {
                            id: Some(qdrant_client::qdrant::PointId {
                                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(v.id as u64)),
                            }),
                            payload,
                            score,
                            version: 0,
                            vectors: None,
                            order_value: None,
                            shard_key: None,
                        }
                    }).collect();
                    
                    scored_points.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    scored_points.truncate(topk as usize);
                    Ok(scored_points)
                } else {
                    Ok(Vec::new())
                }
            },
            ServerStatus::Clustered => {
                // Search in all local cluster collections
                let clusters = self.inner.node_clusters.read().await;
                let mut all_results: Vec<qdrant_client::qdrant::ScoredPoint> = Vec::new();
                for cluster in clusters.iter() {
                    let collection_name = format!("vectors_{}", cluster.id);
                    // Ignore errors if collection doesn't exist (might be empty)
                    if let Ok(results) = self.inner.qdrant.search(&collection_name, query_vec.clone(), topk).await {
                        all_results.extend(results);
                    }
                }
                // Sort and take topk
                all_results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                all_results.truncate(topk as usize);
                Ok(all_results)
            },
        }
    }
}

#[tonic::async_trait]
impl NodeService for ServerApp {
    async fn set_clusters(&self, request: Request<SetClustersRequest>) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        
        // Update MetaHNSW
        if !req.meta_hnsw.is_empty() {
            if let Ok(meta) = MetaHNSW::from_bytes(&req.meta_hnsw, self.inner.dimension) {
                let mut meta_guard = self.inner.meta_hnsw.write().await;
                *meta_guard = Some(meta);
                info!("Received and updated MetaHNSW.");
            }
        }
        
        // Update Global Assignment
        if !req.global_assignment.is_empty() {
            let mut assignment_map: HashMap<String, Vec<String>> = HashMap::new();
            for ga in req.global_assignment {
                assignment_map.insert(ga.node_id, ga.cluster_ids);
            }
            let mut ga_guard = self.inner.global_assignment.write().await;
            *ga_guard = assignment_map;
            info!("Received and updated Global Assignment.");
        }
        
        // Update Clusters and Create Collections
        let mut clusters_guard = self.inner.node_clusters.write().await;
        let mut new_clusters = Vec::new();
        for c in req.clusters {
            new_clusters.push(Cluster {
                id: c.id.clone(),
                centroid: c.centroid,
                vector_ids: Vec::new(), // Empty, we don't send vector IDs in set_clusters usually
            });
            
            let collection_name = format!("vectors_{}", c.id);
            info!("SetClusters: Creating collection '{}'", collection_name);
            if let Err(e) = self.inner.qdrant.create_collection(&collection_name, self.inner.dimension as u64).await {
                error!("SetClusters: Failed to create collection '{}': {}", collection_name, e);
            }
        }
        *clusters_guard = new_clusters;
        
        // Set status to Clustered
        let mut status = self.inner.status.write().await;
        *status = ServerStatus::Clustered;
        
        Ok(Response::new(Empty {}))
    }
    
    async fn receive_vectors(&self, request: Request<AddVectorsRequest>) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let vectors: Vec<VectorWithPayload> = req.content.into_iter().map(|v| v.into()).collect();
        
        counter!("app_requests_total", 1, "operation" => "receive_vectors");
        
        if req.request_type.starts_with("cluster_") {
            // Direct insertion into cluster collection with retry logic
            let cluster_id = req.request_type.strip_prefix("cluster_").unwrap();
            let collection_name = format!("vectors_{}", cluster_id);
            
            // Wait for clustering to finish and collection to be created
            // set_clusters holds write lock, so this read lock will wait
            {
                let clusters = self.inner.node_clusters.read().await;
                if !clusters.iter().any(|c| c.id == cluster_id) {
                    // This node is not responsible for this cluster.
                    // Since the coordinator broadcasts to all peers, this is expected.
                    return Ok(Response::new(Empty {}));
                }
            }

            // Retry up to 10 times with exponential backoff
            let max_retries = 10;
            let mut retries = 0;
            
            loop {
                match self.inner.qdrant.insert_vectors(&collection_name, vectors.clone()).await {
                    Ok(_) => {
                        if retries > 0 {
                            info!("Successfully inserted vectors into '{}' after {} retries", collection_name, retries);
                        }
                        break;
                    },
                    Err(e) => {


                        if retries < max_retries {
                            let delay_ms = 100 * 2_u64.pow(retries as u32);
                            warn!("Insert into '{}' failed (retry {}/{}): {}. Retrying in {}ms...", 
                                  collection_name, retries + 1, max_retries, e, delay_ms);
                            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                            retries += 1;
                        } else {
                            error!("Failed to insert vectors into '{}' after {} retries: {}", 
                                   collection_name, max_retries, e);
                            break;
                        }
                    }
                }
            }
        } else {
            // Bootstrap/Normal flow
            let _ = self.inner.vector_queue.send((vectors, req.request_type, req.req_id)).await;
        }
        
        Ok(Response::new(Empty {}))
    }

    async fn query_peer(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        if req.query_vectors.is_empty() {
             return Ok(Response::new(QueryResponse { results: vec![] }));
        }
        
        // Convert proto VectorList to Vec<f32>
        let query_vec = req.query_vectors[0].values.clone();
        
        let result = match self.search_local(query_vec, req.topk as u64).await {
            Ok(results) => {
                let proto_results: Vec<ScoredPoint> = results.into_iter().map(|sp| {
                    // Convert qdrant ScoredPoint to proto ScoredPoint
                     ScoredPoint {
                        id: sp.id.map(|id| match id.point_id_options {
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n as i64,
                            _ => 0,
                        }).unwrap_or(0),
                        score: sp.score,
                        payload: Some(VectorPoint {
                             vector: vec![], // Optimization: don't return vector
                             id: 0, 
                             payload: sp.payload.get("payload").and_then(|v| match &v.kind {
                                 Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
                                 _ => None,
                             }).unwrap_or_default(),
                             cluster: "".to_string(),
                        }),
                    }
                }).collect();
                Ok(Response::new(QueryResponse { results: proto_results }))
            },
            Err(e) => {
                error!("Error in query_peer: {}", e);
                Ok(Response::new(QueryResponse { results: vec![] }))
            }
        };
        histogram!("app_peer_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "query");
        result
    }

    async fn register_peers(&self, request: Request<RegisterPeersRequest>) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let mut peers = self.inner.peers.write().await;
        for p in req.peers {
            peers.push(Peer {
                id: p.id,
                http_url: p.http_url,
                grpc_url: p.grpc_url,
            });
        }
        Ok(Response::new(Empty {}))
    }
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
