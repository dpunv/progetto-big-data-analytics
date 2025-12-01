use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    pub id: String,
    pub http_url: String,
    pub grpc_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerStatus {
    Bootstrap,
    Clustering,
    Clustered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub id: String,
    pub centroid: Vec<f32>,
    pub vector_ids: Vec<i64>,
}
