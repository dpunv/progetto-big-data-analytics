pub mod p2p {
    tonic::include_proto!("p2p");
}

pub mod clustering;
pub mod server;
pub mod qdrant;
pub mod models;

use p2p::VectorPoint;

// Common types
pub type Vector = Vec<f32>;
pub type VectorId = i64;

#[derive(Debug, Clone)]
pub struct VectorWithPayload {
    pub vector: Vector,
    pub id: VectorId,
    pub payload: String,
    pub cluster: String,
}

impl From<VectorPoint> for VectorWithPayload {
    fn from(vp: VectorPoint) -> Self {
        VectorWithPayload {
            vector: vp.vector,
            id: vp.id,
            payload: vp.payload,
            cluster: vp.cluster,
        }
    }
}

impl From<VectorWithPayload> for VectorPoint {
    fn from(v: VectorWithPayload) -> Self {
        VectorPoint {
            vector: v.vector,
            id: v.id,
            payload: v.payload,
            cluster: v.cluster,
        }
    }
}
