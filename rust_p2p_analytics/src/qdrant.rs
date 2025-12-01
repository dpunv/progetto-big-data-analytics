use qdrant_client::qdrant::{
    PointStruct, Vectors, Vector, Value, with_payload_selector::SelectorOptions,
    SearchPoints, ScoredPoint, CreateCollection, VectorParams, Distance,
    CountPoints,
};
use qdrant_client::client::QdrantClient;
use anyhow::Result;
use crate::VectorWithPayload;
use std::collections::HashMap;
use std::sync::Arc;
use metrics::{histogram, counter};

#[derive(Clone)]
pub struct QdrantHandler {
    client: Arc<QdrantClient>,
}

impl QdrantHandler {
    pub fn new(url: String) -> Result<Self> {
        let client = QdrantClient::from_url(&url).build()?;
        Ok(Self { client: Arc::new(client) })
    }

    pub async fn create_collection(&self, name: &str, dim: u64) -> Result<()> {
        let start = std::time::Instant::now();
        tracing::info!("Qdrant: Creating collection '{}'", name);
        if self.client.collection_exists(name).await? {
            tracing::info!("Qdrant: Collection '{}' already exists", name);
            return Ok(());
        }
        
        match self.client.create_collection(&CreateCollection {
            collection_name: name.to_string(),
            vectors_config: Some(qdrant_client::qdrant::VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                    VectorParams {
                        size: dim,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    }
                ))
            }),
            ..Default::default()
        }).await {
            Ok(_) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "create_collection", "status" => "success");
                tracing::info!("Qdrant: Successfully created collection '{}'", name);
                Ok(())
            },
            Err(e) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "create_collection", "status" => "error");
                counter!("app_db_errors_total", 1, "operation" => "create_collection");
                tracing::error!("Qdrant: Failed to create collection '{}': {}", name, e);
                Err(e.into())
            }
        }
    }

    pub async fn insert_vectors(&self, collection_name: &str, vectors: Vec<VectorWithPayload>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let start = std::time::Instant::now();
        let points: Vec<PointStruct> = vectors.into_iter().map(|v| {
            let mut payload = HashMap::new();
            payload.insert("payload".to_string(), Value::from(v.payload));
            payload.insert("cluster".to_string(), Value::from(v.cluster));

            PointStruct::new(
                v.id as u64,
                v.vector,
                payload
            )
        }).collect();

        let count = points.len();
        // Adding timeout to detect hanging
        match tokio::time::timeout(std::time::Duration::from_secs(30), self.client.upsert_points(collection_name, None, points, None)).await {
            Ok(result) => match result {
                Ok(_) => {
                    histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "insert_vectors", "status" => "success");
                    counter!("qdrant_vectors_inserted_total", count as u64, "collection" => collection_name.to_string());
                    tracing::debug!("Qdrant: Inserted {} vectors into '{}'", count, collection_name);
                    Ok(())
                },
                Err(e) => {
                    histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "insert_vectors", "status" => "error");
                    counter!("app_db_errors_total", 1, "operation" => "insert_vectors");
                    tracing::error!("Qdrant: Failed to insert {} vectors into '{}': {}", count, collection_name, e);
                    Err(e.into())
                }
            },
            Err(_) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "insert_vectors", "status" => "timeout");
                counter!("app_db_errors_total", 1, "operation" => "insert_vectors_timeout");
                tracing::error!("Qdrant: Timed out inserting {} vectors into '{}'", count, collection_name);
                Err(anyhow::anyhow!("Qdrant insert timed out"))
            }
        }
    }

    pub async fn search(&self, collection_name: &str, vector: Vec<f32>, topk: u64) -> Result<Vec<ScoredPoint>> {
        let start = std::time::Instant::now();
        let search_result = self.client.search_points(&SearchPoints {
            collection_name: collection_name.to_string(),
            vector: vector,
            limit: topk,
            with_payload: Some(true.into()),
            ..Default::default()
        }).await;

        match search_result {
            Ok(res) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "search", "status" => "success");
                Ok(res.result)
            },
            Err(e) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "search", "status" => "error");
                counter!("app_db_errors_total", 1, "operation" => "search");
                Err(e.into())
            }
        }
    }

    pub async fn count(&self, collection_name: &str) -> Result<u64> {
        let start = std::time::Instant::now();
        let result = self.client.count(&CountPoints {
            collection_name: collection_name.to_string(),
            ..Default::default()
        }).await;

        match result {
            Ok(res) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "count", "status" => "success");
                Ok(res.result.unwrap_or_default().count)
            },
            Err(e) => {
                histogram!("app_db_latency_seconds", start.elapsed().as_secs_f64(), "operation" => "count", "status" => "error");
                counter!("app_db_errors_total", 1, "operation" => "count");
                Err(e.into())
            }
        }
    }
}
