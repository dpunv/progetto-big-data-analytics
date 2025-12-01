use clap::Parser;
use rust_p2p_analytics::server::ServerApp;
use rust_p2p_analytics::p2p::node_service_server::NodeServiceServer;
use rust_p2p_analytics::{VectorWithPayload, Vector};
use rust_p2p_analytics::models::Peer;
use tonic::transport::Server;
use std::net::SocketAddr;
use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
    http::StatusCode,
};
use std::sync::Arc;
use tracing_subscriber;
use serde::{Deserialize, Serialize};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::future::ready;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    node_name: String,
    #[arg(long)]
    node_url: String,
    #[arg(long)]
    node_grpc_url: String,
    #[arg(long)]
    qdrant_url: String,
    #[arg(long)]
    coordinator_url: String,
    #[arg(long, default_value_t = 1)]
    replicas: usize,
    #[arg(long, default_value_t = 10000)]
    num_before_clustering: usize,
    #[arg(long, default_value_t = 384)]
    dimension: usize,
}

#[derive(Deserialize)]
struct AddVectorsPayload {
    id: i64,
    content: Vec<(Vector, i64, String, String)>, // vector, id, payload, cluster
}

#[derive(Deserialize)]
struct RegisterPeersPayload {
    id: i64,
    peers: Vec<(String, String, String)>, // id, http_url, grpc_url
}

#[derive(Deserialize)]
struct QueryPayload {
    id: i64,
    query: Vec<Vector>,
    topk: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let app = ServerApp::new(
        args.node_name.clone(),
        args.node_url.clone(),
        args.node_grpc_url.clone(),
        args.qdrant_url.clone(),
        args.coordinator_url.clone(),
        args.replicas,
        args.num_before_clustering,
        args.dimension,
    ).await;

    // Initialize Prometheus
    let builder = PrometheusBuilder::new();
    let handle = builder.install_recorder().expect("failed to install recorder");


    // Start gRPC Server
    let grpc_addr: SocketAddr = args.node_grpc_url.replace("http://", "").replace("https://", "").parse()?;
    let grpc_app = app.clone();
    
    tokio::spawn(async move {
        println!("gRPC server listening on {}", grpc_addr);
        Server::builder()
            .add_service(NodeServiceServer::new(grpc_app))
            .serve(grpc_addr)
            .await
            .unwrap();
    });

    // Start HTTP Server
    let http_addr: SocketAddr = args.node_url.replace("http://", "").replace("https://", "").parse()?;
    
    let app_router = Router::new()
        .route("/", get(health_check))
        .route("/add", post(add_vectors))
        .route("/register_peers", post(register_peers))
        .route("/query", post(query))
        .route("/count", get(get_count))
        .route("/metrics", get(move || ready(handle.render())))
        .with_state(app)
        .layer(axum::extract::DefaultBodyLimit::max(500 * 1024 * 1024));

    println!("HTTP server listening on {}", http_addr);
    let listener = tokio::net::TcpListener::bind(http_addr).await.unwrap();
    axum::serve(listener, app_router).await.unwrap();

    Ok(())
}

async fn health_check() -> &'static str {
    "healthy"
}

async fn query(
    State(app): State<ServerApp>,
    Json(payload): Json<QueryPayload>,
) -> Json<serde_json::Value> {
    if payload.query.is_empty() {
        return Json(serde_json::json!({"results": []}));
    }

    let query_vec = payload.query[0].clone();
    
    match app.query(query_vec, payload.topk as u64).await {
        Ok(results) => {
            let serializable_results: Vec<SerializableScoredPoint> = results.into_iter().map(|sp| sp.into()).collect();
            Json(serde_json::json!({"results": serializable_results}))
        },
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

async fn get_count(State(app): State<ServerApp>) -> Json<serde_json::Value> {
    match app.get_count().await {
        Ok(count) => Json(serde_json::json!({"count": count})),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

#[derive(Serialize)]
struct SerializableScoredPoint {
    id: u64,
    version: u64,
    score: f32,
    payload: Option<serde_json::Value>,
    vector: Option<Vec<f32>>,
}

fn qdrant_value_to_json(value: qdrant_client::qdrant::Value) -> serde_json::Value {
    match value.kind {
        Some(qdrant_client::qdrant::value::Kind::NullValue(_)) => serde_json::Value::Null,
        Some(qdrant_client::qdrant::value::Kind::DoubleValue(v)) => serde_json::json!(v),
        Some(qdrant_client::qdrant::value::Kind::IntegerValue(v)) => serde_json::json!(v),
        Some(qdrant_client::qdrant::value::Kind::StringValue(v)) => serde_json::json!(v),
        Some(qdrant_client::qdrant::value::Kind::BoolValue(v)) => serde_json::json!(v),
        Some(qdrant_client::qdrant::value::Kind::StructValue(v)) => {
            let mut map = serde_json::Map::new();
            for (k, v) in v.fields {
                map.insert(k, qdrant_value_to_json(v));
            }
            serde_json::Value::Object(map)
        },
        Some(qdrant_client::qdrant::value::Kind::ListValue(v)) => {
            let list: Vec<serde_json::Value> = v.values.into_iter().map(qdrant_value_to_json).collect();
            serde_json::Value::Array(list)
        },
        None => serde_json::Value::Null,
    }
}

impl From<qdrant_client::qdrant::ScoredPoint> for SerializableScoredPoint {
    fn from(sp: qdrant_client::qdrant::ScoredPoint) -> Self {
        let payload_map: serde_json::Map<String, serde_json::Value> = sp.payload.into_iter()
            .map(|(k, v)| (k, qdrant_value_to_json(v)))
            .collect();

        SerializableScoredPoint {
            id: sp.id.map(|id| match id.point_id_options {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n,
                _ => 0,
            }).unwrap_or(0),
            version: sp.version,
            score: sp.score,
            payload: Some(serde_json::Value::Object(payload_map)),
            vector: None,
        }
    }
}

async fn add_vectors(
    State(app): State<ServerApp>,
    Json(payload): Json<AddVectorsPayload>,
) -> StatusCode {
    let vectors: Vec<VectorWithPayload> = payload.content.into_iter().map(|(vec, id, payload, cluster)| {
        VectorWithPayload {
            vector: vec,
            id: id as i64,
            payload,
            cluster,
        }
    }).collect();

    if let Err(e) = app.vector_queue.send((vectors, "bootstrap".to_string(), payload.id)).await {
        eprintln!("Error sending to queue: {}", e);
        return StatusCode::INTERNAL_SERVER_ERROR;
    }

    StatusCode::OK
}

async fn register_peers(
    State(app): State<ServerApp>,
    Json(payload): Json<RegisterPeersPayload>,
) -> StatusCode {
    let mut peers = app.peers.write().await;
    for (id, http, grpc) in payload.peers {
        // Avoid duplicates
        if !peers.iter().any(|p| p.id == id) {
            peers.push(Peer {
                id,
                http_url: http,
                grpc_url: grpc,
            });
        }
    }
    StatusCode::OK
}


