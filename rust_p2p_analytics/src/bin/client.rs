use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tokio::sync::Mutex;
use reqwest::Client;
use std::time::{Duration, Instant};
use rust_p2p_analytics::{VectorWithPayload, Vector};
use futures::stream::{self, StreamExt};

#[derive(Deserialize, Debug)]
struct Config {
    servers: Vec<ServerConfig>,
    num_vectors: usize,
    batch_size: usize,
    batch_size_retry: usize,
    replicas: usize,
    num_before_clustering: usize,
    max_retries: usize,
    embedding_file: String,
}

#[derive(Deserialize, Debug, Clone)]
struct ServerConfig {
    id: String,
    url: String,
    grpc_url: String,
    is_coordinator: bool,
}

#[derive(Deserialize, Debug)]
struct EmbeddingData {
    embedding: Vector,
    text: String,
}

#[derive(Serialize)]
struct RegisterPeersRequest {
    id: i64,
    peers: Vec<(String, String, String)>,
}

#[derive(Serialize)]
struct AddVectorsRequest {
    id: i64,
    content: Vec<(Vector, i64, String, String)>, // vector, id, payload, cluster
}

#[derive(Serialize)]
struct QueryRequest {
    id: i64,
    query: Vec<Vector>,
    topk: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    // Load config
    let config_file = File::open("config.json")?;
    let reader = BufReader::new(config_file);
    let config: Config = serde_json::from_reader(reader)?;
    
    // Load data
    println!("Loading embeddings...");
    let data_file = File::open(config.embedding_file)?;
    let reader = BufReader::new(data_file);
    let data: Vec<EmbeddingData> = serde_json::from_reader(reader)?;
    println!("Loaded {} embeddings", data.len());

    let client = Client::new();
    
    // Register peers
    for server in &config.servers {
        let peers: Vec<(String, String, String)> = config.servers.iter()
            .filter(|s| s.id != server.id)
            .map(|s| (s.id.clone(), s.url.clone(), s.grpc_url.clone()))
            .collect();
            
        let req = RegisterPeersRequest {
            id: 1,
            peers,
        };
        
        let _ = client.post(format!("{}/register_peers", server.url))
            .json(&req)
            .send()
            .await?;
    }
    println!("Peers registered");

    // Send vectors
    let num_to_send = config.num_vectors.min(data.len());
    let batch_size = config.batch_size;
    let total_batches = (num_to_send + batch_size - 1) / batch_size;
    
    let mut batches = Vec::new();
    for i in 0..total_batches {
        let start = i * batch_size;
        let end = (start + batch_size).min(num_to_send);
        let batch_data = &data[start..end];
        
        // Convert to payload format
        // We need unique IDs.
        let batch_content: Vec<(Vector, i64, String, String)> = batch_data.iter().enumerate().map(|(idx, d)| {
            (d.embedding.clone(), (start + idx) as i64, d.text.clone(), "-1".to_string())
        }).collect();
        
        let target_server = &config.servers[i % config.servers.len()];
        batches.push((target_server.clone(), batch_content, i));
    }

    println!("Sending {} batches...", batches.len());
    let start_time = Instant::now();

    let results = stream::iter(batches)
        .map(|(server, batch, batch_num)| {
            let client = client.clone();
            async move {
                let start = Instant::now();
                let req = AddVectorsRequest {
                    id: batch_num as i64,
                    content: batch,
                };
                
                match client.post(format!("{}/add", server.url))
                    .json(&req)
                    .send()
                    .await {
                        Ok(res) => {
                            if res.status().is_success() {
                                Ok(start.elapsed())
                            } else {
                                Err(format!("Server error: {}", res.status()))
                            }
                        },
                        Err(e) => Err(e.to_string())
                    }
            }
        })
        .buffer_unordered(10) // Parallelism
        .collect::<Vec<_>>()
        .await;

    let elapsed = start_time.elapsed();
    println!("Sent vectors in {:.2?}", elapsed);
    
    // Calculate statistics
    let mut latencies: Vec<f64> = results.iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|d| d.as_secs_f64())
        .collect();
    
    if !latencies.is_empty() {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = latencies.first().unwrap();
        let max = latencies.last().unwrap();
        let sum: f64 = latencies.iter().sum();
        let avg = sum / latencies.len() as f64;
        
        println!("\n--- Statistics ---");
        println!("Total Requests: {}", latencies.len());
        println!("Min Latency: {:.4}s", min);
        println!("Max Latency: {:.4}s", max);
        println!("Avg Latency: {:.4}s", avg);
        println!("------------------\n");
    }
    
    // Check for errors
    let errors: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
    if !errors.is_empty() {
        println!("{} batches failed", errors.len());
        if let Some(Err(e)) = errors.first() {
            println!("First error: {}", e);
        }
    }

    // Wait for clustering to finish
    println!("Waiting for vectors to be saved...");
    tokio::time::sleep(Duration::from_secs(10)).await; // Give some time for clustering to start
    
    let mut retries = 0;
    let mut last_count = 0;
    let mut stable_count = 0;
    let expected_count = num_to_send as u64 * config.replicas as u64;
    
    loop {
        let mut total_count = 0;
        for server in &config.servers {
            if let Ok(res) = client.get(format!("{}/count", server.url)).send().await {
                if let Ok(json) = res.json::<serde_json::Value>().await {
                    if let Some(count) = json.get("count").and_then(|c| c.as_u64()) {
                        total_count += count;
                    }
                }
            }
        }
        
        if total_count >= expected_count {
             println!("Clustering finished. Total vectors: {}/{}", total_count, expected_count);
             break;
        }
        
        if total_count == last_count {
            stable_count += 1;
        } else {
            stable_count = 0;
        }
        last_count = total_count;
        
        if retries > 300 { // Wait up to 300 seconds
            println!("Timeout waiting for clustering. Current count: {}/{}", total_count, expected_count);
            break;
        }
        
        if retries % 5 == 0 {
             println!("Waiting for vectors to be indexed... Current count: {}/{}", total_count, expected_count);
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        retries += 1;
    }

    // Wait for clustering to finish
    //println!("Waiting for clustering to finish...");
    //tokio::time::sleep(Duration::from_secs(100)).await; // Give some time for clustering to start
    
    // Count
    println!("Counting vectors...");
    let mut total_count = 0;
    for server in &config.servers {
        let res = client.get(format!("{}/count", server.url))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;
        
        if let Some(count) = res.get("count").and_then(|c| c.as_u64()) {
            println!("Node {}: {}", server.id, count);
            total_count += count;
        } else {
            println!("Node {}: Error getting count: {:?}", server.id, res);
        }
    }
    println!("Total vectors: {}", total_count);

    // Query
    println!("Querying...");
    let query_vec = data[0].embedding.clone();
    let query_req = QueryRequest {
        id: 999,
        query: vec![query_vec.clone()],
        topk: 5,
    };
    
    let res = client.post(format!("{}/query", config.servers[0].url))
        .json(&query_req)
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;
        
    println!("Query Result: {:?}", res);

    // Verify results with linear search
    println!("\n--- Verification ---");
    println!("Calculating true top-k using linear search...");
    
    // Only consider vectors that were actually sent
    let sent_data = &data[0..num_to_send];
    let mut scored_points: Vec<(f32, i64, String)> = sent_data.iter().enumerate().map(|(idx, d)| {
        let score = cosine_similarity(&query_vec, &d.embedding);
        (score, idx as i64, d.text.clone())
    }).collect();
    
    // Sort by score descending
    scored_points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let top_k_expected = &scored_points[0..query_req.topk.min(scored_points.len())];
    
    println!("\nExpected Top-{}:", query_req.topk);
    for (i, (score, id, text)) in top_k_expected.iter().enumerate() {
        println!("{}. ID: {}, Score: {:.4}, Text: {}", i + 1, id, score, text);
    }
    
    println!("\nActual Results:");
    if let Some(results) = res.get("results").and_then(|r| r.as_array()) {
        for (i, result) in results.iter().enumerate() {
            let id = result.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
            let score = result.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let payload = result.get("payload").and_then(|p| p.get("payload")).and_then(|v| v.as_str()).unwrap_or("");
            println!("{}. ID: {}, Score: {:.4}, Text: {}", i + 1, id, score, payload);
        }
        
        // Simple validation check (checking IDs)
        let actual_ids: Vec<u64> = results.iter()
            .map(|r| r.get("id").and_then(|v| v.as_u64()).unwrap_or(0))
            .collect();
            
        let expected_ids: Vec<u64> = top_k_expected.iter().map(|(_, id, _)| *id as u64).collect();
        
        if actual_ids == expected_ids {
             println!("\n✅ Verification PASSED: Results match expected IDs.");
        } else {
             println!("\n❌ Verification FAILED: Results do not match expected IDs.");
             println!("Expected IDs: {:?}", expected_ids);
             println!("Actual IDs:   {:?}", actual_ids);
        }

    } else {
        println!("No results found in response.");
    }

    Ok(())
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_v1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_v2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        0.0
    } else {
        dot_product / (norm_v1 * norm_v2)
    }
}
