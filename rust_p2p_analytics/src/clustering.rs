use ndarray::{Array2, Axis, ArrayView1};
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use hnsw_rs::prelude::*;
use anyhow::Result;
use crate::{Vector, VectorId, VectorWithPayload};
use crate::models::{Cluster, Peer};
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
pub struct MetaHNSW {
    #[serde(skip)] // We handle serialization manually or recreate it
    pub index: Option<Hnsw<'static, f32, DistCosine>>,
    pub dimension: usize,
    pub data: Vec<(VectorId, Vector)>, // Store centroids to rebuild if needed
}

impl MetaHNSW {
    pub fn new(dimension: usize) -> Self {
        Self {
            index: None,
            dimension,
            data: Vec::new(),
        }
    }

    pub fn build(&mut self, clusters: &Vec<Cluster>) {
        let max_elements = clusters.len();
        let hnsw = Hnsw::new(
            max_elements, 
            self.dimension, 
            16, // M
            200, // ef_construction
            DistCosine
        );

        let mut data = Vec::new();
        for (i, cluster) in clusters.iter().enumerate() {
            // We use cluster ID as index ID if possible, but HNSW needs usize or u64?
            // hnsw_rs uses usize for data point index.
            // We map internal ID to cluster ID string.
            // Actually, let's just use the index in the list.
            hnsw.insert((&cluster.centroid, i));
            data.push((i as i64, cluster.centroid.clone()));
        }
        
        self.index = Some(hnsw);
        self.data = data;
    }

    pub fn search(&self, query: &Vector, k: usize) -> Vec<usize> {
        if let Some(index) = &self.index {
            let res = index.search(query, k, 50); // ef_search
            res.iter().map(|n| n.d_id).collect()
        } else {
            vec![]
        }
    }
    
    // For serialization, we might just serialize the data and rebuild on the other side
    // because hnsw_rs serialization might be complex to pass via proto bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self.data).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Result<Self> {
        let data: Vec<(VectorId, Vector)> = bincode::deserialize(bytes)?;
        let mut meta = MetaHNSW::new(dimension);
        
        let max_elements = data.len();
        let hnsw = Hnsw::new(max_elements, dimension, 16, 200, DistCosine);
        
        for (id, vec) in &data {
            hnsw.insert((vec, *id as usize));
        }
        
        meta.index = Some(hnsw);
        meta.data = data;
        Ok(meta)
    }
}

pub fn perform_kmeans_auto_k(vectors: &Vec<VectorWithPayload>, min_k: usize, max_k: usize) -> Vec<Cluster> {
    println!("perform_kmeans_auto_k: Starting auto-k selection from {} to {}", min_k, max_k);
    
    let mut best_score = -1.0;
    let mut best_k = min_k;
    let mut best_clusters = Vec::new();
    
    for k in min_k..=max_k {
        let clusters = perform_kmeans(vectors, k);
        if clusters.len() < 2 {
            continue;
        }
        
        let score = calculate_silhouette_score(vectors, &clusters);
        println!("perform_kmeans_auto_k: k={} -> Silhouette Score: {:.4}", k, score);
        
        if score > best_score {
            best_score = score;
            best_k = k;
            best_clusters = clusters;
        }
    }
    
    println!("perform_kmeans_auto_k: Best k={} with score {:.4}", best_k, best_score);
    best_clusters
}

fn calculate_silhouette_score(vectors: &Vec<VectorWithPayload>, clusters: &Vec<Cluster>) -> f32 {
    if clusters.len() < 2 || vectors.is_empty() {
        return -1.0;
    }

    // Sampling if too many vectors
    let max_samples = 2000;
    let (sampled_vectors, use_sampling) = if vectors.len() > max_samples {
        let mut rng = thread_rng();
        let sampled: Vec<&VectorWithPayload> = vectors.choose_multiple(&mut rng, max_samples).collect();
        (sampled, true)
    } else {
        (vectors.iter().collect(), false)
    };

    // Pre-compute cluster centroids map for faster lookup
    let mut vector_to_cluster: HashMap<i64, usize> = HashMap::new();
    for (c_idx, cluster) in clusters.iter().enumerate() {
        for v_id in &cluster.vector_ids {
            vector_to_cluster.insert(*v_id, c_idx);
        }
    }

    // Parallel calculation of silhouette coefficient for each sample
    let scores: Vec<f32> = sampled_vectors.par_iter().map(|&vector| {
        let u_cluster_idx = match vector_to_cluster.get(&vector.id) {
            Some(&idx) => idx,
            None => return -1.0, // Should not happen
        };
        
        // a(i): Mean distance to other points in the same cluster
        let mut a_dist_sum = 0.0;
        let mut a_count = 0;
        
        // b(i): Min mean distance to points in any other cluster
        let mut b_min_mean_dist = f32::MAX;
        
        // Optimization: If using sampling, we only compare against sampled vectors?
        // No, silhouette is defined against the cluster.
        // If we sample, we should probably compare against all other vectors in the sample?
        // Yes, standard approximation is to calculate silhouette on the sample using distances to other points in the sample.
        
        let mut cluster_dists: HashMap<usize, (f32, usize)> = HashMap::new(); // cluster_idx -> (sum_dist, count)
        
        for &other in &sampled_vectors {
            if vector.id == other.id {
                continue;
            }
            
            let dist = squared_euclidean(&vector.vector, &other.vector).sqrt(); // Euclidean distance
            
            // We need to know the cluster of 'other'.
            // If 'other' is in sampled_vectors, we can look it up.
            if let Some(&other_cluster_idx) = vector_to_cluster.get(&other.id) {
                if other_cluster_idx == u_cluster_idx {
                    a_dist_sum += dist;
                    a_count += 1;
                } else {
                    let entry = cluster_dists.entry(other_cluster_idx).or_insert((0.0, 0));
                    entry.0 += dist;
                    entry.1 += 1;
                }
            }
        }
        
        let a = if a_count > 0 { a_dist_sum / a_count as f32 } else { 0.0 };
        
        for (_, (sum, count)) in cluster_dists {
            if count > 0 {
                let mean_dist = sum / count as f32;
                if mean_dist < b_min_mean_dist {
                    b_min_mean_dist = mean_dist;
                }
            }
        }
        
        if b_min_mean_dist == f32::MAX {
            return 0.0;
        }
        
        let max_ab = a.max(b_min_mean_dist);
        if max_ab == 0.0 {
            0.0
        } else {
            (b_min_mean_dist - a) / max_ab
        }
    }).collect();

    let total_score: f32 = scores.iter().sum();
    total_score / sampled_vectors.len() as f32
}

pub fn perform_kmeans(vectors: &Vec<VectorWithPayload>, k: usize) -> Vec<Cluster> {
    // println!("perform_kmeans: Starting with {} vectors, k={}", vectors.len(), k);
    if vectors.is_empty() {
        return vec![];
    }
    
    let dim = vectors[0].vector.len();
    let n_samples = vectors.len();
    let k = k.min(n_samples);
    
    // println!("perform_kmeans: Dim={}, Samples={}, K={}", dim, n_samples, k);

    let data: Vec<&Vec<f32>> = vectors.iter().map(|v| &v.vector).collect();
    
    // Initialize centroids randomly
    let mut rng = thread_rng();
    let mut centroids: Vec<Vec<f32>> = data.choose_multiple(&mut rng, k)
        .map(|v| (*v).clone())
        .collect();
        
    // println!("perform_kmeans: Centroids initialized.");

    let max_iterations = 100;
    let tolerance = 1e-4;
    
    let mut assignments = vec![0; n_samples];
    
    for _iter in 0..max_iterations {
        // if iter % 10 == 0 {
        //     println!("perform_kmeans: Iteration {}", iter);
        // }
        // Assignment step
        // Parallel assignment
        let assignments_and_changes: Vec<(usize, bool)> = data.par_iter().enumerate().map(|(i, point)| {
            let mut min_dist = f32::MAX;
            let mut best_cluster = 0;
            
            for (c_idx, centroid) in centroids.iter().enumerate() {
                let dist = squared_euclidean(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c_idx;
                }
            }
            
            (best_cluster, assignments[i] != best_cluster)
        }).collect();
        
        let mut _changed = 0;
        for (i, (best_cluster, changed)) in assignments_and_changes.into_iter().enumerate() {
            assignments[i] = best_cluster;
            if changed {
                _changed += 1;
            }
        }
        
        // println!("perform_kmeans: Assignment step done. Changed: {}", _changed);
        
        // Update step
        // println!("perform_kmeans: Starting update step...");
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0; k];
        
        for (i, point) in data.iter().enumerate() {
            let cluster_idx = assignments[i];
            for d in 0..dim {
                new_centroids[cluster_idx][d] += point[d];
            }
            counts[cluster_idx] += 1;
        }
        // println!("perform_kmeans: Update step accumulation done.");
        
        let mut max_shift = 0.0;
        for c_idx in 0..k {
            if counts[c_idx] > 0 {
                for d in 0..dim {
                    new_centroids[c_idx][d] /= counts[c_idx] as f32;
                }
            } else {
                // Re-initialize empty cluster
                // println!("perform_kmeans: Re-initializing empty cluster {}", c_idx);
                if let Some(random_point) = data.choose(&mut rng) {
                     new_centroids[c_idx] = (*random_point).clone();
                }
            }
            
            let shift = squared_euclidean(&centroids[c_idx], &new_centroids[c_idx]);
            if shift > max_shift {
                max_shift = shift;
            }
        }
        // println!("perform_kmeans: Update step normalization done. Max shift: {}", max_shift);
        
        centroids = new_centroids;
        
        if max_shift < tolerance {
            // println!("perform_kmeans: Converged at iteration {}", iter);
            break;
        }
    }
    
    // println!("perform_kmeans: Finished iterations.");
    
    // Build result
    let mut clusters: Vec<Cluster> = Vec::with_capacity(k);
    for i in 0..k {
        clusters.push(Cluster {
            id: i.to_string(),
            centroid: centroids[i].clone(),
            vector_ids: Vec::new(),
        });
    }
    
    for (i, cluster_idx) in assignments.iter().enumerate() {
        clusters[*cluster_idx].vector_ids.push(vectors[i].id);
    }
    
    // Remove empty clusters?
    clusters.retain(|c| !c.vector_ids.is_empty());
    
    // println!("perform_kmeans: Returning {} clusters.", clusters.len());
    clusters
}

fn squared_euclidean(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

pub fn get_node_assignment(clusters: &Vec<Cluster>, peers: &Vec<Peer>, replication_factor: usize) -> HashMap<String, Vec<Cluster>> {
    let beam_width = 50;
    println!("get_node_assignment: Starting Beam Search with width={} for {} clusters, {} peers, replication={}", 
             beam_width, clusters.len(), peers.len(), replication_factor);

    // 1. Sort clusters by load (descending)
    let mut sorted_clusters: Vec<&Cluster> = clusters.iter().collect();
    sorted_clusters.sort_by(|a, b| b.vector_ids.len().cmp(&a.vector_ids.len()));

    // 2. Map peers to indices for performance
    let peer_ids: Vec<String> = peers.iter().map(|p| p.id.clone()).collect();
    let n_peers = peer_ids.len();
    let peer_indices: Vec<usize> = (0..n_peers).collect();

    // 3. Pre-calculate all combinations of peers
    let combinations = get_combinations(&peer_indices, replication_factor);
    println!("get_node_assignment: Generated {} peer combinations.", combinations.len());

    // 4. Beam Search State: (score, assignment_indices, node_loads)
    // score: sum of squared loads (lower is better)
    // assignment_indices: Vec<usize> -> index into `combinations` for each cluster
    // node_loads: Vec<usize> -> current load for each peer index
    
    #[derive(Clone)]
    struct State {
        score: u64,
        assignment_indices: Vec<usize>, // Index in `combinations`
        node_loads: Vec<usize>,
    }

    let initial_state = State {
        score: 0,
        assignment_indices: Vec::with_capacity(sorted_clusters.len()),
        node_loads: vec![0; n_peers],
    };

    let mut beam = vec![initial_state];

    for (i, cluster) in sorted_clusters.iter().enumerate() {
        let cluster_load = cluster.vector_ids.len();
        let mut next_beam = Vec::new();

        for state in beam {
            for (combo_idx, combo) in combinations.iter().enumerate() {
                let mut new_loads = state.node_loads.clone();
                
                // Update loads
                for &peer_idx in combo {
                    new_loads[peer_idx] += cluster_load;
                }

                // Calculate new score (Sum of Squares)
                let new_score: u64 = new_loads.iter().map(|&load| (load as u64).pow(2)).sum();

                let mut new_assignment = state.assignment_indices.clone();
                new_assignment.push(combo_idx);

                next_beam.push(State {
                    score: new_score,
                    assignment_indices: new_assignment,
                    node_loads: new_loads,
                });
            }
        }

        // Keep top-k best states (lowest score)
        // Sort ascending by score
        next_beam.sort_by(|a, b| a.score.cmp(&b.score));
        next_beam.truncate(beam_width);
        beam = next_beam;
        
        if i % 5 == 0 {
             // println!("get_node_assignment: Processed {}/{} clusters. Best score: {}", i + 1, sorted_clusters.len(), beam[0].score);
        }
    }

    // 5. Construct final assignment from best state
    let best_state = &beam[0];
    println!("get_node_assignment: Beam Search finished. Best score: {}", best_state.score);

    let mut assignment: HashMap<String, Vec<Cluster>> = HashMap::new();
    for peer_id in &peer_ids {
        assignment.insert(peer_id.clone(), Vec::new());
    }

    for (cluster_idx, &combo_idx) in best_state.assignment_indices.iter().enumerate() {
        let cluster = sorted_clusters[cluster_idx];
        let combo = &combinations[combo_idx];
        
        for &peer_idx in combo {
            let peer_id = &peer_ids[peer_idx];
            assignment.get_mut(peer_id).unwrap().push(cluster.clone());
        }
    }
    
    // Log load distribution
    for (peer_idx, load) in best_state.node_loads.iter().enumerate() {
        println!("get_node_assignment: Peer {} load: {} vectors", peer_ids[peer_idx], load);
    }

    assignment
}

fn get_combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.is_empty() {
        return vec![];
    }

    let head = items[0].clone();
    let tail = &items[1..];

    // Combinations including head
    let mut with_head = get_combinations(tail, k - 1);
    for combo in &mut with_head {
        combo.insert(0, head.clone());
    }

    // Combinations excluding head
    let without_head = get_combinations(tail, k);

    with_head.extend(without_head);
    with_head
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorWithPayload;

    fn create_dummy_vectors(count: usize, dim: usize) -> Vec<VectorWithPayload> {
        let mut vectors = Vec::new();
        for i in 0..count {
            let vector: Vec<f32> = (0..dim).map(|d| (i + d) as f32).collect();
            vectors.push(VectorWithPayload {
                id: i as i64,
                vector,
                payload: format!("payload_{}", i),
                cluster: "unknown".to_string(),
            });
        }
        vectors
    }

    #[test]
    fn test_squared_euclidean() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert_eq!(squared_euclidean(&v1, &v2), 27.0);
    }

    #[test]
    fn test_perform_kmeans_basic() {
        let vectors = create_dummy_vectors(10, 2);
        // 10 vectors, k=2
        let clusters = perform_kmeans(&vectors, 2);
        assert_eq!(clusters.len(), 2);
        
        let total_assigned: usize = clusters.iter().map(|c| c.vector_ids.len()).sum();
        assert_eq!(total_assigned, 10);
    }

    #[test]
    fn test_perform_kmeans_empty() {
        let vectors = Vec::new();
        let clusters = perform_kmeans(&vectors, 2);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_perform_kmeans_k_greater_than_n() {
        let vectors = create_dummy_vectors(3, 2);
        let clusters = perform_kmeans(&vectors, 5);
        assert_eq!(clusters.len(), 3); // Should be capped at n_samples
    }

    #[test]
    fn test_calculate_silhouette_score() {
        let vectors = create_dummy_vectors(10, 2);
        let clusters = perform_kmeans(&vectors, 2);
        let score = calculate_silhouette_score(&vectors, &clusters);
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_perform_kmeans_auto_k() {
        let vectors = create_dummy_vectors(20, 2);
        // Should find best k between 2 and 5
        let clusters = perform_kmeans_auto_k(&vectors, 2, 5);
        assert!(clusters.len() >= 2 && clusters.len() <= 5);
    }

    #[test]
    fn test_get_node_assignment() {
        let mut clusters = Vec::new();
        for i in 0..5 {
            clusters.push(Cluster {
                id: i.to_string(),
                centroid: vec![0.0, 0.0],
                vector_ids: vec![0; 10], // Load of 10
            });
        }

        let peers = vec![
            Peer { id: "node1".to_string(), http_url: "".to_string(), grpc_url: "".to_string() },
            Peer { id: "node2".to_string(), http_url: "".to_string(), grpc_url: "".to_string() },
            Peer { id: "node3".to_string(), http_url: "".to_string(), grpc_url: "".to_string() },
        ];

        let replication_factor = 2;
        let assignment = get_node_assignment(&clusters, &peers, replication_factor);

        // Check that each cluster is assigned to exactly `replication_factor` nodes
        let mut cluster_counts: HashMap<String, usize> = HashMap::new();
        for (_, assigned_clusters) in &assignment {
            for cluster in assigned_clusters {
                *cluster_counts.entry(cluster.id.clone()).or_insert(0) += 1;
            }
        }

        for i in 0..5 {
            assert_eq!(*cluster_counts.get(&i.to_string()).unwrap(), replication_factor);
        }

        // Check that all peers are in the assignment map
        assert_eq!(assignment.len(), 3);
    }

    #[test]
    fn test_meta_hnsw_serialization() {
        let mut meta = MetaHNSW::new(2);
        let clusters = vec![
            Cluster {
                id: "0".to_string(),
                centroid: vec![1.0, 2.0],
                vector_ids: vec![],
            },
            Cluster {
                id: "1".to_string(),
                centroid: vec![3.0, 4.0],
                vector_ids: vec![],
            }
        ];
        
        meta.build(&clusters);
        
        let bytes = meta.to_bytes().unwrap();
        let meta_deserialized = MetaHNSW::from_bytes(&bytes, 2).unwrap();
        
        assert_eq!(meta_deserialized.data.len(), 2);
        // Check if data is preserved
        assert_eq!(meta_deserialized.data[0].1, vec![1.0, 2.0]);
    }
}


