"""
Node coordinator for distributed operations: peer registration, Meta-HNSW setup, health checks.
"""
import requests
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


class NodeCoordinator:
    """
    Handles broadcast operations to all nodes in the cluster.
    Encapsulates common patterns: "for each node, do X, collect results".
    """
    
    def __init__(self, config):
        """
        Initialize coordinator with configuration.
        
        Args:
            config: AppConfig instance
        """
        self.config = config
        self.node_urls = config.get_node_urls()
    
    def setup_peer_network(self, centroids: List[List[float]], node_assignments: Dict[int, List[int]]) -> bool:
        """
        Setup complete peer network:
        1. Set node vectors (representative centroids)
        2. Register all peers with each other
        
        Args:
            centroids: All cluster centroids
            node_assignments: Dict mapping node_idx -> list of cluster indices
            
        Returns:
            True if setup successful
        """
        print("\n" + "="*60)
        print("1. SETTING NODE VECTORS & REGISTERING PEERS")
        print("="*60)
        
        try:
            # Step 1: Set node vectors
            print("Setting node vectors (multiple per node)...")
            for node_idx in range(self.config.num_nodes):
                node_id = self.config.get_node_id(node_idx)
                node_url = self.node_urls[node_idx]
                
                # Get cluster indices for this node
                cluster_indices = node_assignments.get(str(node_idx), node_assignments.get(node_idx, []))
                
                # Get actual centroid vectors
                node_vectors = [centroids[cluster_idx] for cluster_idx in cluster_indices]
                
                if not node_vectors:
                    print(f"  ⚠️  {node_id}: No clusters assigned! Using random vector.")
                    node_vectors = [np.random.rand(self.config.vector_size).tolist()]
                
                # Send vectors to node
                response = requests.post(f"{node_url}/set_node_vectors", json=node_vectors, timeout=5)
                response.raise_for_status()
                print(f"  {node_id}: Set {len(node_vectors)} representative vector(s) for clusters {cluster_indices}")
            
            # Step 2: Register peers
            print("\nRegistering peers...")
            for node_idx in range(self.config.num_nodes):
                host_id = self.config.get_node_id(node_idx)
                host_url = self.node_urls[node_idx]
                
                peers_registered = 0
                for peer_idx in range(self.config.num_nodes):
                    if node_idx == peer_idx:
                        continue
                    
                    peer_id = self.config.get_node_id(peer_idx)
                    peer_url = self.node_urls[peer_idx]
                    
                    # Get peer's cluster info
                    peer_cluster_indices = node_assignments.get(str(peer_idx), node_assignments.get(peer_idx, []))
                    peer_vectors = [centroids[cluster_idx] for cluster_idx in peer_cluster_indices]
                    
                    if not peer_vectors:
                        peer_vectors = [np.random.rand(self.config.vector_size).tolist()]
                    
                    payload = {
                        "peer_id": peer_id,
                        "peer_url": peer_url,
                        "node_vectors": peer_vectors
                    }
                    
                    response = requests.post(f"{host_url}/register_peer", json=payload, timeout=5)
                    response.raise_for_status()
                    peers_registered += 1
                
                print(f"  Host {host_id}: Registered {peers_registered} peers.")
            
            print("✅ Peers registered and vectors exchanged successfully.\n")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"!!! Error setting/registering peers: {e}")
            print("!!! Please ensure all servers are running.")
            return False
    
    def distribute_meta_hnsw(self, centroids: List[List[float]], node_assignments: Dict) -> bool:
        """
        Initialize Meta-HNSW on all nodes with complete cluster data.
        
        Args:
            centroids: All cluster centroids
            node_assignments: Node assignment mapping
            
        Returns:
            True if all nodes initialized successfully
        """
        print("\n" + "="*60)
        print("1.5. Initializing Distributed Meta-HNSW on Nodes")
        print("="*60)
        
        payload = {
            "dimension": self.config.vector_size,
            "max_clusters": max(len(centroids), self.config.num_nodes * 10),
            "centroids": centroids,
            "node_assignments": node_assignments
        }
        
        success_count = 0
        for i, node_url in enumerate(self.node_urls):
            node_id = self.config.get_node_id(i)
            try:
                response = requests.post(
                    f"{node_url}/init-meta-hnsw",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print(f"  ✓ {node_id}: Meta-HNSW initialized")
                    success_count += 1
                else:
                    print(f"  ✗ {node_id}: Failed (HTTP {response.status_code}) - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ✗ {node_id}: Error - {e}")
        
        print(f"\n✅ Distributed Meta-HNSW initialized on {success_count}/{self.config.num_nodes} nodes\n")
        
        if success_count < self.config.num_nodes:
            print("⚠️  Warning: Not all nodes initialized Meta-HNSW successfully!")
            print("   The system might not route queries correctly. Check server logs.")
        
        return success_count == self.config.num_nodes
    
    def verify_node_counts(self, insertions: List[str], replication_factor: int) -> Dict:
        """
        Check final vector counts on all nodes and validate distribution.
        
        Args:
            insertions: List of node IDs where vectors were inserted (client-side log)
            replication_factor: Expected replication factor
            
        Returns:
            Dict with statistics
        """
        print("\n" + "="*60)
        print("3. CHECKING VECTOR COUNTS")
        print("="*60)
        
        try:
            all_counts = []
            total_count = 0
            
            # Collect counts from all nodes
            for i in range(self.config.num_nodes):
                node_id = self.config.get_node_id(i)
                node_url = self.node_urls[i]
                
                response = requests.get(f"{node_url}/count", timeout=5)
                response.raise_for_status()
                count = response.json().get('count', 0)
                
                all_counts.append((node_id, count))
                total_count += count
            
            # Print counts per node
            print("Vector counts per node:")
            for node_id, count in all_counts:
                print(f"  - {node_id} Count: {count}")
            
            print(f"\nTotal Vectors Stored: {total_count}")
            print(f"Expected (with replication): {self.config.num_vectors * replication_factor}")
            print(f"Expected (unique): {self.config.num_vectors}")
            
            # Check from client-side log
            print("\n(Client-side insertion log check):")
            for i in range(self.config.num_nodes):
                node_id = self.config.get_node_id(i)
                script_count = insertions.count(node_id)
                print(f"  - {node_id} received: {script_count}")
            
            # Validate totals
            expected_total = self.config.num_vectors * replication_factor
            if abs(total_count - expected_total) < self.config.num_vectors * 0.05:  # 5% tolerance
                print(f"✅  Total counts match expected (within 5% tolerance).")
            else:
                print(f"⚠️  Counts deviate from expected!")
            
            # Check load balancing
            expected_avg = expected_total / self.config.num_nodes
            max_count = max(c for _, c in all_counts)
            min_count = min(c for _, c in all_counts)
            imbalance = (max_count - min_count) / expected_avg * 100 if expected_avg > 0 else 0
            
            print(f"\nLoad Balance Statistics:")
            print(f"  - Expected avg per node: {expected_avg:,.1f}")
            print(f"  - Actual range: {min_count:,} to {max_count:,}")
            print(f"  - Imbalance: {imbalance:.1f}%")
            
            if imbalance < 20:
                print(f"✅  Distribution is well balanced (<20% imbalance).")
            else:
                print(f"⚠️  Distribution could be more balanced!")
            print("")
            
            return {
                'total_count': total_count,
                'expected_total': expected_total,
                'imbalance': imbalance,
                'all_counts': all_counts
            }
            
        except requests.exceptions.RequestException as e:
            print(f"!!! Error checking counts: {e}")
            return {}
    
    def update_unavailable_nodes(self) -> Tuple[set, int]:
        """
        Query /peers on all nodes and aggregate health status.
        Marks nodes as unavailable if majority of peers report them DOWN.
        
        Returns:
            (unavailable_nodes: set, majority_threshold: int)
        """
        counts_down = {self.config.get_node_id(i): 0 for i in range(self.config.num_nodes)}
        reporters = 0
        
        for i, reporter_url in enumerate(self.node_urls):
            try:
                response = requests.get(f"{reporter_url}/peers", timeout=2)
                if response.status_code != 200:
                    continue
                    
                reporters += 1
                info = response.json()
                peers = info.get("peers", {})
                
                for peer_id, pdata in peers.items():
                    status = pdata.get("status", "UNKNOWN")
                    if status == "DOWN":
                        counts_down[peer_id] = counts_down.get(peer_id, 0) + 1
                        
            except requests.exceptions.RequestException:
                continue
        
        # Mark as unavailable if >= MAJORITY reporters say DOWN
        unavailable = set()
        for node_id, down_count in counts_down.items():
            if down_count >= self.config.majority_threshold:
                unavailable.add(node_id)
        
        return unavailable, reporters
