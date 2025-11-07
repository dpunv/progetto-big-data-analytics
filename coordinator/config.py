"""
Centralized configuration for the distributed Qdrant application.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    """Application configuration with smart defaults."""
    
    # Core settings
    num_nodes: int
    num_vectors: int = 20000
    vector_size: int = 384
    
    # Batch configuration
    batch_tiers: List[int] = field(default_factory=lambda: [800, 256, 64])
    batch_size_optimal: int = field(init=False)
    batch_size_min: int = field(init=False)
    max_retries: int = field(init=False)
    
    # Training configuration
    training_vectors: int = 10000
    assignments_file: str = 'node_assignments.json'
    centroids_file: str = 'centroids.json'
    force_retrain: bool = False
    
    # Network configuration
    base_port: int = 8000
    
    # Peer health monitoring
    peer_report_interval: int = 30
    majority_threshold: int = field(init=False)
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        self.batch_size_optimal = self.batch_tiers[0]
        self.batch_size_min = self.batch_tiers[-1]
        self.max_retries = len(self.batch_tiers)
        self.majority_threshold = (self.num_nodes // 2) + 1
        
        # Validation
        if self.vector_size < self.num_nodes:
            raise ValueError(
                f"VECTOR_SIZE ({self.vector_size}) must be >= NUM_NODES ({self.num_nodes})"
            )
    
    def get_node_urls(self) -> List[str]:
        """Generate node URLs based on configuration."""
        return [f"http://localhost:{self.base_port + i}" for i in range(1, self.num_nodes + 1)]
    
    def get_node_id(self, index: int) -> str:
        """Get node ID for a given index (0-based)."""
        return f"node{index + 1}"
    
    def __str__(self) -> str:
        """Human-readable configuration summary."""
        return (
            f"AppConfig:\n"
            f"  Nodes: {self.num_nodes}\n"
            f"  Vectors: {self.num_vectors:,} (size: {self.vector_size})\n"
            f"  Training: {self.training_vectors:,} vectors\n"
            f"  Batch: {self.batch_size_optimal} (optimal) → {self.batch_size_min} (min)\n"
            f"  Ports: {self.base_port + 1}-{self.base_port + self.num_nodes}"
        )
