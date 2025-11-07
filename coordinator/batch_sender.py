"""
Adaptive batch sender with intelligent retry logic.
"""
import time
import requests
from typing import List, Dict, Tuple
from threading import Lock
from utils.serialization import serialize_msgpack


class AdaptiveBatchMetrics:
    """Tracks batch sending metrics and adjusts batch size dynamically."""
    
    def __init__(self, optimal_size: int, min_size: int):
        self.current_batch_size = optimal_size
        self.optimal_size = optimal_size
        self.min_size = min_size
        
        self.success_count = 0
        self.failure_count = 0
        self.total_retries = 0
        self.total_splits = 0
        self.lock = Lock()
    
    def on_success(self, batch_size: int):
        """Called when a batch succeeds."""
        with self.lock:
            self.success_count += 1
            self.failure_count = 0
            
            if self.success_count >= 20 and self.current_batch_size < self.optimal_size:
                old_size = self.current_batch_size
                self.current_batch_size = min(int(self.current_batch_size * 1.5), self.optimal_size)
                print(f"📈 Increasing batch size: {old_size} → {self.current_batch_size} (after {self.success_count} successes)")
                self.success_count = 0
    
    def on_failure(self):
        """Called when a batch fails."""
        with self.lock:
            self.failure_count += 1
            self.success_count = 0
            
            if self.failure_count >= 2:
                old_size = self.current_batch_size
                self.current_batch_size = max(self.current_batch_size // 2, self.min_size)
                print(f"📉 Reducing batch size preventively: {old_size} → {self.current_batch_size} (after {self.failure_count} failures)")
    
    def on_retry(self):
        """Called when a retry happens."""
        with self.lock:
            self.total_retries += 1
    
    def on_split(self):
        """Called when a batch is split."""
        with self.lock:
            self.total_splits += 1
    
    def get_stats(self) -> Dict:
        """Returns statistics summary."""
        with self.lock:
            return {
                "current_batch_size": self.current_batch_size,
                "total_retries": self.total_retries,
                "total_splits": self.total_splits,
                "success_count": self.success_count,
                "failure_count": self.failure_count
            }


class BatchSender:
    """Handles batch sending with adaptive retry logic."""
    
    def __init__(self, batch_tiers: List[int], base_timeout: int = 10):
        self.batch_tiers = batch_tiers
        self.base_timeout = base_timeout
        self.metrics = AdaptiveBatchMetrics(batch_tiers[0], batch_tiers[-1])
    
    def calculate_timeout(self, batch_size: int) -> int:
        """Calculate adaptive timeout based on batch size."""
        per_vector_time = 0.01
        timeout = self.base_timeout + int(batch_size * per_vector_time)
        return max(10, min(timeout, 120))
    
    def send_batch(
        self,
        vectors_batch: List[Dict],
        node_url: str,
        sent_to_node_id: str,
        retry_count: int = 0
    ) -> Tuple[bool, Dict]:
        """
        Send a batch with intelligent retry using predefined size tiers.
        
        Returns:
            (success: bool, response_data: dict)
        """
        batch_size = len(vectors_batch)
        
        if batch_size == 0:
            return True, {"batches": {}}
        
        if retry_count >= len(self.batch_tiers):
            print(f"❌ All retry tiers exhausted for batch of {batch_size} vectors")
            self.metrics.on_failure()
            return False, {}
        
        current_tier_size = self.batch_tiers[retry_count]
        
        # Split if batch exceeds tier size
        if batch_size > current_tier_size:
            print(f"📦 Batch size {batch_size} exceeds tier {retry_count+1} ({current_tier_size}), splitting...")
            self.metrics.on_split()
            
            chunks = [vectors_batch[i:i+current_tier_size] 
                     for i in range(0, batch_size, current_tier_size)]
            
            print(f"🔀 Split into {len(chunks)} chunks of max {current_tier_size} vectors")
            
            all_results = []
            for chunk in chunks:
                success, data = self.send_batch(chunk, node_url, sent_to_node_id, retry_count)
                if not success:
                    return False, {}
                all_results.append(data)
            
            # Merge results
            merged_batches = {}
            for data in all_results:
                for node_id, count in data.get('batches', {}).items():
                    merged_batches[node_id] = merged_batches.get(node_id, 0) + count
            
            return True, {"batches": merged_batches}
        
        # Send batch
        timeout = self.calculate_timeout(batch_size)
        
        try:
            binary_data = serialize_msgpack(vectors_batch)
            
            response = requests.post(
                f"{node_url}/add_vectors_bulk",
                data=binary_data,
                headers={"Content-Type": "application/msgpack"},
                timeout=timeout
            )
            response.raise_for_status()
            
            res_data = response.json()
            self.metrics.on_success(batch_size)
            return True, res_data
            
        except requests.exceptions.Timeout:
            next_tier = retry_count + 1
            if next_tier < len(self.batch_tiers):
                next_tier_size = self.batch_tiers[next_tier]
                print(f"⏱️  Timeout with batch size {batch_size} (tier {retry_count+1}: {current_tier_size})")
                print(f"   Falling back to tier {next_tier+1} (max size: {next_tier_size})...")
                self.metrics.on_retry()
                return self.send_batch(vectors_batch, node_url, sent_to_node_id, next_tier)
            else:
                print(f"❌ Timeout even with smallest tier ({current_tier_size})")
                self.metrics.on_failure()
                return False, {}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error with batch size {batch_size} (tier {retry_count+1}): {e}")
            self.metrics.on_retry()
            
            if retry_count == 0:
                time.sleep(2)
            
            next_tier = retry_count + 1
            if next_tier < len(self.batch_tiers):
                return self.send_batch(vectors_batch, node_url, sent_to_node_id, next_tier)
            else:
                self.metrics.on_failure()
                return False, {}
