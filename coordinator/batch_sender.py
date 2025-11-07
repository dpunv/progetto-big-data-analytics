"""
Adaptive batch sender with intelligent retry logic.
NOW USES: gRPC streaming instead of HTTP+MessagePack
"""
import time
import grpc
from typing import List, Dict, Tuple
from threading import Lock
from generated import vector_service_pb2, vector_service_pb2_grpc


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
    """
    Handles batch sending with adaptive retry logic via gRPC.
    REPLACES: HTTP+MessagePack with gRPC streaming
    """
    
    def __init__(self, batch_tiers: List[int], base_timeout: int = 10, grpc_port_offset: int = 9000):
        self.batch_tiers = batch_tiers
        self.base_timeout = base_timeout
        self.metrics = AdaptiveBatchMetrics(batch_tiers[0], batch_tiers[-1])
        self.grpc_port_offset = grpc_port_offset
        
        # gRPC channel cache
        self.grpc_channels: Dict[str, grpc.Channel] = {}
        self.grpc_stubs: Dict[str, vector_service_pb2_grpc.VectorServiceStub] = {}
    
    def _get_grpc_stub(self, node_url: str) -> vector_service_pb2_grpc.VectorServiceStub:
        """
        Get or create gRPC stub for a node.
        
        Args:
            node_url: FastAPI URL (e.g., "http://localhost:8001")
            
        Returns:
            gRPC stub for the node
        """
        if node_url in self.grpc_stubs:
            return self.grpc_stubs[node_url]
        
        # Extract host and infer gRPC port
        from urllib.parse import urlparse
        parsed = urlparse(node_url)
        host = parsed.hostname or "localhost"
        fastapi_port = parsed.port or 8001
        
        # Infer gRPC port (convention: node1 @ 8001 → gRPC @ 9001)
        node_num = fastapi_port - 8000
        grpc_port = self.grpc_port_offset + node_num
        
        grpc_address = f"{host}:{grpc_port}"
        
        # Create gRPC channel
        channel = grpc.insecure_channel(
            grpc_address,
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
            ]
        )
        
        stub = vector_service_pb2_grpc.VectorServiceStub(channel)
        
        self.grpc_channels[node_url] = channel
        self.grpc_stubs[node_url] = stub
        
        return stub
    
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
        Send a batch via gRPC streaming with intelligent retry.
        
        REPLACES: HTTP POST with MessagePack
        NOW USES: gRPC streaming RPC
        
        Args:
            vectors_batch: List of vector dicts
            node_url: Target node FastAPI URL (for stub lookup)
            sent_to_node_id: Target node ID (for logging)
            retry_count: Current retry attempt
            
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
        
        # Send batch via gRPC
        timeout = self.calculate_timeout(batch_size)
        
        try:
            stub = self._get_grpc_stub(node_url)
            
            # Create gRPC streaming generator
            def vector_generator():
                for vec_dict in vectors_batch:
                    yield vector_service_pb2.VectorData(
                        id=vec_dict['id'],
                        vector=vec_dict['vector'],
                        payload={k: str(v) for k, v in vec_dict.get('payload', {}).items()},
                        from_node=sent_to_node_id
                    )
            
            # Call gRPC streaming method
            response = stub.AddVectorsBulk(vector_generator(), timeout=timeout)
            
            res_data = {
                'batches': dict(response.batches),
                'vectors_stored': response.vectors_stored,
                'node_id': response.node_id
            }
            
            self.metrics.on_success(batch_size)
            return True, res_data
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                # Timeout - try next tier
                next_tier = retry_count + 1
                if next_tier < len(self.batch_tiers):
                    next_tier_size = self.batch_tiers[next_tier]
                    print(f"⏱️  gRPC timeout with batch size {batch_size} (tier {retry_count+1}: {current_tier_size})")
                    print(f"   Falling back to tier {next_tier+1} (max size: {next_tier_size})...")
                    self.metrics.on_retry()
                    return self.send_batch(vectors_batch, node_url, sent_to_node_id, next_tier)
                else:
                    print(f"❌ gRPC timeout even with smallest tier ({current_tier_size})")
                    self.metrics.on_failure()
                    return False, {}
            else:
                # Other gRPC error
                print(f"❌ gRPC error with batch size {batch_size} (tier {retry_count+1}): {e.code()} - {e.details()}")
                self.metrics.on_retry()
                
                if retry_count == 0:
                    time.sleep(2)
                
                next_tier = retry_count + 1
                if next_tier < len(self.batch_tiers):
                    return self.send_batch(vectors_batch, node_url, sent_to_node_id, next_tier)
                else:
                    self.metrics.on_failure()
                    return False, {}
                    
        except Exception as e:
            print(f"❌ Unexpected error with batch size {batch_size}: {e}")
            self.metrics.on_failure()
            return False, {}
    
    def close_all_channels(self):
        """Close all gRPC channels."""
        for channel in self.grpc_channels.values():
            channel.close()
        self.grpc_channels.clear()
        self.grpc_stubs.clear()
