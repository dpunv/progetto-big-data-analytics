"""
gRPC Client wrapper for internal P2P communication.
Provides high-level methods for calling gRPC services on peer nodes.
"""
import grpc
from generated import vector_service_pb2, vector_service_pb2_grpc
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
import time


@dataclass
class GRPCPeerInfo:
    """Information about a gRPC peer."""
    node_id: str
    grpc_host: str
    grpc_port: int
    
    @property
    def address(self) -> str:
        return f"{self.grpc_host}:{self.grpc_port}"


class GRPCClient:
    """
    High-level gRPC client for internal P2P operations.
    Manages channels, stubs, and provides clean API for peer communication.
    """
    
    def __init__(self, node_id: str):
        """
        Initialize gRPC client.
        
        Args:
            node_id: This node's ID (for logging)
        """
        self.node_id = node_id
        self.channels: Dict[str, grpc.Channel] = {}
        self.stubs: Dict[str, vector_service_pb2_grpc.VectorServiceStub] = {}
    
    def register_peer(self, peer_id: str, grpc_host: str, grpc_port: int):
        """
        Register a peer and create gRPC channel.
        
        Args:
            peer_id: Peer node ID
            grpc_host: Peer gRPC host
            grpc_port: Peer gRPC port
        """
        address = f"{grpc_host}:{grpc_port}"
        
        # Create insecure channel (for development/local)
        # TODO: Use secure channels in production
        channel = grpc.insecure_channel(
            address,
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
            ]
        )
        
        stub = vector_service_pb2_grpc.VectorServiceStub(channel)
        
        self.channels[peer_id] = channel
        self.stubs[peer_id] = stub
        
        print(f"gRPC Client: Registered peer {peer_id} at {address}")
    
    def close_peer(self, peer_id: str):
        """Close gRPC channel for a peer."""
        if peer_id in self.channels:
            self.channels[peer_id].close()
            del self.channels[peer_id]
            del self.stubs[peer_id]
    
    def close_all(self):
        """Close all gRPC channels."""
        for peer_id in list(self.channels.keys()):
            self.close_peer(peer_id)
    
    def send_vectors_bulk(
        self,
        peer_id: str,
        vectors_data: List[Dict],
        timeout: float = 30.0
    ) -> Optional[Dict]:
        """
        Send bulk vectors to peer via gRPC streaming.
        
        Args:
            peer_id: Target peer ID
            vectors_data: List of vector dicts with 'id', 'vector', 'payload'
            timeout: Request timeout in seconds
            
        Returns:
            Response dict with storage stats, or None on failure
        """
        if peer_id not in self.stubs:
            print(f"gRPC Client: Peer {peer_id} not registered")
            return None
        
        stub = self.stubs[peer_id]
        
        try:
            # Create vector stream generator
            def vector_generator() -> Iterator[vector_service_pb2.VectorData]:
                for vec_dict in vectors_data:
                    yield vector_service_pb2.VectorData(
                        id=vec_dict['id'],
                        vector=vec_dict['vector'],
                        payload={k: str(v) for k, v in vec_dict.get('payload', {}).items()},
                        from_node=self.node_id
                    )
            
            # Call gRPC streaming method
            response = stub.AddVectorsBulk(vector_generator(), timeout=timeout)
            
            return {
                'vectors_stored': response.vectors_stored,
                'node_id': response.node_id,
                'batches': dict(response.batches)
            }
        
        except grpc.RpcError as e:
            print(f"gRPC Client: Error sending bulk to {peer_id}: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            print(f"gRPC Client: Unexpected error sending bulk to {peer_id}: {e}")
            return None
    
    def send_vector(
        self,
        peer_id: str,
        vector_dict: Dict,
        timeout: float = 10.0
    ) -> Optional[Dict]:
        """
        Send single vector to peer via gRPC.
        
        Args:
            peer_id: Target peer ID
            vector_dict: Vector dict with 'id', 'vector', 'payload'
            timeout: Request timeout in seconds
            
        Returns:
            Response dict, or None on failure
        """
        if peer_id not in self.stubs:
            print(f"gRPC Client: Peer {peer_id} not registered")
            return None
        
        stub = self.stubs[peer_id]
        
        try:
            request = vector_service_pb2.VectorData(
                id=vector_dict['id'],
                vector=vector_dict['vector'],
                payload={k: str(v) for k, v in vector_dict.get('payload', {}).items()},
                from_node=self.node_id
            )
            
            response = stub.SendVector(request, timeout=timeout)
            
            return {
                'success': response.success,
                'message': response.message,
                'node_id': response.node_id
            }
        
        except grpc.RpcError as e:
            print(f"gRPC Client: Error sending vector to {peer_id}: {e.code()}")
            return None
        except Exception as e:
            print(f"gRPC Client: Unexpected error: {e}")
            return None
    
    def search_peer(
        self,
        peer_id: str,
        query_vector: List[float],
        top_k: int = 5,
        timeout: float = 10.0
    ) -> Optional[List[Dict]]:
        """
        Search peer node via gRPC.
        
        Args:
            peer_id: Target peer ID
            query_vector: Query vector
            top_k: Number of results
            timeout: Request timeout
            
        Returns:
            List of result dicts, or None on failure
        """
        if peer_id not in self.stubs:
            print(f"gRPC Client: Peer {peer_id} not registered")
            return None
        
        stub = self.stubs[peer_id]
        
        try:
            request = vector_service_pb2.SearchRequest(
                from_node=self.node_id,
                query_vector=query_vector,
                top_k=top_k
            )
            
            response = stub.SearchPeer(request, timeout=timeout)
            
            # Convert protobuf results to dicts
            results = []
            for result in response.results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'vector': list(result.vector),
                    'payload': dict(result.payload)
                })
            
            return results
        
        except grpc.RpcError as e:
            print(f"gRPC Client: Error searching {peer_id}: {e.code()}")
            return None
        except Exception as e:
            print(f"gRPC Client: Unexpected error: {e}")
            return None
    
    def health_check(
        self,
        peer_id: str,
        timeout: float = 2.0
    ) -> Optional[Dict]:
        """
        Check peer health via gRPC.
        
        Args:
            peer_id: Target peer ID
            timeout: Request timeout
            
        Returns:
            Health status dict, or None on failure
        """
        if peer_id not in self.stubs:
            return None
        
        stub = self.stubs[peer_id]
        
        try:
            request = vector_service_pb2.HealthCheckRequest(node_id=self.node_id)
            response = stub.HealthCheck(request, timeout=timeout)
            
            return {
                'status': response.status,
                'node_id': response.node_id,
                'timestamp': response.timestamp
            }
        
        except grpc.RpcError:
            return None
        except Exception:
            return None
