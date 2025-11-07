"""
gRPC Service Implementation for internal P2P communication.
Handles vector bulk insert, peer search, gossip protocol, health checks.
"""
import grpc
from generated import vector_service_pb2, vector_service_pb2_grpc
from typing import Iterator
import time


class VectorServiceServicer(vector_service_pb2_grpc.VectorServiceServicer):
    """Implementation of VectorService gRPC service."""
    
    def __init__(self, node_wrapper):
        """
        Initialize servicer with node wrapper.
        
        Args:
            node_wrapper: QdrantNodeWrapper instance
        """
        self.node = node_wrapper
    
    def AddVectorsBulk(
        self, 
        request_iterator: Iterator[vector_service_pb2.VectorData],
        context: grpc.ServicerContext
    ) -> vector_service_pb2.BulkInsertResponse:
        """
        Receive streaming bulk vectors and store them.
        
        Args:
            request_iterator: Stream of VectorData messages
            context: gRPC context
            
        Returns:
            BulkInsertResponse with storage statistics
        """
        from server import VectorData
        
        vectors_data = []
        from_node = None
        
        try:
            # Collect vectors from stream
            for vector_proto in request_iterator:
                if from_node is None:
                    from_node = vector_proto.from_node or "unknown"
                
                # Convert protobuf to VectorData dataclass
                vec_data = VectorData(
                    id=vector_proto.id,
                    vector=list(vector_proto.vector),
                    payload=dict(vector_proto.payload) if vector_proto.payload else {}
                )
                vectors_data.append(vec_data)
            
            # Store bulk
            if vectors_data:
                success = self.node.receive_vectors_bulk(from_node, vectors_data)
                
                if success:
                    return vector_service_pb2.BulkInsertResponse(
                        vectors_stored=len(vectors_data),
                        node_id=self.node.node_id,
                        batches={self.node.node_id: len(vectors_data)}
                    )
                else:
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details("Failed to store vectors in Qdrant")
                    return vector_service_pb2.BulkInsertResponse()
            else:
                # Empty stream
                return vector_service_pb2.BulkInsertResponse(
                    vectors_stored=0,
                    node_id=self.node.node_id
                )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing bulk insert: {str(e)}")
            return vector_service_pb2.BulkInsertResponse()
    
    def SendVector(
        self,
        request: vector_service_pb2.VectorData,
        context: grpc.ServicerContext
    ) -> vector_service_pb2.VectorResponse:
        """
        Receive single vector (for compatibility).
        
        Args:
            request: VectorData message
            context: gRPC context
            
        Returns:
            VectorResponse with success status
        """
        from server import VectorData
        
        try:
            vec_data = VectorData(
                id=request.id,
                vector=list(request.vector),
                payload=dict(request.payload) if request.payload else {}
            )
            
            from_node = request.from_node or "unknown"
            success = self.node.receive_vector(from_node, vec_data)
            
            return vector_service_pb2.VectorResponse(
                success=success,
                message=f"Vector {vec_data.id} stored" if success else "Failed to store vector",
                node_id=self.node.node_id
            )
        
        except Exception as e:
            return vector_service_pb2.VectorResponse(
                success=False,
                message=str(e),
                node_id=self.node.node_id
            )
    
    def SearchPeer(
        self,
        request: vector_service_pb2.SearchRequest,
        context: grpc.ServicerContext
    ) -> vector_service_pb2.SearchResponse:
        """
        Search local database (called by peer nodes via gRPC).
        
        Args:
            request: SearchRequest with query vector
            context: gRPC context
            
        Returns:
            SearchResponse with results
        """
        try:
            query_vector = list(request.query_vector)
            top_k = request.top_k or 5
            
            results = self.node.search_local(query_vector, top_k)
            
            if results is None:
                return vector_service_pb2.SearchResponse(
                    results=[],
                    node_id=self.node.node_id
                )
            
            # Convert Qdrant results to protobuf
            proto_results = []
            for result in results:
                proto_result = vector_service_pb2.SearchResult(
                    id=result.get('id', ''),
                    score=float(result.get('score', 0.0)),
                    vector=result.get('vector', []),
                    payload={k: str(v) for k, v in result.get('payload', {}).items()}
                )
                proto_results.append(proto_result)
            
            return vector_service_pb2.SearchResponse(
                results=proto_results,
                node_id=self.node.node_id
            )
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Search error: {str(e)}")
            return vector_service_pb2.SearchResponse(
                results=[],
                node_id=self.node.node_id
            )
    
    def GossipClusters(
        self,
        request_iterator: Iterator[vector_service_pb2.ClusterUpdate],
        context: grpc.ServicerContext
    ) -> Iterator[vector_service_pb2.ClusterUpdate]:
        """
        Bidirectional gossip protocol for cluster updates.
        
        Args:
            request_iterator: Stream of cluster updates from peer
            context: gRPC context
            
        Yields:
            ClusterUpdate messages with local clusters
        """
        try:
            # Receive peer's clusters
            for update in request_iterator:
                from_node = update.from_node
                cluster_vectors = [list(cv.centroid) for cv in update.cluster_vectors]
                
                if cluster_vectors:
                    self.node.receive_peer_clusters(from_node, cluster_vectors)
            
            # Send back own clusters
            if self.node.meta_hnsw:
                local_cluster_ids = self.node.meta_hnsw.node_to_clusters.get(self.node.node_id, [])
                if local_cluster_ids:
                    local_clusters = [
                        self.node.meta_hnsw.cluster_centroids[cid].tolist() 
                        for cid in local_cluster_ids
                    ]
                    
                    cluster_protos = [
                        vector_service_pb2.ClusterVector(
                            centroid=cluster,
                            cluster_id=cid
                        )
                        for cid, cluster in zip(local_cluster_ids, local_clusters)
                    ]
                    
                    yield vector_service_pb2.ClusterUpdate(
                        from_node=self.node.node_id,
                        cluster_vectors=cluster_protos
                    )
        
        except Exception as e:
            print(f"Gossip error: {e}")
    
    def HealthCheck(
        self,
        request: vector_service_pb2.HealthCheckRequest,
        context: grpc.ServicerContext
    ) -> vector_service_pb2.HealthCheckResponse:
        """
        Health check endpoint for gRPC.
        
        Args:
            request: HealthCheckRequest
            context: gRPC context
            
        Returns:
            HealthCheckResponse with node status
        """
        return vector_service_pb2.HealthCheckResponse(
            status="UP",
            node_id=self.node.node_id,
            timestamp=int(time.time() * 1000)
        )
