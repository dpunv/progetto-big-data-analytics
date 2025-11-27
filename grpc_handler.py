import grpc
import p2p_pb2
import p2p_pb2_grpc
import pickle
import logging
from compound_types import *

logger = logging.getLogger(__name__)

class P2PNodeServicer(p2p_pb2_grpc.P2PNodeServicer):
    def __init__(self, server_app):
        self.server_app = server_app

    def ReceiveVectors(self, request, context):
        logger.info(f"[GRPC IN] ReceiveVectors called. ReqID: {request.req_id}, Type: {request.type}, Count: {len(request.content)}")
        # Convert Proto messages back to Python Tuples for ServerApp
        converted_content = []
        for vp in request.content:
            # (Vector, VectorId, VectorPayload, ClusterId)
            converted_content.append((list(vp.vector), vp.id, vp.payload, vp.cluster_id))
        
        self.server_app.add_vectors(converted_content, request.type, request.req_id)
        logger.info(f"[GRPC IN] ReceiveVectors processed successfully.")
        return p2p_pb2.Empty()

    def QueryPeer(self, request, context):
        logger.info(f"[GRPC IN] QueryPeer called. ReqID: {request.req_id}, TopK: {request.topk}, Vectors: {len(request.query_vectors)}, ClusterID: {request.cluster_id}")
        # Convert Proto vectors to Python lists
        queries = [list(q.values) for q in request.query_vectors]
        
        results = self.server_app.query_me(queries, request.topk, request.cluster_id)
        
        # Convert Python dict results back to Proto ScoredPoints
        response_points = []
        for r in results:
            sp = p2p_pb2.ScoredPoint(
                id=r['id'],
                score=r['score'],
                payload=p2p_pb2.VectorPoint(
                    vector=r['payload']['vector'],
                    id=r['id'],
                    payload=r['payload']['string']
                )
            )
            response_points.append(sp)
            
        logger.info(f"[GRPC IN] QueryPeer returning {len(response_points)} results.")
        return p2p_pb2.QueryResponse(results=response_points)

    def SetClusters(self, request, context):
        logger.info(f"[GRPC IN] SetClusters called. ReqID: {request.req_id}")
        # Deserialize the complex dictionary
        try:
            content = pickle.loads(request.content_pickle)
            self.server_app.set_clusters(content, request.req_id)
            logger.info(f"[GRPC IN] SetClusters applied successfully.")
        except Exception as e:
            logger.error(f"[GRPC IN] Error deserializing clusters: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Failed to deserialize cluster data')
            return p2p_pb2.Empty()
            
        return p2p_pb2.Empty()