import grpc
import p2p_pb2
import p2p_pb2_grpc
import pickle
from compound_types import *

class P2PNodeServicer(p2p_pb2_grpc.P2PNodeServicer):
    def __init__(self, server_app):
        self.server_app = server_app

    def ReceiveVectors(self, request, context):
        # Convert Proto messages back to Python Tuples for ServerApp
        converted_content = []
        for vp in request.content:
            # (Vector, VectorId, VectorPayload)
            converted_content.append((list(vp.vector), vp.id, vp.payload))
        
        self.server_app.add_vectors(converted_content, request.type, request.req_id)
        return p2p_pb2.Empty()

    def QueryPeer(self, request, context):
        # Convert Proto vectors to Python lists
        queries = [list(q.values) for q in request.query_vectors]
        
        results = self.server_app.query_me(queries, request.topk)
        
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
            
        return p2p_pb2.QueryResponse(results=response_points)

    def SetClusters(self, request, context):
        # Deserialize the complex dictionary
        content = pickle.loads(request.content_pickle)
        self.server_app.set_clusters(content, request.req_id)
        return p2p_pb2.Empty()

    def NotifyClustering(self, request, context):
        self.server_app.notify_clustering()
        return p2p_pb2.Empty()