import grpc
import base64
import pickle
from typing import Union
from .base import BaseCommunicator
from protos import p2p_pb2, p2p_pb2_grpc

class GRPCCommunicator(BaseCommunicator):
    async def send(self, ip: str, port: int, query: str, *args):
        target = f"{ip}:{port}"
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        
        async with grpc.aio.insecure_channel(target, options=options) as channel:
            stub = p2p_pb2_grpc.P2PNodeStub(channel)
            
            try:
                response = None
                
                # Dynamic dispatch based on query string
                if query == "get_id":
                    response_proto = await stub.GetId(p2p_pb2.Empty())
                    # Expect IdResponse
                    response_val = response_proto.id
                    
                elif query == "similarity":
                    # args[0] is vector
                    vec = args[0]
                    req = p2p_pb2.VectorRequest(values=vec)
                    response_proto = await stub.Similarity(req)
                    response_val = response_proto.score
                    
                elif query == "receive":
                    # args[0] is list of vectors, args[1] is status
                    # Vector structure: (values, id, payload, cluster_id, [ver, dest])
                    vectors_data = args[0]
                    status = args[1]
                    
                    proto_vectors = []
                    for v in vectors_data:
                        # Handle variable length (legacy vs full)
                        values = v[0]
                        vid = v[1]
                        payload = v[2]
                        cid = v[3]
                        
                        ts = 0.0
                        nid = 0
                        dests = []
                        
                        if len(v) > 4:
                            # Version: (ts, nid)
                            ver = v[4]
                            if ver:
                                ts = ver[0]
                                nid = ver[1]
                        
                        if len(v) > 5:
                            # Destinations: set/list of ints
                            d_set = v[5]
                            if d_set:
                                dests = list(d_set)
                        
                        vc = p2p_pb2.VectorComplete(
                            values=values,
                            id=vid,
                            payload=payload,
                            cluster_id=cid,
                            timestamp=ts,
                            version_node_id=nid,
                            destinations=dests
                        )
                        proto_vectors.append(vc)
                        
                    req = p2p_pb2.ReceiveRequest(vectors=proto_vectors, status=status)
                    await stub.Receive(req)
                    response_val = None # Void return
                    
                elif query == "i_am_coord":
                    response_proto = await stub.IAmCoord(p2p_pb2.Empty())
                    response_val = response_proto.value
                    
                elif query == "set_clusters":
                    # args[0] is clusters dict, args[1] is assignment
                    clusters_in = args[0]
                    assignment_in = args[1]
                    
                    # Convert clusters dict to Map<int, ClusterData>
                    clusters_map = {}
                    for cid, data in clusters_in.items():
                        center = data['center']
                        members = []
                        for m in data['members']:
                             # Member is VectorComplete-like tuple
                             # Re-use logic or duplicate? Simple map here.
                             # member tuple: (vec, id, payload, cluster_id, ...)
                             # Check length... consistent logic needed.
                             
                             # Simplify: For now, assume consistent structure or just minimal needed fields
                             # Members in clustering usually full tuples.
                             mv = m[0]
                             mid = m[1]
                             mp = m[2]
                             mcid = m[3]
                             mts = 0.0; mnid = 0; mdests = []
                             if len(m) > 4 and m[4]:
                                 mts, mnid = m[4]
                             if len(m) > 5 and m[5]:
                                 mdests = list(m[5])
                                 
                             members.append(p2p_pb2.VectorComplete(
                                 values=mv, id=mid, payload=mp, cluster_id=mcid,
                                 timestamp=mts, version_node_id=mnid, destinations=mdests
                             ))
                        
                        clusters_map[cid] = p2p_pb2.ClusterData(center=center, members=members)
                        
                    # Assignment is complex map of lists... use pickle for this part as established in plan
                    assign_bytes = pickle.dumps(assignment_in)
                    
                    req = p2p_pb2.SetClustersRequest(clusters=clusters_map, assignment_pickle=assign_bytes)
                    await stub.SetClusters(req)
                    response_val = None
                    
                elif query == "search_vectors_local":
                    # args[0] vectors (list of tuples), args[1] top_k
                    # Vector input here is usually list of (vector, id) or similar?
                    # Check usage: search_vectors_local(vectors, top_k)
                    # vectors arg in logic is ListOfVectorsWithId? [(id, vector)]?
                    # In server.py: "search_vectors_local(query, top_k)" where query is list.
                    # test_server.py: query = [([1.0, 0.0], 100)] -> (vector, id)
                    # Wait, proto def says VectorWithId: id, values.
                    
                    vecs_in = args[0]
                    top_k = args[1]
                    
                    proto_vecs = []
                    for v_item in vecs_in:
                         # v_item is (vector, id)
                         proto_vecs.append(p2p_pb2.VectorWithId(values=v_item[0], id=v_item[1]))
                    
                    req = p2p_pb2.SearchVectorsContext(vectors=proto_vecs, top_k=top_k)
                    resp = await stub.SearchVectorsLocal(req)
                    
                    # Convert response back to list of tuples
                    # [(vector, id, payload, sim)]
                    results = []
                    for item in resp.results:
                        results.append((list(item.vector), item.id, item.payload, item.similarity))
                    response_val = results
                    
                elif query == "query":
                    # args[0] vectors (list of (vec, id)), args[1] status
                    vecs_in = args[0]
                    status = args[1]
                    
                    proto_vecs = []
                    for v_item in vecs_in:
                         proto_vecs.append(p2p_pb2.VectorWithId(values=v_item[0], id=v_item[1]))
                         
                    req = p2p_pb2.QueryRequest(vectors=proto_vecs, status=status)
                    resp = await stub.Query(req)
                    
                    results = []
                    for item in resp.results:
                        results.append((list(item.vector), item.id, item.payload, item.similarity))
                    response_val = results

                elif query == "get_vector_digest":
                    resp = await stub.GetVectorDigest(p2p_pb2.Empty())
                    # Convert map<int, VersionInfo> to dict {id: (ts, nid)}
                    digest = {}
                    for vid, vinfo in resp.digest.items():
                        digest[vid] = (vinfo.timestamp, vinfo.node_id)
                    response_val = digest
                    
                elif query == "get_vectors_by_ids":
                    # args[0] list of ints
                    ids = args[0]
                    req = p2p_pb2.IdList(ids=ids)
                    resp = await stub.GetVectorsByIds(req)
                    
                    # Convert back to list of vector tuples
                    vecs_out = []
                    for v in resp.vectors:
                        # Reconstruct tuple
                        # (values, id, payload, cluster_id, version, destinations)
                        # destinations is list in proto, frozen set in python
                        dests = frozenset(v.destinations) if v.destinations else frozenset()
                        ver = (v.timestamp, v.version_node_id)
                        
                        # Check original tuple size expectation? 
                        # Use max size (6) for safety/completeness
                        vec_tuple = (list(v.values), v.id, v.payload, v.cluster_id, ver, dests)
                        vecs_out.append(vec_tuple)
                    response_val = vecs_out
                    
                elif query == "get_partition_coordinator_id":
                    resp = await stub.GetPartitionCoordinatorId(p2p_pb2.Empty())
                    response_val = resp.id
                    
                elif query == "respond_to_ping":
                    resp = await stub.RespondToPing(p2p_pb2.Empty())
                    response_val = resp.value
                    
                else:
                    return {"status": -1, "error": f"Unknown gRPC method: {query}", "response": None}

                # Return response directly (consistent with QUIC and updated HTTP)
                return {"status": 0, "error": None, "response": response_val}
                
            except grpc.RpcError as e:
                # print(f"GRPC Error {query}: {e.code()}")
                return {"status": -1, "error": str(e), "response": None}
            except Exception as e:
                print(f"GRPC Unexpected Error {query}: {e}")
                return {"status": -1, "error": str(e), "response": None}
