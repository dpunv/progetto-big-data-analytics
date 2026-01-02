import aiohttp
import os
import pickle
import base64
from typing import Callable, Any, Union
from .base import BaseCommunicator

class HTTPCommunicator(BaseCommunicator):
    async def send(self, ip: str, port: int, query: str, *args):
        base_url = f"http://{ip}:{port}"
        
        json_payload = {}
        target_endpoint = f"/{query}"
        
        # Prepare payload based on query type
        if query == "get_id":
            # No args
            pass
            
        elif query == "similarity":
            # args[0] is vector
            json_payload = {'vector': args[0]}
            
        elif query == "receive":
            # args[0] is list of vectors, args[1] is status
            # Vector structure: (values, id, payload, cluster_id, [ver, dest])
            vectors_data = args[0]
            status = args[1]
            
            # Serialize vectors to list of dicts/lists for JSON
            # We keep it simple: list of lists/objects
            # Since JSON handles lists natively, we just need to ensure everything inside is serializable.
            # Vectors are typically lists of floats.
            # tuples will become lists.
            # sets will need conversion to lists.
            
            serializable_vectors = []
            for v in vectors_data:
                # v is a tuple, convert to list for mutation/serialization
                v_list = list(v)
                
                # Check for destinations set at expected index 5
                if len(v_list) > 5:
                    if isinstance(v_list[5], (set, frozenset)):
                        v_list[5] = list(v_list[5])
                
                serializable_vectors.append(v_list)
                
            json_payload = {'vectors': serializable_vectors, 'status': status}

        elif query == "i_am_coord":
             pass
             
        elif query == "set_clusters":
            # args[0] is clusters dict, args[1] is assignment (List of tuples)
            clusters_in = args[0]
            assignment_in = args[1]
            
            # clusters_in: {int_id: {'center': [float], 'members': [[vec...]]}}
            # JSON keys must be strings.
            clusters_serializable = {}
            for cid, data in clusters_in.items():
                # Members might have sets in them
                ser_members = []
                for m in data['members']:
                    m_list = list(m)
                    if len(m_list) > 5 and isinstance(m_list[5], (set, frozenset)):
                        m_list[5] = list(m_list[5])
                    ser_members.append(m_list)
                
                clusters_serializable[str(cid)] = {
                    'center': data['center'],
                    'members': ser_members
                }
            
            # assignment_in: [(cluster_id, center), ...]
            # assignment is list of tuples. JSON handles it as list of lists.
            # We can just pass it directly if center is list.
            # Just to be safe and consistent with other serializations:
            assignment_serializable = []
            for item in assignment_in:
                 # item is (cid, center)
                 assignment_serializable.append([item[0], item[1]])
                
            json_payload = {'clusters': clusters_serializable, 'assignment': assignment_serializable}
            
        elif query == "search_vectors_local":
            # args[0] vectors (list of (vec, id)), args[1] top_k
            # vec is list of floats, id is int.
            vecs_in = []
            for v_item in args[0]:
                vecs_in.append({'values': v_item[0], 'id': v_item[1]})
                
            json_payload = {'vectors': vecs_in, 'top_k': args[1]}
            
        elif query == "query":
            # args[0] vectors (list of (vec, id)), args[1] status
            vecs_in = []
            for v_item in args[0]:
                vecs_in.append({'values': v_item[0], 'id': v_item[1]})
                
            json_payload = {'vectors': vecs_in, 'status': args[1]}
            
        elif query == "get_vector_digest":
            pass
            
        elif query == "get_vectors_by_ids":
             # args[0] list of ints
             json_payload = {'ids': args[0]}
             
        elif query == "get_partition_coordinator_id":
            pass
            
        elif query == "respond_to_ping":
            pass
            
        else:
            return {"status": -1, "error": f"Unknown HTTP query: {query}", "response": None}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(base_url + target_endpoint, json=json_payload) as response:
                    if response.status != 200:
                         text = await response.text()
                         return {"status": -1, "error": f"HTTP {response.status}: {text}", "response": None}
                    
                    resp_json = await response.json()
                    
                    # Post-process response if needed (convert structure back to what peer expects)
                    # p2p_pb2 returns objects, existing http returns dict/objects via pickle.
                    # We need to match what grpc_comm returns in structure.
                    
                    response_val = resp_json.get('response')
                    
                    # Special handling for types that lost info in JSON (e.g. keys are strings now)
                    if query == "get_vector_digest":
                        # response_val is dict {str_id: [ts, nid]} -> convert to {int: tuple}
                        new_digest = {}
                        if response_val:
                            for k, v in response_val.items():
                                new_digest[int(k)] = tuple(v)
                        response_val = new_digest
                        
                    elif query == "search_vectors_local" or query == "query":
                         # response_val is list of [vec, id, payload, sim]
                         # Ensure tuples if expected
                         if response_val:
                             response_val = [tuple(x) for x in response_val]
                             
                    elif query == "get_vectors_by_ids":
                        # response_val is list of vectors
                        # Restore tuples and sets
                        if response_val:
                            new_vecs = []
                            for v in response_val:
                                v_list = list(v)
                                if len(v_list) > 5:
                                    # dests back to frozenset? server usually expects iterables or sets.
                                    # In peer/server logic, it often converts to set/frozenset.
                                    pass # List is fine, will be converted by receiver if needed?
                                    # Actually, let's look at grpc_comm. It returns tuples.
                                    # And dests as frozenset.
                                    if v_list[5]:
                                         v_list[5] = frozenset(v_list[5])
                                    else:
                                         v_list[5] = frozenset()
                                         
                                # Version (ts, nid) is list in JSON
                                if len(v_list) > 4:
                                     v_list[4] = tuple(v_list[4])
                                     
                                new_vecs.append(tuple(v_list))
                            response_val = new_vecs

                    return {"status": 0, "error": None, "response": response_val}
                            
            except Exception as e:
                # print(f"HTTP Send failed: {e}")
                return {"status": -1, "error": str(e), "response": None}
