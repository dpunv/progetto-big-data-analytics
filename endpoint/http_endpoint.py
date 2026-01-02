from .base import BaseEndpoint
from aiohttp import web
import pickle
import base64
import asyncio

class HTTPEndpoint(BaseEndpoint):
    async def start(self, ip: str, port: int):
        # Ensure max client size is large enough for vectors
        # 100MB limit (default is usually small)
        self.app = web.Application(client_max_size=1024*1024*100)
        
        # Explicit routes
        self.app.router.add_post('/get_id', self.handle_get_id)
        self.app.router.add_post('/similarity', self.handle_similarity)
        self.app.router.add_post('/receive', self.handle_receive)
        self.app.router.add_post('/i_am_coord', self.handle_i_am_coord)
        self.app.router.add_post('/set_clusters', self.handle_set_clusters)
        self.app.router.add_post('/search_vectors_local', self.handle_search_vectors_local)
        self.app.router.add_post('/query', self.handle_query)
        self.app.router.add_post('/get_vector_digest', self.handle_get_vector_digest)
        self.app.router.add_post('/get_vectors_by_ids', self.handle_get_vectors_by_ids)
        self.app.router.add_post('/get_partition_coordinator_id', self.handle_get_partition_coordinator_id)
        self.app.router.add_post('/respond_to_ping', self.handle_respond_to_ping)
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, ip, port)
        await self.site.start()
        print(f"HTTP Endpoint started on {ip}:{port}")
        
    async def _run_sync(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args))

    # Helper to return JSON response
    def json_response(self, data):
        return web.json_response({'response': data})

    async def handle_get_id(self, request):
        try:
            res = await self._run_sync(self.server.get_id)
            return self.json_response(res)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_similarity(self, request):
        try:
            data = await request.json()
            vec = data.get('vector')
            res = await self._run_sync(self.server.similarity, vec)
            return self.json_response(res)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_receive(self, request):
        try:
            data = await request.json()
            vectors_in = data.get('vectors') # List of lists
            status = data.get('status')
            
            # Reconstruct tuples from lists
            vectors_tuples = []
            for v_list in vectors_in:
                # v_list: [values, id, payload, cluster_id, ver, dest]
                # Convert back to tuple structure expected by server
                # Handle types: values is list, id int, payload str, cid int.
                # ver is list [ts, nid], dest is list [ids]
                
                # Check list length to be safe
                ts = 0.0
                nid = 0
                dests = []
                
                if len(v_list) > 4:
                     # Version
                     # In JSON, version might be list or null
                     if v_list[4]:
                         ts = v_list[4][0]
                         nid = v_list[4][1]
                         
                if len(v_list) > 5:
                    if v_list[5]:
                        dests = list(v_list[5]) # kept as list or convert to set?
                        # Start with list, eventually code converts to frozenset if needed
                        
                # Tuple construction
                # (vec, id, payload, cid, (ts, nid), dests)
                # Ensure values is list of floats
                values = v_list[0]
                vid = v_list[1]
                payload = v_list[2]
                cid = v_list[3]
                
                ver_tuple = (ts, nid)
                dests_set = frozenset(dests)
                
                # Create the tuple. Length 6 is standard now.
                vec_tuple = (values, vid, payload, cid, ver_tuple, dests_set)
                vectors_tuples.append(vec_tuple)
                
            await self._run_sync(self.server.receive, vectors_tuples, status)
            return self.json_response(None)
        except Exception as e:
            print(f"Error HTTP Receive: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def handle_i_am_coord(self, request):
        try:
            res = await self._run_sync(self.server.i_am_coord)
            return self.json_response(res)
        except Exception:
            return self.json_response(False)

    async def handle_set_clusters(self, request):
        try:
            data = await request.json()
            clusters_raw = data.get('clusters')
            assignment_raw = data.get('assignment')
            
            # Reconstruct clusters: keys from str to int
            clusters = {}
            for cid_str, cdata in clusters_raw.items():
                cid = int(cid_str)
                center = cdata['center']
                members_raw = cdata['members']
                
                members = []
                for m_list in members_raw:
                    # Same reconstruction as receive, logic duplicated slightly but necessary
                    ts = 0.0
                    nid = 0
                    dests = frozenset()
                    
                    if len(m_list) > 4 and m_list[4]:
                        ts = m_list[4][0]
                        nid = m_list[4][1]
                    if len(m_list) > 5 and m_list[5]:
                        dests = frozenset(m_list[5])
                        
                    mt = (m_list[0], m_list[1], m_list[2], m_list[3], (ts, nid), dests)
                    members.append(mt)
                    
                clusters[cid] = {'center': center, 'members': members}
                
            # Reconstruct assignment: keys from str to int
            # assignment_raw is list of [cid, center]
            assignment = []
            if assignment_raw:
                for item in assignment_raw:
                     # item is [cid, center]
                     assignment.append((item[0], item[1]))
                 
            await self._run_sync(self.server.set_clusters, clusters, assignment)
            return self.json_response(None)
        except Exception as e:
            print(f"Error SetClusters: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def handle_search_vectors_local(self, request):
        try:
            data = await request.json()
            vecs_in = data.get('vectors') # List of dicts {values, id}
            top_k = data.get('top_k')
            
            q_vecs = []
            for item in vecs_in:
                q_vecs.append((item['values'], item['id']))
                
            results = await self._run_sync(self.server.search_vectors_local, q_vecs, top_k)
            # results: list of (vector, id, payload, sim)
            # Convert numpy arrays to lists
            json_results = []
            if results:
                for r in results:
                    # r[0] is ndarray
                    v_list = r[0].tolist() if hasattr(r[0], 'tolist') else list(r[0])
                    json_results.append((v_list, r[1], r[2], r[3]))
            
            return self.json_response(json_results)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_query(self, request):
        try:
            data = await request.json()
            vecs_in = data.get('vectors')
            status = data.get('status')
            
            q_vecs = []
            for item in vecs_in:
                q_vecs.append((item['values'], item['id']))
            
            results = await self._run_sync(self.server.query, q_vecs, status)
            json_results = []
            if results:
                 for r in results:
                    v_list = r[0].tolist() if hasattr(r[0], 'tolist') else list(r[0])
                    json_results.append((v_list, r[1], r[2], r[3]))
            return self.json_response(json_results)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_get_vector_digest(self, request):
        try:
            digest = await self._run_sync(self.server.get_vector_digest)
            # digest: {int_id: (ts, nid)}
            # keys to str for JSON
            json_digest = {}
            for k, v in digest.items():
                json_digest[str(k)] = v
            return self.json_response(json_digest)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_get_vectors_by_ids(self, request):
        try:
            data = await request.json()
            ids = data.get('ids')
            vecs = await self._run_sync(self.server.get_vectors_by_ids, ids)
            
            # vecs is list of tuples.
            # tuples -> lists for JSON.
            
            json_vecs = []
            for v in vecs:
                v_list = list(v)
                # v[0] might be numpy array
                if hasattr(v_list[0], 'tolist'):
                    v_list[0] = v_list[0].tolist()
                elif isinstance(v_list[0], (list, tuple)):
                    pass # Already list
                else: 
                     # Should be list/array
                     pass
                     
                # dests at 5
                if len(v_list) > 5 and isinstance(v_list[5], (set, frozenset)):
                    v_list[5] = list(v_list[5])
                json_vecs.append(v_list)
                
            return self.json_response(json_vecs)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_get_partition_coordinator_id(self, request):
        try:
            res = await self._run_sync(self.server.get_partition_coordinator_id)
            return self.json_response(res)
        except Exception:
            return self.json_response(-1)

    async def handle_respond_to_ping(self, request):
        try:
            res = await self._run_sync(self.server.respond_to_ping)
            return self.json_response(res)
        except Exception:
             return self.json_response(False)

    async def stop(self):
        if hasattr(self, 'runner'):
            await self.runner.cleanup()
