"""
Search endpoints: local, federated, P2P search with Meta-HNSW routing.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from api.models import SearchRequest

router = APIRouter(prefix="/search", tags=["search"])


def get_node():
    """Dependency to get the current node instance."""
    return router.node


@router.post("")
async def search_endpoint(request: SearchRequest, node=Depends(get_node)):
    """
    Search local Qdrant database (called by peer nodes)
    """
    results = node.search_local(request.query_vector, request.top_k)
    if results is not None:
        return results
    else:
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/local")
async def search_local_endpoint(query_vector: List[float], top_k: int = 5, node=Depends(get_node)):
    """
    Search only the local database
    """
    results = node.search_local(query_vector, top_k)
    if results is not None:
        return {
            "node_id": node.node_id,
            "results": results,
            "count": len(results)
        }
    else:
        raise HTTPException(status_code=500, detail="Local search failed")


@router.post("/federated")
async def federated_search_endpoint(query_vector: List[float], top_k: int = 5, node=Depends(get_node)):
    """
    Search across all nodes (local + peers)
    """
    results = node.federated_search(query_vector, top_k)
    total_results = sum(len(r) for r in results.values())
    return {
        "status": "success",
        "results": results,
        "nodes_searched": len(results),
        "total_results": total_results
    }


@router.post("/p2p")
async def search_p2p_endpoint(payload: Dict[str, Any], node=Depends(get_node)):
    """
    P2P search with server-side Meta-HNSW routing.
    Queries top-K nodes in parallel and aggregates results.
    """
    try:
        query_vector = np.array(payload['query_vector'], dtype=np.float32)
        k_nodes = payload.get('top_k_nodes', 3)
        k_results = payload.get('top_k_results', 5)
        
        print(f"Node {node.node_id}: P2P search entry point activated")
        print(f"  Query params: top_k_nodes={k_nodes}, top_k_results={k_results}")
        
        # STEP 1: Routing with Meta-HNSW
        if node.meta_hnsw is None:
            print(f"  ⚠️  No Meta-HNSW available, falling back to broadcast")
            target_nodes = [node.node_id] + list(node.peer_nodes.keys())
            target_nodes = target_nodes[:k_nodes]
        else:
            print(f"  🔍 Using local Meta-HNSW for routing...")
            nearest_nodes = node.meta_hnsw.find_nearest_nodes(query_vector, k_nodes=k_nodes)
            target_nodes = [node_name for node_name, _ in nearest_nodes]
            
            print(f"  📍 Meta-HNSW routing result: {target_nodes}")
            for node_name, distance in nearest_nodes:
                print(f"     - {node_name}: distance={distance:.4f}")
        
        # STEP 2: Parallel queries
        all_results = {}
        best_score = -2.0
        best_node = "N/A"
        
        with ThreadPoolExecutor(max_workers=len(target_nodes)) as executor:
            future_to_node = {}
            
            for target_node_id in target_nodes:
                if target_node_id == node.node_id:
                    print(f"  🔎 Submitting local search to executor...")
                    future = executor.submit(node.search_local, query_vector.tolist(), k_results)
                else:
                    print(f"  📡 Submitting remote search to peer {target_node_id}...")
                    future = executor.submit(node.query_peer, target_node_id, query_vector.tolist(), k_results)
                future_to_node[future] = target_node_id

            for future in as_completed(future_to_node):
                target_node_id = future_to_node[future]
                try:
                    results = future.result()
                    if results:
                        all_results[target_node_id] = results
                        
                        if results and results[0].get('score', -2) > best_score:
                            best_score = results[0]['score']
                            best_node = target_node_id
                            print(f"     ✓ New best match from {best_node} (score: {best_score:.4f})")
                except Exception as e:
                    print(f"  ❌ Error querying {target_node_id}: {e}")

        total_results = sum(len(r) for r in all_results.values())
        print(f"  ✅ P2P search complete: {total_results} results from {len(all_results)} nodes")
        
        return {
            "status": "success",
            "entry_node": node.node_id,
            "nodes_queried": len(all_results),
            "target_nodes": target_nodes,
            "total_results": total_results,
            "results_per_node": all_results,
            "best_match": {"node": best_node, "score": best_score},
            "routing_method": "meta_hnsw" if node.meta_hnsw else "broadcast"
        }
        
    except Exception as e:
        print(f"Node {node.node_id}: P2P search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/local/string")
async def search_local_string_endpoint(payload: Dict[str, Any], top_k: int = 5, node=Depends(get_node)):
    """Search local database using text string (generates embedding)"""
    try:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string")

        svc = getattr(node, "embedding_service", None)
        if svc is None:
            raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

        embedding = svc.get_embedding(text)
        results = node.search_local(embedding, top_k)
        
        if results is not None:
            return {"node_id": node.node_id, "results": results, "count": len(results)}
        else:
            raise HTTPException(status_code=500, detail="Local search failed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/federated/string")
async def federated_search_string_endpoint(payload: Dict[str, Any], top_k: int = 5, node=Depends(get_node)):
    """Federated search using text string"""
    try:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string")

        svc = getattr(node, "embedding_service", None)
        if svc is None:
            raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

        embedding = svc.get_embedding(text)
        results = node.federated_search(embedding, top_k)
        total_results = sum(len(r) for r in results.values())
        
        return {
            "status": "success",
            "results": results,
            "nodes_searched": len(results),
            "total_results": total_results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
