"""
Peer management endpoints: registration, listing, gossip.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import numpy as np
from api.models import RegisterPeerRequest

router = APIRouter(prefix="", tags=["peer"])


def get_node():
    """Dependency to get the current node instance."""
    # This will be set when the router is included in the app
    return router.node


@router.post("/register_peer")
async def register_peer_endpoint(request: RegisterPeerRequest, node=Depends(get_node)):
    """
    Register a new peer node and exchange representative vectors.
    """
    node.register_peer(request.peer_id, request.peer_url)
    
    # Normalize and cache peer representative vectors
    normalized = []
    try:
        for v in request.node_vectors:
            if len(v) != node.vector_size:
                raise ValueError(f"Peer {request.peer_id}: Vector size mismatch. Expected {node.vector_size}, got {len(v)}")
            arr = np.array(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            normalized.append(arr.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    node.peer_node_vectors[request.peer_id] = normalized
    print(f"Node {node.node_id}: Cached {len(normalized)} representative vectors for {request.peer_id}")

    if not node.node_vectors:
        print(f"Node {node.node_id}: ERROR: Peer registered but this node's representative vectors are not set.")
        raise HTTPException(status_code=500, detail="This node's representative vectors are not set.")

    return {
        "status": "success",
        "message": f"Peer {request.peer_id} registered",
        "total_peers": len(node.peer_nodes),
        "node_id": node.node_id,
        "node_vectors": node.node_vectors
    }


@router.get("/peers")
async def list_peers_endpoint(node=Depends(get_node)):
    """List all registered peer nodes and their cached vector status"""
    peers_with_vectors = {
        pid: {
            "url": url,
            "vector_count": len(node.peer_node_vectors.get(pid, [])),
            "status": node.peer_status.get(pid, {}).get("status", "UNKNOWN"),
            "last_ok": node.peer_status.get(pid, {}).get("last_ok"),
            "last_check": node.peer_status.get(pid, {}).get("last_check")
        }
        for pid, url in node.peer_nodes.items()
    }
    return {
        "node_id": node.node_id,
        "peers": peers_with_vectors,
        "peer_count": len(node.peer_nodes)
    }


@router.post("/gossip/clusters")
async def gossip_clusters_endpoint(payload: Dict[str, Any], node=Depends(get_node)):
    """
    Receive cluster updates from peers via gossip protocol.
    """
    from_node = payload.get("from_node")
    cluster_vectors = payload.get("cluster_vectors", [])
    
    if not cluster_vectors:
        return {"status": "ok", "message": "No clusters received"}
    
    success = node.receive_peer_clusters(from_node, cluster_vectors)
    
    # Send back own clusters
    local_cluster_ids = node.meta_hnsw.node_to_clusters.get(node.node_id, []) if node.meta_hnsw else []
    local_clusters = [node.meta_hnsw.cluster_centroids[cid].tolist() for cid in local_cluster_ids] if local_cluster_ids else []
    
    return {
        "status": "success" if success else "failed",
        "cluster_vectors": local_clusters
    }
