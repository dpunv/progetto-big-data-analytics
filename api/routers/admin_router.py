"""
Admin endpoints: configuration, monitoring, topology.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import socket
import requests
import numpy as np

from api.models import SyncRequest

router = APIRouter(prefix="", tags=["admin"])


def get_node():
    """Dependency to get the current node instance."""
    return router.node


@router.get("/count")
async def count_endpoint(node=Depends(get_node)):
    """Get the current vector count for this node"""
    count = node.count_local_vectors()
    if count != -1:
        return {"node_id": node.node_id, "count": count}
    else:
        raise HTTPException(status_code=500, detail="Failed to get count")


@router.post("/set_node_vectors")
async def set_node_vectors_endpoint(vectors: List[List[float]], node=Depends(get_node)):
    """Set multiple representative vectors for this node"""
    try:
        node.set_node_vectors(vectors)
        return {
            "status": "success",
            "node_id": node.node_id,
            "message": f"Set {len(node.node_vectors)} representative vectors",
            "node_vectors": node.node_vectors
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sync")
async def sync_vector_endpoint(request: SyncRequest, node=Depends(get_node)):
    """Sync a specific vector to a target node"""
    success = node.sync_vector(request.vector_id, request.target_node_id)
    if success:
        return {
            "status": "success",
            "message": f"Vector {request.vector_id} synced to {request.target_node_id}"
        }
    else:
        raise HTTPException(status_code=500, detail=f"Failed to sync vector {request.vector_id}")


@router.post("/get-embedding")
async def get_embedding_endpoint(payload: Dict[str, Any], node=Depends(get_node)):
    """Generate embedding from text"""
    try:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string")

        svc = getattr(node, "embedding_service", None)
        if svc is None:
            raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

        embedding = svc.get_embedding(text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add-vector/string")
async def add_vector_string_endpoint(payload: Dict[str, Any], node=Depends(get_node)):
    """Add vector from text string (generates embedding and routes to replicas)"""
    from server import VectorData  # Import here to avoid circular dependency
    
    try:
        text = payload.get("text")
        vector_id = payload.get("id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Field 'text' must be a non-empty string")
        if not isinstance(vector_id, str) or not vector_id.strip():
            raise ValueError("Field 'id' must be a non-empty string")

        svc = getattr(node, "embedding_service", None)
        if svc is None:
            raise HTTPException(status_code=500, detail="Embedding service not initialized on this node")

        embedding = svc.get_embedding(text)
        vec_data = VectorData(id=vector_id, vector=embedding, payload={"text": text})

        replica_node_ids = node.find_best_nodes(vec_data.vector)
        print(f"Node {node.node_id}: Routing vector {vec_data.id} to {len(replica_node_ids)} replicas: {replica_node_ids}")

        success_nodes = []
        failed_nodes = []

        for node_id in replica_node_ids:
            success = False
            if node_id == node.node_id:
                print(f"Node {node.node_id}: Storing vector {vec_data.id} locally (replica).")
                success = node.receive_vector(from_node_id="external_client_routed", vector_data=vec_data)
            else:
                print(f"Node {node.node_id}: Forwarding vector {vec_data.id} to replica {node_id}.")
                success = node.send_vector(node_id, vec_data)

            if success:
                success_nodes.append(node_id)
            else:
                failed_nodes.append(node_id)

        if not success_nodes:
            raise HTTPException(status_code=500, detail=f"Failed to store vector {vector_id} on any replica.")

        return {
            "status": "success",
            "message": f"Vector {vector_id} routed to {len(replica_node_ids)} replicas.",
            "vector_id": vector_id,
            "replicas_targeted": replica_node_ids,
            "replicas_succeeded": success_nodes,
            "replicas_failed": failed_nodes
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init-meta-hnsw")
async def init_meta_hnsw_endpoint(payload: Dict[str, Any], node=Depends(get_node)):
    """Initialize Meta-HNSW with cluster data from coordinator"""
    try:
        dimension = payload.get("dimension", node.vector_size)
        max_clusters = payload.get("max_clusters", 500)
        all_centroids = payload.get("centroids", [])
        node_assignments = payload.get("node_assignments", {})
        
        node.initialize_meta_hnsw(dimension=dimension, max_clusters=max_clusters)
        
        for node_idx_str, cluster_indices in node_assignments.items():
            peer_node_id = f"node{int(node_idx_str) + 1}"
            cluster_vectors = [all_centroids[idx] for idx in cluster_indices]
            
            if peer_node_id == node.node_id:
                node.add_local_clusters(cluster_vectors)
            else:
                node.receive_peer_clusters(peer_node_id, cluster_vectors)
        
        node.meta_hnsw.force_rebuild()
        
        print(f"Node {node.node_id}: Meta-HNSW initialized with {len(all_centroids)} total clusters")
        
        return {
            "status": "success",
            "message": f"Meta-HNSW initialized with {len(all_centroids)} clusters",
            "node_id": node.node_id
        }
    except Exception as e:
        print(f"Node {node.node_id}: Error initializing Meta-HNSW: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-topology")
async def get_topology(node=Depends(get_node)):
    """Generate network topology from this node's perspective"""
    visited = set()
    topology = {}

    def _parse_url_info(url: Optional[str]):
        if not url:
            return (None, None, None)
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            ip = None
            try:
                if hostname:
                    ip = socket.gethostbyname(hostname)
            except Exception:
                ip = None
            return (hostname, ip, port)
        except Exception:
            return (None, None, None)

    def traverse(node_id: str, node_url: Optional[str]):
        if node_id in visited:
            return
        visited.add(node_id)

        use_url = node_url if node_url else (node.self_url if node_id == node.node_id else None)
        dns_name, ip_addr, port_num = _parse_url_info(use_url)

        peers_info: Dict[str, Dict[str, Optional[Any]]] = {}

        if node_id == node.node_id:
            peers = node.peer_nodes
            for pid, purl in peers.items():
                pdns, pip, pport = _parse_url_info(purl)
                peers_info[pid] = {"dns": pdns, "ip": pip, "port": pport}
        else:
            if not node_url:
                topology[node_id] = {"dns": dns_name, "ip": ip_addr, "port": port_num, "peers": {}}
                return
            try:
                resp = requests.get(f"{node_url}/peers", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    remote_peers = data.get("peers", {})
                    for pid, info in remote_peers.items():
                        purl = info.get("url")
                        pdns, pip, pport = _parse_url_info(purl)
                        peers_info[pid] = {"dns": pdns, "ip": pip, "port": pport}
                else:
                    peers_info = {}
            except requests.RequestException:
                peers_info = {}

        topology[node_id] = {"dns": dns_name, "ip": ip_addr, "port": port_num, "peers": peers_info}

        for pid, info in peers_info.items():
            if pid not in visited:
                peer_url = node.peer_nodes.get(pid)
                if not peer_url:
                    pip = info.get("ip")
                    pport = info.get("port")
                    pdns = info.get("dns")
                    if pip:
                        peer_url = f"http://{pip}:{pport}"
                    elif pdns:
                        peer_url = f"http://{pdns}:{pport}"
                traverse(pid, peer_url)

    traverse(node.node_id, node.self_url)

    topology_list = [
        {
            "node_id": node_id,
            "details": details,
            "peers": list(details["peers"].keys())
        }
        for node_id, details in topology.items()
    ]
    
    return {"node_id": node.node_id, "topology": topology_list}
