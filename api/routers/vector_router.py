"""
Vector insertion and forwarding endpoints.
PUBLIC API: Receives JSON from external clients
INTERNAL: Delegates to gRPC for P2P communication
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from typing import Dict, List, Tuple
from collections import defaultdict

from api.models import VectorDataModel, SendVectorRequest, BroadcastRequest
from utils.serialization import deserialize_request
from server import VectorData

router = APIRouter(prefix="", tags=["vectors"])


def get_node():
    """Dependency to get the current node instance."""
    return router.node


@router.post("/receive_vector")
async def receive_vector_endpoint(request: SendVectorRequest, node=Depends(get_node)):
    """
    Endpoint to receive a SINGLE vector from other nodes.
    NOTE: This endpoint is kept for backward compatibility but internal P2P uses gRPC.
    """
    vector_data = VectorData(
        id=request.vector_data.id,
        vector=request.vector_data.vector,
        payload=request.vector_data.payload
    )
    success = node.receive_vector(request.from_node, vector_data)
    if success:
        return {
            "status": "success",
            "message": f"Vector {vector_data.id} stored successfully",
            "node_id": node.node_id
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to store vector")


@router.post("/receive_vectors_bulk")
async def receive_vectors_bulk_endpoint(request: Request, node=Depends(get_node)):
    """
    Endpoint to receive a BATCH of vectors.
    NOTE: Public endpoint (JSON). Internal P2P uses gRPC directly.
    """
    try:
        # Deserialize JSON (for external clients)
        payload = await deserialize_request(request)
        
        from_node = payload.get("from_node")
        vectors_data_raw = payload.get("vectors_data", [])
        
        # Convert to VectorData dataclass
        vectors_data_list = [
            VectorData(
                id=vd['id'],
                vector=vd['vector'],
                payload=vd.get('payload', {})
            ) for vd in vectors_data_raw
        ]
        
        success = node.receive_vectors_bulk(from_node, vectors_data_list)
        
        if success:
            return {
                "status": "success",
                "message": f"Stored {len(vectors_data_list)} vectors",
                "node_id": node.node_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store bulk vectors")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")


@router.post("/send_vector/{target_node_id}")
async def send_vector_endpoint(target_node_id: str, vector_data: VectorDataModel, node=Depends(get_node)):
    """
    Send a vector to a specific peer node.
    NOTE: Public endpoint. Uses gRPC internally.
    """
    vec_data = VectorData(
        id=vector_data.id,
        vector=vector_data.vector,
        payload=vector_data.payload
    )
    success = node.send_vector(target_node_id, vec_data)
    if success:
        return {
            "status": "success",
            "message": f"Vector sent to {target_node_id}",
            "vector_id": vector_data.id
        }
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"Failed to send vector to {target_node_id}"
        )


@router.post("/broadcast")
async def broadcast_vector_endpoint(request: BroadcastRequest, node=Depends(get_node)):
    """
    Store a vector locally AND broadcast it to all peer nodes.
    """
    vector_data = VectorData(
        id=request.vector_data.id,
        vector=request.vector_data.vector,
        payload=request.vector_data.payload
    )
    
    print(f"Node {node.node_id}: Storing broadcast vector {vector_data.id} locally...")
    local_success = node.receive_vector(node.node_id, vector_data)
    if not local_success:
        print(f"Node {node.node_id}: WARNING - Failed to store broadcast vector locally.")

    broadcast_results = node.broadcast_vector(vector_data)
    return {
        "status": "success",
        "vector_id": vector_data.id,
        "local_storage": "success" if local_success else "failed",
        "broadcast_results": broadcast_results,
        "success_count": sum(broadcast_results.values()),
        "total_peers": len(broadcast_results)
    }


@router.post("/add_vector")
async def add_vector_endpoint(vector_data: VectorDataModel, node=Depends(get_node)):
    """
    Add a new SINGLE vector from an external client (JSON).
    This node will find the best cluster and route via gRPC to replica nodes.
    """
    vec_data = VectorData(
        id=vector_data.id,
        vector=vector_data.vector,
        payload=vector_data.payload
    )
    
    # Find all replica nodes for this vector
    replica_node_ids = node.find_best_nodes(vec_data.vector)
    
    print(f"Node {node.node_id}: Routing vector {vec_data.id} to {len(replica_node_ids)} replicas: {replica_node_ids}")

    success_nodes = []
    failed_nodes = []

    # Send to all replicas (uses gRPC for peer forwarding)
    for node_id in replica_node_ids:
        success = False
        if node_id == node.node_id:
            print(f"Node {node.node_id}: Storing vector {vec_data.id} locally (replica).")
            success = node.receive_vector(
                from_node_id="external_client_routed", 
                vector_data=vec_data
            )
        else:
            print(f"Node {node.node_id}: Forwarding vector {vec_data.id} to replica {node_id} via gRPC.")
            success = node.send_vector(node_id, vec_data)
        
        if success:
            success_nodes.append(node_id)
        else:
            failed_nodes.append(node_id)

    if not success_nodes:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to store vector {vector_data.id} on any replica."
        )

    return {
        "status": "success",
        "message": f"Vector {vector_data.id} routed to {len(replica_node_ids)} replicas via gRPC.",
        "action": "routed_to_replicas",
        "replicas_targeted": replica_node_ids,
        "replicas_succeeded": success_nodes,
        "replicas_failed": failed_nodes
    }


@router.post("/add_vectors_bulk")
async def add_vectors_bulk_endpoint(request: Request, background_tasks: BackgroundTasks, node=Depends(get_node)):
    """
    Add a new BATCH of vectors from an external client (JSON).
    PUBLIC ENDPOINT: Receives JSON, routes via gRPC to ALL replica nodes.
    
    FIXED: Properly implements replication factor routing.
    """
    try:
        # Deserialize JSON from external client
        vectors_raw = await deserialize_request(request)
        
        # Convert to VectorDataModel (Pydantic validation)
        vectors = [
            VectorDataModel(
                id=v['id'],
                vector=v['vector'],
                payload=v.get('payload', {})
            ) for v in vectors_raw
        ]
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
    
    # Classify all vectors and group them by target replica nodes
    nodes_to_vectors: Dict[str, List[VectorData]] = defaultdict(list)
    total_routings = 0
    
    for vd_model in vectors:
        vec_data = VectorData(
            id=vd_model.id,
            vector=vd_model.vector,
            payload=vd_model.payload
        )
        
        # Find ALL nodes responsible for this vector
        replica_node_ids = node.find_best_nodes(vec_data.vector)
        
        # Add the vector to the batch of EACH replica node
        for node_id in replica_node_ids:
            nodes_to_vectors[node_id].append(vec_data)
            total_routings += 1
    
    num_vectors = len(vectors)
    replication_factor_achieved = total_routings / num_vectors if num_vectors > 0 else 0
    
    print(f"Node {node.node_id}: Received bulk of {num_vectors}. Routing to {len(nodes_to_vectors)} nodes (total routings: {total_routings}) via gRPC.")
    print(f"  Replication factor achieved: {replication_factor_achieved:.1f}x")

    # Process/forward the batches in the background (gRPC streaming)
    routing_summary = {}
    
    for target_node_id, vectors_list in nodes_to_vectors.items():
        batch_size = len(vectors_list)
        routing_summary[target_node_id] = batch_size
        
        if target_node_id == node.node_id:
            print(f"Node {node.node_id}: Queuing local storage of {batch_size} vectors (includes replicas).")
            background_tasks.add_task(
                node.receive_vectors_bulk,
                from_node_id="external_client_routed",
                vectors_data=vectors_list
            )
        else:
            print(f"Node {node.node_id}: Queuing gRPC forward of {batch_size} vectors to {target_node_id} (includes replicas).")
            background_tasks.add_task(
                node.send_vectors_bulk,
                target_node_id=target_node_id,
                vectors_data=vectors_list
            )
    
    return {
        "status": "processing_bulk",
        "message": f"Processing {num_vectors} vectors with {replication_factor_achieved:.1f}x replication via gRPC. Total routings: {total_routings} across {len(nodes_to_vectors)} nodes.",
        "batches": routing_summary,
        "transport": "gRPC_streaming",
        "replication_factor": replication_factor_achieved
    }
