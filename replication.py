"""
Sistema di replicazione intelligente per load balancing e fault tolerance.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import time
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


class ReplicationManager:
    """Gestisce repliche intelligenti basate su carico e accessi."""
    
    def __init__(self, replication_factor: int = 2):
        self.replication_factor = replication_factor
        self.cluster_access_counts = defaultdict(int)
        self.last_access_reset = time.time()
        self.stats_reset_interval = 3600
        
    def record_cluster_access(self, cluster_id: str):
        """Registra accesso a un cluster."""
        self.cluster_access_counts[cluster_id] += 1
        
        if time.time() - self.last_access_reset > self.stats_reset_interval:
            self.cluster_access_counts.clear()
            self.last_access_reset = time.time()
    
    def calculate_replication_plan(self,
                                   node_loads: Dict[str, int],
                                   cluster_loads: Dict[str, int],
                                   routing_table,
                                   threshold_pct: float = 0.85,
                                   force_replication: bool = False,
                                   min_replicas_per_cluster: int = None,
                                   max_replicas_per_node: int = None) -> List[Dict]:
        """
        Calcola piano di replicazione.
        
        Args:
            node_loads: {node_name: num_vectors}
            cluster_loads: {cluster_id: num_query_accesses}
            routing_table: RoutingTable
            threshold_pct: Soglia capacità
            force_replication: Forza replica hot cluster
            min_replicas_per_cluster: Minimo repliche per cluster (2=primary+1replica)
        """
        avg_load = sum(node_loads.values()) / len(node_loads) if node_loads else 0
        capacity = avg_load * 1.5
        threshold = capacity * threshold_pct
        
        overloaded = {n: load for n, load in node_loads.items() if load > threshold}
        underloaded = {n: load for n, load in node_loads.items() if load < avg_load * 0.7}
        
        plan = []
        
        # ═══════════════════════════════════════════════════════════════
        # NUOVO: Min replicas policy
        # ═══════════════════════════════════════════════════════════════
        
        if min_replicas_per_cluster and min_replicas_per_cluster >= 2:
            print(f"\n  🔄 Min replicas mode: ensuring {min_replicas_per_cluster} copies per cluster")
            
            node_replica_counts = defaultdict(int)
            
            # Conta repliche già esistenti
            for cluster_id_str, nodes in routing_table.replicas.items():
                for replica_node in nodes[1:]:
                    node_replica_counts[replica_node] += 1
            
            if max_replicas_per_node:
                print(f"     Max replicas per node: {max_replicas_per_node}")
                print(f"     Current replica counts: {dict(node_replica_counts)}")
            
            total_clusters = max(cluster_loads.keys(), key=lambda x: int(x), default='0')
            total_clusters_int = int(total_clusters) + 1 if total_clusters else 10
            
            # ═══════════════════════════════════════════════════════════════
            # MODIFICATO: Copia node_loads per tracking carico stimato
            # ═══════════════════════════════════════════════════════════════
            estimated_node_loads = node_loads.copy()
            
            for cluster_id in range(total_clusters_int):
                cluster_id_str = str(cluster_id)
                
                existing_nodes = routing_table.get_all_nodes_for_cluster(cluster_id_str)
                current_replicas = len(existing_nodes)
                
                needed_replicas = min_replicas_per_cluster - current_replicas
                
                if needed_replicas <= 0:
                    continue
                
                try:
                    source_node = routing_table.get_primary_node(cluster_id_str)
                except:
                    continue
                
                replicas_created_for_cluster = 0
                
                for _ in range(needed_replicas):
                    # ═══════════════════════════════════════════════════════════════
                    # FIX: Re-sort nodi OGNI VOLTA prima di scegliere target
                    # ═══════════════════════════════════════════════════════════════
                    
                    # Ordina nodi per carico stimato (include repliche già assegnate)
                    nodes_by_space = sorted(estimated_node_loads.items(), key=lambda x: x[1])
                    
                    target_node = None
                    
                    for candidate_node, candidate_load in nodes_by_space:
                        # Skip se è già in existing_nodes
                        if candidate_node in existing_nodes or candidate_node == source_node:
                            continue
                        
                        # Controlla limite repliche
                        if max_replicas_per_node is not None:
                            if node_replica_counts[candidate_node] >= max_replicas_per_node:
                                continue
                        
                        # Candidato valido trovato
                        target_node = candidate_node
                        break
                    
                    if not target_node:
                        print(f"     ⚠️  Cluster {cluster_id}: cannot create replica {replicas_created_for_cluster + 1}/{needed_replicas}")
                        print(f"         Reason: all nodes are at max_replicas_per_node={max_replicas_per_node} or already have this cluster")
                        break
                    
                    estimated_vectors = node_loads.get(source_node, 0) // max(1, len(routing_table.get_clusters_for_node(source_node)))
                    
                    plan.append({
                        'cluster_id': cluster_id,
                        'source_node': source_node,
                        'target_node': target_node,
                        'estimated_vectors': estimated_vectors,
                        'query_count': cluster_loads.get(cluster_id_str, 0),
                        'reason': 'min_replicas_policy'
                    })
                    
                    # ═══════════════════════════════════════════════════════════════
                    # FIX: Aggiorna carico stimato SUBITO dopo assegnazione
                    # ═══════════════════════════════════════════════════════════════
                    
                    estimated_node_loads[target_node] += estimated_vectors
                    node_replica_counts[target_node] += 1
                    
                    # Aggiorna existing_nodes per evitare doppie repliche
                    existing_nodes.append(target_node)
                    
                    replicas_created_for_cluster += 1
            
            if plan:
                print(f"     Created {len(plan)} replicas to satisfy min_replicas={min_replicas_per_cluster}")
                
                # ═══════════════════════════════════════════════════════════════
                # NUOVO: Statistiche distribuzione repliche
                # ═══════════════════════════════════════════════════════════════
                print(f"\n     📊 Replica Distribution:")
                for node_name in sorted(routing_table._node_names):
                    count = node_replica_counts.get(node_name, 0)
                    status = "✓" if count <= (max_replicas_per_node or float('inf')) else "⚠️"
                    print(f"       {status} {node_name}: {count} replicas")
            
            return plan
        
        # ═══════════════════════════════════════════════════════════════
        # Force replication mode
        # ═══════════════════════════════════════════════════════════════
        
        if force_replication and not overloaded:
            print(f"\n  🔄 Force replication mode: creating replicas even with balanced nodes")
            
            top_hot_clusters = sorted(cluster_loads.items(), key=lambda x: -x[1])[:3]
            
            if not top_hot_clusters:
                return []
            
            nodes_by_space = sorted(node_loads.items(), key=lambda x: x[1])
            
            for cluster_id, query_count in top_hot_clusters:
                try:
                    source_node = routing_table.get_primary_node(str(cluster_id))
                except:
                    continue
                
                existing_nodes = routing_table.get_all_nodes_for_cluster(str(cluster_id))
                
                target_node = None
                for candidate_node, candidate_load in nodes_by_space:
                    if candidate_node not in existing_nodes and candidate_node != source_node:
                        target_node = candidate_node
                        break
                
                if not target_node:
                    continue
                
                plan.append({
                    'cluster_id': int(cluster_id),
                    'source_node': source_node,
                    'target_node': target_node,
                    'estimated_vectors': node_loads.get(source_node, 0) // max(1, len(routing_table.get_clusters_for_node(source_node))),
                    'query_count': query_count,
                    'reason': 'forced_hot_cluster_replication'
                })
            
            return plan
        
        # ═══════════════════════════════════════════════════════════════
        # Replication normale (solo se overloaded)
        # ═══════════════════════════════════════════════════════════════
        
        if not overloaded or not underloaded:
            return []
        
        print(f"\n  📊 Replication Analysis:")
        print(f"     Threshold: {threshold:,.0f} vectors")
        print(f"     Overloaded nodes: {len(overloaded)}")
        print(f"     Underloaded nodes: {len(underloaded)}")
        
        for source_node, load in sorted(overloaded.items(), key=lambda x: -x[1]):
            assigned_clusters = routing_table.get_clusters_for_node(source_node)
            
            hot_clusters = sorted(
                [(c, cluster_loads.get(str(c), 0)) for c in assigned_clusters],
                key=lambda x: -x[1]
            )
            
            for cluster_id, query_count in hot_clusters[:2]:
                existing_nodes = routing_table.get_all_nodes_for_cluster(str(cluster_id))
                
                target_node = None
                for candidate in sorted(underloaded.keys(), key=lambda n: underloaded[n]):
                    if candidate not in existing_nodes:
                        target_node = candidate
                        break
                
                if not target_node:
                    continue
                
                plan.append({
                    'cluster_id': cluster_id,
                    'source_node': source_node,
                    'target_node': target_node,
                    'estimated_vectors': load // len(assigned_clusters),
                    'query_count': query_count,
                    'reason': 'hot_cluster_replication'
                })
                
                # Aggiorna carico stimato
                underloaded[target_node] += load // len(assigned_clusters)
        
        return plan
    
    def execute_replication(self,
                           clients: Dict[str, QdrantClient],
                           plan: List[Dict],
                           collection_name: str,
                           routing_table) -> int:
        """Esegue piano di replicazione."""
        if not plan:
            return 0
        
        print(f"\n  🔄 Executing replication plan ({len(plan)} operations)...")
        
        total_replicated = 0
        
        for i, op in enumerate(plan, 1):
            source_client = clients[op['source_node']]
            target_client = clients[op['target_node']]
            
            print(f"\n     [{i}/{len(plan)}] Replicating cluster {op['cluster_id']}")
            print(f"           {op['source_node']} → {op['target_node']}")
            print(f"           Query count: {op['query_count']}, Reason: {op['reason']}")
            
            try:
                all_records = []
                offset = None
                batch_size = 100
                scroll_attempts = 0
                max_scroll_attempts = 1000
                
                print(f"           Fetching vectors (batch size: {batch_size})...", end='', flush=True)
                
                while scroll_attempts < max_scroll_attempts:
                    try:
                        records, next_offset = source_client.scroll(
                            collection_name=collection_name,
                            scroll_filter=Filter(must=[
                                FieldCondition(key="cluster_id", match=MatchValue(value=str(op['cluster_id'])))
                            ]),
                            limit=batch_size,
                            offset=offset,
                            with_payload=True,
                            with_vectors=True
                        )
                        
                        if not records:
                            break
                        
                        all_records.extend(records)
                        offset = next_offset
                        scroll_attempts += 1
                        
                        if scroll_attempts % 10 == 0:
                            print(f"\r           Fetching vectors: {len(all_records):,} collected...", end='', flush=True)
                        
                        if offset is None:
                            break
                    
                    except Exception as scroll_error:
                        error_msg = str(scroll_error)
                        
                        if "400" in error_msg or "Payload" in error_msg or "Bad Request" in error_msg:
                            print(f"\n           ⚠️  Payload overflow, reducing batch size to {batch_size // 2}")
                            batch_size = max(10, batch_size // 2)
                            continue
                        else:
                            raise
                
                print(f"\r           ✓ Fetched {len(all_records):,} vectors                ")
                
                if not all_records:
                    print(f"           ⚠️  No vectors found, skipping")
                    continue
                
                print(f"           Preparing replicas...", end='', flush=True)
                
                replica_points = []
                for record in all_records:
                    new_payload = record.payload.copy()
                    new_payload['is_replica'] = True
                    new_payload['primary_node'] = op['source_node']
                    new_payload['replica_id'] = f"rep-{op['cluster_id']}-{op['target_node']}"
                    new_payload['replica_reason'] = op['reason']
                    new_payload['replicated_at'] = time.time()
                    
                    replica_points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=record.vector,
                        payload=new_payload
                    ))
                
                print(f"\r           ✓ Prepared {len(replica_points):,} replicas           ")
                
                print(f"           Inserting replicas...", end='', flush=True)
                
                insert_batch_size = 500
                total_inserted = 0
                
                for batch_start in range(0, len(replica_points), insert_batch_size):
                    batch_end = min(batch_start + insert_batch_size, len(replica_points))
                    batch = replica_points[batch_start:batch_end]
                    
                    target_client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True
                    )
                    
                    total_inserted += len(batch)
                    
                    progress_pct = (total_inserted / len(replica_points)) * 100
                    print(f"\r           Inserting replicas: {total_inserted:,}/{len(replica_points):,} ({progress_pct:.0f}%)...", end='', flush=True)
                
                print(f"\r           ✓ Inserted {total_inserted:,} replicas                    ")
                
                routing_table.add_replica(str(op['cluster_id']), op['target_node'])
                
                total_replicated += len(replica_points)
                
                print(f"           ✅ {len(replica_points):,} vectors replicated successfully")
                
            except Exception as e:
                print(f"\n           ⚠️  Error: {str(e)[:200]}")
                import traceback
                print(f"           Stack trace: {traceback.format_exc()[:500]}")
                continue
        
        print(f"\n  ✅ Replication complete: {total_replicated:,} total vectors replicated")
        
        return total_replicated


class LoadAwareReplicaSelector:
    """Seleziona replica ottimale basandosi su carico real-time."""
    
    def __init__(self):
        self.active_queries = defaultdict(int)
    
    def select_best_replica(self,
                           cluster_id: str,
                           routing_table,
                           node_loads: Dict[str, int] = None) -> str:
        """Seleziona miglior replica per query."""
        candidates = routing_table.get_all_nodes_for_cluster(cluster_id)
        
        if not candidates:
            raise ValueError(f"Cluster {cluster_id} non ha nodi assegnati")
        
        if len(candidates) == 1:
            return candidates[0]
        
        loads = node_loads if node_loads else self.active_queries
        
        best_node = min(candidates, key=lambda n: loads.get(n, 0))
        
        return best_node
    
    def record_query_start(self, node_name: str):
        """Registra inizio query su nodo."""
        self.active_queries[node_name] += 1
    
    def record_query_end(self, node_name: str):
        """Registra fine query su nodo."""
        self.active_queries[node_name] = max(0, self.active_queries[node_name] - 1)
