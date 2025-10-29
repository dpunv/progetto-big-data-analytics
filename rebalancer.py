# rebalancer.py
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

from routing import RoutingTable
from config import COLLECTION_NAME, SPLIT_FACTOR

class Rebalancer:
    def __init__(self, qdrant_clients: dict[str, QdrantClient], routing_table: RoutingTable):
        self.clients = qdrant_clients
        self.routing_table = routing_table

    def monitor_clusters(self) -> dict[str, int]:
        """
        Scansiona tutti i nodi per contare i punti per ogni cluster_id.
        NOTA: In produzione, questo è inefficiente. Usare contatori esterni (es. Redis)
        è la strategia migliore. Qui lo facciamo per semplicità dimostrativa.
        """
        print("\nMonitoring cluster density...")
        cluster_counts = {}
        all_known_clusters = list(self.routing_table._routing_map.keys())

        for cluster_id in tqdm(all_known_clusters, desc="Monitoring Clusters"):
            nodes_to_check = self.routing_table.get_nodes(cluster_id)
            total_count = 0
            for node in nodes_to_check:
                try:
                    count_res = self.clients[node].count(
                        collection_name=COLLECTION_NAME,
                        count_filter=Filter(must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]),
                        exact=True
                    )
                    total_count += count_res.count
                except Exception as e:
                    print(f"Could not count cluster {cluster_id} on node {node}: {e}")
            cluster_counts[cluster_id] = total_count
        
        print("Cluster counts:", cluster_counts)
        return cluster_counts

    def fetch_all_points_for_cluster(self, cluster_id: str, source_node: str):
        """
        Estrae tutti i punti per un dato cluster_id dal nodo di origine in modo robusto,
        prevenendo i loop infiniti.
        """
        client = self.clients[source_node]
        all_points = []
        offset = None
        seen_ids = set() # ✅ SOLUZIONE 1: Traccia gli ID per evitare loop
        max_loops = 10000 # Interruttore di sicurezza
        
        loop_count = 0
        while loop_count < max_loops:
            try:
                points, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]),
                    limit=256,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset
                )
            except Exception as e:
                print(f"Error during scroll on node {source_node} for cluster {cluster_id}: {e}")
                break

            if not points:
                break # Condizione di uscita normale

            new_points_found = False
            for point in points:
                if point.id not in seen_ids:
                    all_points.append(point)
                    seen_ids.add(point.id)
                    new_points_found = True
            
            if not new_points_found and next_offset is not None:
                print(f"WARN: Potential infinite loop detected for cluster {cluster_id}. Breaking scroll.")
                break
                
            offset = next_offset
            if offset is None:
                break
                
            loop_count += 1
            
        if loop_count >= max_loops:
            print(f"ERROR: Max loops reached for cluster {cluster_id}. Aborting scroll.")

        return all_points

    def split_and_migrate_cluster(self, cluster_id_to_split: int):
        """Orchestra lo split e la migrazione di un cluster "hot"."""
        print(f"\n🔥🔥🔥 Hotspot detected on cluster {cluster_id_to_split}! Starting split and migration. 🔥🔥🔥")
        
        source_node_name = self.routing_table.get_nodes(str(cluster_id_to_split))[0]
        available_nodes = list(self.clients.keys())
        
        print(f"Fetching all points from cluster {cluster_id_to_split} on node {source_node_name}...")
        points_to_migrate = self.fetch_all_points_for_cluster(str(cluster_id_to_split), source_node_name)
        
        if not points_to_migrate:
            print("Cluster is empty, aborting split.")
            return
            
        vectors = np.array([p.vector for p in points_to_migrate])
        print(f"Fetched {len(vectors)} points to migrate.")

        print(f"Re-clustering points into {SPLIT_FACTOR} sub-clusters...")
        kmeans = KMeans(n_clusters=SPLIT_FACTOR, random_state=42, n_init='auto').fit(vectors)

        new_sub_cluster_ids = [f"{cluster_id_to_split}.{i}" for i in range(SPLIT_FACTOR)]
        node_assignments = {sub_id: available_nodes[i % len(available_nodes)] for i, sub_id in enumerate(new_sub_cluster_ids)}
        
        buffers = {node_name: [] for node_name in self.clients.keys()}
        
        print("Re-assigning points to new sub-clusters and preparing for migration...")
        for i, record in enumerate(tqdm(points_to_migrate, desc="Preparing Migration")):
            # 'record' è un oggetto di tipo Record
            sub_cluster_local_id = kmeans.labels_[i]
            new_cluster_id = f"{cluster_id_to_split}.{sub_cluster_local_id}"
            target_node = node_assignments[new_cluster_id]
            
            # Prepara il nuovo payload
            payload = record.payload or {}
            payload["cluster_id"] = new_cluster_id
            
            # ✅ SOLUZIONE 2: Crea un PointStruct dal Record
            new_point_struct = models.PointStruct(
                id=record.id,
                vector=record.vector,
                payload=payload
            )
            
            buffers[target_node].append(new_point_struct)

        print("Migrating points to new nodes via bulk upsert...")
        for node_name, points_batch in buffers.items():
            if points_batch:
                client = self.clients[node_name]
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points_batch,
                        wait=True
                    )
                    print(f"Upserted {len(points_batch)} points to node {node_name}.")
                except Exception as e:
                    print(f"ERROR upserting to {node_name}: {e}")


        self.routing_table.split_cluster_mapping(int(cluster_id_to_split), new_sub_cluster_ids, node_assignments)

        print(f"Cleaning up original points from cluster {cluster_id_to_split} on node {source_node_name}...")
        source_client = self.clients[source_node_name]
        try:
            source_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="cluster_id", match=models.MatchValue(value=str(cluster_id_to_split)))]
                    )
                ),
                wait=True,
            )
        except Exception as e:
            print(f"ERROR deleting old points from {source_node_name}: {e}")
        
        print("Migration complete! ✨")