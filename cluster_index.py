import base64
import logging
import pickle
from typing import List

import hnswlib
import numpy as np

from compound_types import ListOfVectorsWithId, Vector

logger = logging.getLogger(__name__)


class ClusterIndex:
    """An HNSW-based index for fast retrieval of nearest clusters.
    
    This index is used to route queries to the appropriate clusters by storing 
    cluster centroids and allowing approximate nearest neighbor search.
    """

    def __init__(
        self,
        dimension: int,
        max_clusters: int = 500,
        ef_construction: int = 200,
        M: int = 16,
    ):
        self.dimension = dimension
        self.max_clusters = max_clusters
        self.ef_construction = ef_construction
        self.M = M
        self.hnsw_index = None

    def build(self, clusters: ListOfVectorsWithId):
        """
        Builds the HNSW index from a list of (cluster_id, centroid_vector) tuples.
        """
        if not clusters:
            logger.warning("No clusters to build index from.")
            return

        logger.info("Building HNSW index for clusters...")
        self.hnsw_index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.hnsw_index.init_index(
            max_elements=max(len(clusters), self.max_clusters),
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.hnsw_index.set_ef(50)


        indices = []
        centroids = []

        for cluster in clusters:
            try:
                cid = int(cluster[0])
                indices.append(cid)
                centroids.append(cluster[1])
            except ValueError:
                logger.error(
                    f"Cluster ID {cluster[0]} is not an integer. HNSW requires int labels."
                )

        if centroids:
            self.hnsw_index.add_items(
                np.array(centroids, dtype=np.float32), np.array(indices)
            )
            logger.info(f"HNSW index built with {len(indices)} items.")
        else:
            logger.warning("No valid items to add to HNSW index.")

    def find_nearest_clusters(self, query_vector: Vector, k: int = 1) -> List[int]:
        """
        Finds the nearest k clusters for a single query vector.
        Returns a list of cluster IDs.
        """
        if self.hnsw_index is None:
            return []

        current_count = self.hnsw_index.element_count
        if k > current_count:
            k = current_count

        if k == 0:
            return []

        if self.hnsw_index.ef < k:
            self.hnsw_index.set_ef(k)

        query = np.array(query_vector, dtype=np.float32)
        cluster_ids, _ = self.hnsw_index.knn_query(query, k=k)

        return [int(id) for id in cluster_ids[0]]

    def search_batch(self, query_vectors: np.ndarray, k: int = 1) -> List[List[int]]:
        """
        Search for a batch of vectors.
        query_vectors: numpy array of shape (N, D)
        Returns: List of lists of cluster IDs.
        """
        if self.hnsw_index is None:
            return []

        current_count = self.hnsw_index.element_count
        if k > current_count:
            k = current_count

        if k == 0:
            return [[] for _ in range(len(query_vectors))]

        if self.hnsw_index.ef < k:
            self.hnsw_index.set_ef(k)

        labels, _ = self.hnsw_index.knn_query(query_vectors, k=k)
        return labels.astype(int).tolist()

    def to_serializable(self):
        """
        Returns a dict enabling reconstruction/distribution of the index.
        """
        index_base64 = None
        if self.hnsw_index:
            index_binary = pickle.dumps(self.hnsw_index)
            index_base64 = base64.b64encode(index_binary).decode("utf-8")
        return {
            "dimension": self.dimension,
            "max_clusters": self.max_clusters,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "index_data": index_base64,
        }

    @classmethod
    def from_serializable(cls, data: dict):
        new_obj = cls(
            dimension=data["dimension"],
            max_clusters=data["max_clusters"],
            ef_construction=data["ef_construction"],
            M=data["M"],
        )
        index_base64 = data.get("index_data")
        if index_base64:
            try:
                index_binary = base64.b64decode(index_base64)
                new_obj.hnsw_index = pickle.loads(index_binary)
            except Exception as e:
                logger.error(f"Failed to load HNSW index from pickle: {e}")
        return new_obj
