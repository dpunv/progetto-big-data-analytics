from typing import List, Tuple, Set, FrozenSet

Vector = List[float]
VectorId = int
VectorPayload = str
VectorWithId = Tuple[VectorId, Vector]
VectorWithPayload = Tuple[Vector, VectorPayload]
VectorComplete = Tuple[Vector, VectorId, VectorPayload, int]
ListOfVectors = List[Vector]
ListOfVectorsWithId = List[VectorWithId]
ListOfVectorsWithPayload = List[VectorWithPayload]
ListOfVectorsComplete = List[VectorComplete]

# Partition tolerance types
VectorVersion = Tuple[float, int]  # (timestamp, originating_node_id)
VectorDestinations = FrozenSet[int]  # Set of intended destination node IDs

# Full vector with version and destinations:
# (vector, id, payload, cluster_id, version, destinations)
VectorFull = Tuple[Vector, VectorId, VectorPayload, int, VectorVersion, VectorDestinations]
ListOfVectorsFull = List[VectorFull]

# Legacy type for backwards compatibility
VectorVersioned = Tuple[Vector, VectorId, VectorPayload, int, VectorVersion]
ListOfVectorsVersioned = List[VectorVersioned]