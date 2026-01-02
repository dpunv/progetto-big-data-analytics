from typing import List, Tuple

Vector = List[float]
VectorId = int
VectorPayload = str
VectorCluster = int
VectorWithId = Tuple[VectorId, Vector]
VectorWithPayload = Tuple[Vector, VectorPayload]
VectorComplete = Tuple[Vector, VectorId, VectorPayload, VectorCluster]
ListOfVectors = List[Vector]
ListOfVectorsWithId = List[VectorWithId]
ListOfVectorsWithPayload = List[VectorWithPayload]
ListOfVectorsComplete = List[VectorComplete]
MetaHNSW = None