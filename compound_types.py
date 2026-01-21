from typing import FrozenSet, List, Tuple

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

VectorVersion = Tuple[float, int]
VectorDestinations = FrozenSet[int]

VectorFull = Tuple[
    Vector, VectorId, VectorPayload, int, VectorVersion, VectorDestinations
]
ListOfVectorsFull = List[VectorFull]

VectorVersioned = Tuple[Vector, VectorId, VectorPayload, int, VectorVersion]
ListOfVectorsVersioned = List[VectorVersioned]
