from typing import List, Tuple

Vector = List[float]
VectorId = str
VectorPayload = str
VectorWithId = Tuple[VectorId, Vector]
VectorWithPayload = Tuple[Vector, VectorPayload]
VectorComplete = Tuple[Vector, VectorId, VectorPayload]
ListOfVectors = List[Vector]
ListOfVectorsWithId = List[VectorWithId]
ListOfVectorsWithPayload = List[VectorWithPayload]
ListOfVectorsComplete = List[VectorComplete]
MetaHNSW = None