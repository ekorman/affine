from typing import get_args

import numpy as np
from affine.engine import Engine, Operation
from pydantic import BaseModel


def get_attribute_name_and_operation_from_key(
    k: str,
) -> tuple[str, Operation]:
    s = k.split("__")
    if len(s) == 1:
        s.append("eq")
    if s[1] not in get_args(Operation):
        raise ValueError(
            f"Operation {s[1]} not supported. Supported operations are {get_args(Operation)}"
        )
    return tuple(s)


class Collection(
    BaseModel
):  # think of Collection as a namespace? as a table? engine should be the db
    engine: Engine

    def query(self, **kwargs):
        pass


class TopK(BaseModel):
    vector: np.ndarray
    k: int


# Example
class Person(Collection):
    age: int
    face_embedding: np.ndarray


Person.query(age__gte=18, face_embedding=TopK(vector=np.array([1, 2, 3]), k=3))

# use a global connection (like mongoengine and others?)
