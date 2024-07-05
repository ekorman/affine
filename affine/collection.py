from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, get_args

import numpy as np
from pydantic import BaseModel, model_validator

N = TypeVar("N", bound=int)

Operation = Literal["eq", "lte", "gte"]


class Vector(Generic[N]):
    def __init__(self, array: np.ndarray | list):
        if isinstance(array, list):
            array = np.array(array)
        self.array = array


@dataclass
class TopK:
    vector: np.ndarray | list | Vector
    k: int


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


class Filter(BaseModel):
    field: str
    operation: Operation
    value: Any


class FilterSet(BaseModel):
    filters: list[Filter]
    collection: str

    def __len__(self) -> int:
        return len(self.filters)


class Collection(BaseModel):
    @model_validator(mode="before")
    def validate_arrays(cls, values):
        # check that any vec types have the specified length
        return values

    @classmethod
    def get_filter_from_kwarg(cls, k: str, v: Any) -> Filter:
        s = k.split("__")
        if len(s) == 1:
            s.append("eq")
        if s[1] not in get_args(Operation):
            raise ValueError(
                f"Operation {s[1]} not supported. Supported operations are {get_args(Operation)}"
            )
        field, op = s
        if field not in cls.model_fields:
            raise ValueError(f"Field {field} not in {cls.__name__}")
        return Filter(field=field, operation=op, value=v)

    @classmethod
    def objects(cls, **kwargs) -> FilterSet:
        filters = [cls.get_filter_from_kwarg(k, v) for k, v in kwargs.items()]
        return FilterSet(filters=filters, collection=cls.__name__)


# Example
# class Person(Collection):
#     age: int
#     face_embedding: np.ndarray


# Person.query(age__gte=18, face_embedding=TopK(vector=np.array([1, 2, 3]), k=3))

# use a global connection (like mongoengine and others?)
