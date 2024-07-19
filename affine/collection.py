from dataclasses import dataclass, fields
from typing import Any, Generic, Literal, TypeVar, get_args, get_origin

import numpy as np

N = TypeVar("N", bound=int)

Operation = Literal["eq", "lte", "gte", "topk"]


class Vector(Generic[N]):
    def __init__(self, array: np.ndarray | list):
        if isinstance(array, list):
            array = np.array(array)
        self.array = array

    def __len__(self) -> int:
        return len(self.array)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Vector):
            return False
        return np.allclose(self.array, other.array)


@dataclass
class TopK:
    vector: np.ndarray | list | Vector
    k: int


@dataclass
class Filter:
    field: str
    operation: Operation
    value: Any


@dataclass
class FilterSet:
    filters: list[Filter]
    collection: str

    def __len__(self) -> int:
        return len(self.filters)


class MetaCollection(type):
    """This metaclass is used so that subclasses of Collection are automatically decorated with dataclass"""

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        return dataclass(new_class)


class Collection(metaclass=MetaCollection):
    """Base class for a collection of documents. Subclasses should define fields as class attributes (dataclasses style)."""

    def __post_init__(self):
        for field in fields(self):
            if get_origin(field.type) == Vector:
                n = field.type.__args__[0]
                if len(getattr(self, field.name)) != n:
                    raise ValueError(
                        f"Expected vector of length {n}, got {len(getattr(self, field.name))}"
                    )
        self.id = None

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
        if field not in [f.name for f in fields(cls)] + ["id"]:
            raise ValueError(f"Field {field} not in {cls.__name__}")
        return Filter(field=field, operation=op, value=v)

    @classmethod
    def objects(cls, **kwargs) -> FilterSet:
        filters = [cls.get_filter_from_kwarg(k, v) for k, v in kwargs.items()]
        return FilterSet(filters=filters, collection=cls.__name__)
