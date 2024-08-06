from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Generic, Literal, Type, TypeVar, get_origin

import numpy as np
from typing_extensions import dataclass_transform


class Metric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


N = TypeVar("N", bound=int)
M = TypeVar("M", bound=Metric)

Operation = Literal["eq", "lte", "gte", "topk"]


class Vector(Generic[N, M]):
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

    def normalize(self) -> "Vector":
        """Returns the L2 normalized vector"""
        return Vector(array=self.array / np.linalg.norm(self.array))

    def __repr__(self) -> str:
        return f"<Vector: {self.array}>"


VectorType = Vector | np.ndarray | list


@dataclass
class Filter:
    collection: str
    field: str
    operation: Operation
    value: Any

    def __and__(self, other: "Filter") -> "FilterSet":
        if self.collection != other.collection:
            raise ValueError("Filters must be from the same collection")
        return FilterSet(filters=[self, other], collection=self.collection)


@dataclass
class Similarity:
    collection: str
    field: str
    value: VectorType

    def get_list(self) -> list[float]:
        if isinstance(self.value, Vector):
            return self.value.array.tolist()
        if isinstance(self.value, np.ndarray):
            return self.value.tolist()
        return self.value

    def get_array(self) -> np.ndarray:
        if isinstance(self.value, Vector):
            return self.value.array
        if isinstance(self.value, np.ndarray):
            return self.value
        return np.array(self.value)


@dataclass
class FilterSet:
    filters: list[Filter]
    collection: str

    def __len__(self) -> int:
        return len(self.filters)

    def __and__(self, other: "FilterSet") -> "FilterSet":
        if self.collection != other.collection:
            raise ValueError("Filters must be from the same collection")
        return FilterSet(
            filters=self.filters + other.filters, collection=self.collection
        )


@dataclass
class Attribute:
    collection: str
    name: str

    def __eq__(self, value: object) -> Filter | Similarity:
        if isinstance(value, VectorType):
            return Similarity(
                collection=self.collection,
                field=self.name,
                value=value,
            )
        return Filter(
            field=self.name,
            operation="eq",
            value=value,
            collection=self.collection,
        )

    def __gt__(self, value: object) -> Filter:
        return Filter(
            field=self.name,
            operation="gte",
            value=value,
            collection=self.collection,
        )

    def __ge__(self, value: object) -> Filter:
        return Filter(
            field=self.name,
            operation="gte",
            value=value,
            collection=self.collection,
        )

    def __lt__(self, value: object) -> Filter:
        return Filter(
            field=self.name,
            operation="lte",
            value=value,
            collection=self.collection,
        )

    def __le__(self, value: object) -> Filter:
        return Filter(
            field=self.name,
            operation="lte",
            value=value,
            collection=self.collection,
        )


class MetaCollection(type):
    """This metaclass is used so that subclasses of Collection are automatically decorated with dataclass"""

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        return dataclass(new_class)

    def __getattribute__(cls, name: str) -> Any:
        try:
            if name in super().__getattribute__("__dataclass_fields__"):  # type: ignore
                return Attribute(name=name, collection=cls.__name__)
        # in case `__dataclass_fields__` does not exist yet
        except AttributeError:
            pass
        return super().__getattribute__(name)


@dataclass_transform()
class Collection(metaclass=MetaCollection):
    """Base class for a collection of documents. Subclasses should define fields as class attributes (dataclasses style)."""

    def __post_init__(self):
        for field in fields(self):
            if get_origin(field.type) == Vector:
                n = field.type.__args__[0]
                attr = getattr(self, field.name)
                # when returning a query result the vector may not be present
                if attr is not None and len(getattr(self, field.name)) != n:
                    raise ValueError(
                        f"Expected vector of length {n}, got {len(getattr(self, field.name))}"
                    )
        self.id = None

    @classmethod
    def get_vector_fields(
        cls: Type["Collection"],
    ) -> list[tuple[str, int, Metric]]:
        """Get all the vector fields of a collection

        Returns
        -------
        list[tuple[str, int, Metric]]
            A list of tuples where the first element is the name of the vector field,
            the second element is its dimension, and the third is the metric that should
            be used with it
        """
        return [
            (f.name, f.type.__args__[0], f.type.__args__[1].__forward_arg__)
            for f in fields(cls)
            if get_origin(f.type) == Vector
        ]

    def get_non_vector_dict(self) -> dict[str, Any]:
        """Returns a dictionary of all metadata (i.e. all fields and values that are not vectors)"""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if get_origin(f.type) != Vector
        }
