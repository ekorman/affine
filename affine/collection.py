from dataclasses import dataclass, fields
from typing import Any, Generic, Literal, TypeVar, get_origin

import numpy as np
from typing_extensions import dataclass_transform

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
    collection: str
    field: str
    operation: Operation
    value: Any

    def __and__(self, other: "Filter") -> "FilterSet":
        if self.collection != other.collection:
            raise ValueError("Filters must be from the same collection")
        return FilterSet(filters=[self, other], collection=self.collection)


@dataclass
class TopKFilter(Filter):
    # just used for typing
    value: TopK


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

    def __eq__(self, value: object) -> Filter:
        if isinstance(value, TopK):
            operation = "topk"
        else:
            operation = "eq"
        return Filter(
            field=self.name,
            operation=operation,
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
        # if cls.__inside_dataclass_creator:
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


def get_topk_filter_and_non_topk_filters(
    filters: list[Filter],
) -> tuple[TopKFilter | None, list[Filter]]:
    topk_filters = []
    non_topk_filters = []
    for f in filters:
        if f.operation == "topk":
            topk_filters.append(f)
        else:
            non_topk_filters.append(f)

    if len(topk_filters) > 1:
        raise ValueError(
            f"Only one topk filter is allowed but got {len(topk_filters)}."
        )
    topk_filter = topk_filters[0] if len(topk_filters) == 1 else None
    return topk_filter, non_topk_filters
