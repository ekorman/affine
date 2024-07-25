from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from affine.collection import Collection, FilterSet


@dataclass
class QueryResponse:
    document: Collection
    distance: float | None = None


class Engine(ABC):
    @abstractmethod
    # TODO: add `return_vectors` as an argument here
    def query(self, filter_set: FilterSet) -> list[Collection]:
        pass

    @abstractmethod
    def insert(self, record: Collection) -> int:
        pass

    @abstractmethod
    def delete(self, collection: type, id_: int) -> None:
        pass

    @abstractmethod
    def get_elements_by_ids(
        self, collection: type, ids: list[int]
    ) -> list[Collection]:
        pass

    @abstractmethod
    def register_collection(self, collection_class: Type[Collection]) -> None:
        pass

    def get_element_by_id(self, collection: type, id_: int) -> Collection:
        ret = self.get_elements_by_ids(collection, [id_])
        if len(ret) == 0:
            raise ValueError(f"No record found with id {id_}")
        if len(ret) > 1:
            raise ValueError(f"Multiple records found with id {id_}")
        return ret[0]
