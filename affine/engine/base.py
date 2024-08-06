from abc import ABC, abstractmethod
from typing import Type

from affine.collection import Collection, FilterSet, Similarity
from affine.query import QueryObject


class Engine(ABC):

    _RETURNS_NORMALIZED_FOR_COSINE = False

    @abstractmethod
    # TODO: add `return_vectors` as an argument here?
    def _query(
        self,
        filter_set: FilterSet,
        with_vectors: bool = False,
        similarity: Similarity | None = None,
        limit: int | None = None,
    ) -> list[Collection]:
        pass

    def query(
        self, collection_class: Type[Collection], with_vectors: bool = False
    ) -> QueryObject:
        return QueryObject(self, collection_class, with_vectors=with_vectors)

    @abstractmethod
    def insert(self, record: Collection) -> int | str:
        pass

    @abstractmethod
    def _delete_by_id(self, collection: Type[Collection], id: str) -> None:
        pass

    def delete(
        self,
        *,
        record: Collection | str | None = None,
        collection: Type[Collection] | None = None,
        id: str | None = None,
    ) -> None:
        if bool(record is None) == bool(collection is None and id is None):
            raise ValueError(
                "Either record or collection and id must be provided"
            )
        if record is not None:
            if collection is not None or id is not None:
                raise ValueError(
                    "Either record or collection and id must be provided"
                )
            self._delete_by_id(record.__class__, record.id)
        else:
            if collection is None or id is None:
                raise ValueError(
                    "Either record or collection and id must be provided"
                )
            self._delete_by_id(collection, id)

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
