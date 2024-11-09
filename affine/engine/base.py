from abc import ABC, abstractmethod
from typing import Type

from affine.collection import Collection, FilterSet, Similarity
from affine.query import QueryObject


class Engine(ABC):

    _RETURNS_NORMALIZED_FOR_COSINE = False

    @abstractmethod
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
        """
        Parameters
        ----------
        collection_class
            the collection class to query
        with_vectors
            wether or not the returned objects should have their vector attributes populated
            (or otherwise be set to `None`)

        Returns
        -------
        QueryObject
            the resulting QueryObject
        """
        return QueryObject(self, collection_class, with_vectors=with_vectors)

    @abstractmethod
    def insert(self, record: Collection) -> int | str:
        """Insert a record

        Parameters
        ----------
        record
            the record to insert

        Returns
        -------
        int | str
            the resulting id of the inserted record
        """
        pass

    @abstractmethod
    def _delete_by_id(self, collection: Type[Collection], id: str) -> None:
        pass

    def delete(
        self,
        *,
        record: Collection | None = None,
        collection: Type[Collection] | None = None,
        id: str | None = None,
    ) -> None:
        """Delete a record from the database. The record can either be specified
        by its `Collection` object or by its id.

        Parameters
        ----------
        record
            the record to delete
        collection
            the collection the record belongs to (needed if and and only deleting a record by its id)
        id
            the id of the record
        """
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
        self, collection: type, ids: list[int | str]
    ) -> list[Collection]:
        """Get elements by ids

        Parameters
        ----------
        ids
            list of ids

        Returns
        -------
        list[collection]
            the resulting collection objects
        """
        pass

    @abstractmethod
    def register_collection(self, collection_class: Type[Collection]) -> None:
        """Register a collection to the database

        Parameters
        ----------
        collection_class
            the class of the collection to register. This class must inherit from `Collection`.
        """
        pass

    def get_element_by_id(
        self, collection: type, id_: int | str
    ) -> Collection:
        """Get an element by its id

        Parameters
        ----------
        collection
            the collection class the record belongs to
        id_
            the id of the record

        Returns
        -------
        collection
            the corresponding collection object for the record.

        Raises
        ------
        ValueError
            if no record is found with the specified id.
        """
        ret = self.get_elements_by_ids(collection, [id_])
        if len(ret) == 0:
            raise ValueError(f"No record found with id {id_}")
        if len(ret) > 1:
            raise ValueError(f"Multiple records found with id {id_}")
        return ret[0]
