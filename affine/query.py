from typing import TYPE_CHECKING, Type

from affine.collection import Collection, Filter, FilterSet, Similarity

if TYPE_CHECKING:
    from affine.engine import Engine


class QueryObject:
    def __init__(
        self,
        db: "Engine",
        collection_class: Type[Collection],
        with_vectors: bool,
    ):
        self.db = db
        self.collection_class = collection_class
        self.with_vectors = with_vectors
        self._filter_set = FilterSet(
            filters=[], collection=collection_class.__name__
        )
        self._similarity = None

    def filter(self, filter_set: FilterSet | Filter) -> "QueryObject":
        if isinstance(filter_set, Filter):
            filter_set = FilterSet(
                filters=[filter_set], collection=filter_set.collection
            )

        self._filter_set = self._filter_set & filter_set
        return self

    def all(self) -> list[Collection]:
        return self.db._query(self._filter_set, with_vectors=self.with_vectors)

    def limit(self, n: int) -> list[Collection]:
        return self.db._query(
            self._filter_set,
            with_vectors=self.with_vectors,
            limit=n,
            similarity=self._similarity,
        )

    def similarity(self, similarity: Similarity) -> "QueryObject":
        self._similarity = similarity
        return self

    def get_by_id(self, id_) -> Collection:
        return self.db.get_element_by_id(self.collection_class, id_)
