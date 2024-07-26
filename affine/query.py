from typing import TYPE_CHECKING, Type

from affine.collection import Collection, Filter, FilterSet

if TYPE_CHECKING:
    from affine.engine import Engine


class QueryObject:
    def __init__(self, db: "Engine", collection_class: Type[Collection]):
        self.db = db
        self.collection_class = collection_class

    def filter(self, filter_set: FilterSet | Filter) -> list[Collection]:
        if isinstance(filter_set, Filter):
            filter_set = FilterSet(
                filters=[filter_set], collection=filter_set.collection
            )
        return self.db._query(filter_set)

    def all(self) -> list[Collection]:
        return self.db._query(
            FilterSet(filters=[], collection=self.collection_class.__name__)
        )

    def get_by_id(self, id_) -> Collection:
        return self.db.get_element_by_id(self.collection_class, id_)
