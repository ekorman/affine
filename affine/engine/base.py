from abc import ABC, abstractmethod

from affine.collection import Collection, FilterSet


class Engine(ABC):
    @abstractmethod
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

    def get_element_by_id(self, collection: type, id_: int) -> Collection:
        ret = self.get_elements_by_ids(collection, [id_])
        if len(ret) == 0:
            raise ValueError(f"No record found with id {id_}")
        if len(ret) > 1:
            raise ValueError(f"Multiple records found with id {id_}")
        return ret[0]
