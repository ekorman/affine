import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Type

import numpy as np

from affine.collection import Collection, Filter, FilterSet, Similarity
from affine.engine import Engine
from affine.query import QueryObject


def apply_filter_to_record(filter_: Filter, record: Collection) -> bool:
    field = getattr(record, filter_.field)
    if filter_.operation == "eq":
        return field == filter_.value
    elif filter_.operation == "gte":
        return field >= filter_.value
    elif filter_.operation == "lte":
        return field <= filter_.value
    elif filter_.operation == "gt":
        return field > filter_.value
    elif filter_.operation == "lt":
        return field < filter_.value
    else:
        raise ValueError(f"Operation {filter_.operation} not supported")


def apply_filters_to_records(
    filters: list[Filter], records: list[Collection]
) -> list[Collection]:
    ret = []
    for record in records:
        record_match = True
        for f in filters:
            if not apply_filter_to_record(f, record):
                record_match = False
                break
        if record_match:
            ret.append(record)
    return ret


def build_data_matrix(
    field_name: str, records: list[Collection]
) -> np.ndarray:
    return np.stack([getattr(r, field_name).array for r in records])


# def filter_by_similarity(
#     similarity: Similarity, limit: int, records: list[Collection]
# ) -> list[Collection]:
#     vectors = build_data_matrix(similarity.field, records)
#     query_vector = similarity.get_array()
#     distances = np.linalg.norm(vectors - query_vector, axis=1)
#     topk_indices = distances.argsort()[:limit]
#     return [records[i] for i in topk_indices]


class LocalBackend(ABC):
    @abstractmethod
    def create_index(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def query(self, q: np.ndarray, k: int) -> list[int]:
        pass

    # @abstractmethod
    # def save(self, fp):
    #     pass

    # @abstractmethod
    # def load(self, fp):
    #     pass


class NumPyBackend(LocalBackend):
    def create_index(self, data: np.ndarray) -> None:
        self._index = data

    def query(self, q: np.ndarray, k: int) -> list[int]:
        return np.linalg.norm(self._index - q, axis=1).argsort()[:k].tolist()


class KDTreeBackend(LocalBackend):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_index(self, data: np.ndarray) -> None:
        try:
            from sklearn.neighbors import KDTree
        except ModuleNotFoundError:
            raise RuntimeError(
                "KDTree backend requires scikit-learn to be installed"
            )
        self.tree = KDTree(data, **self.kwargs)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        # q should be shape (N,)
        assert len(q.shape) == 1
        q = q.reshape(1, -1)
        return self.tree.query(q, k)[1][0].tolist()


class PyNNDescentBackend(LocalBackend):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_index(self, data: np.ndarray) -> None:
        try:
            from pynndescent import NNDescent
        except ModuleNotFoundError:
            raise RuntimeError(
                "PyNNDescentBackend backend requires pynndescent to be installed"
            )
        self.index = NNDescent(data, **self.kwargs)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        idxs, _ = self.index.query(q, k)
        return idxs[0].tolist()


class AnnoyBackend(LocalBackend):
    pass


class FAISSBackend(LocalBackend):
    pass


class LocalEngine(Engine):
    def __init__(
        self, backend: LocalBackend | None = None
    ) -> None:  # maybe add option to the init for ANN algo
        self.records: dict[str, list[Collection]] = defaultdict(list)
        self.build_collection_id_counter()
        self.backend = backend or NumPyBackend()

    def build_collection_id_counter(self):
        # maybe pickle this too on save?
        self.collection_id_counter: dict[str, int] = defaultdict(int)
        for k, recs in self.records.items():
            if len(recs) > 0:
                self.collection_id_counter[k] = max([r.id for r in recs])

    def load(self, fp: str | Path | BinaryIO) -> None:
        self.records: dict[str, list[Collection]] = defaultdict(list)
        if isinstance(fp, (str, Path)):
            with open(fp, "rb") as f:
                self.records = pickle.load(f)
        else:
            self.records = pickle.load(fp)
        self.build_collection_id_counter()

    def save(self, fp: str | Path | BinaryIO = None) -> None:
        fp = fp or self.fp
        if isinstance(fp, (str, Path)):
            with open(fp, "wb") as f:
                pickle.dump(self.records, f)
        else:
            fp.seek(0)
            pickle.dump(self.records, fp)  # don't close, handle it outside

    def _query(
        self,
        filter_set: FilterSet,
        with_vectors: bool = True,
        similarity: Similarity | None = None,
        limit: int | None = None,
    ) -> list[Collection]:
        if not with_vectors:
            warnings.warn("with_vectors=False has no effect in LocalEngine")
        records = self.records[filter_set.collection]
        records = apply_filters_to_records(filter_set.filters, records)
        if similarity is None:
            if limit is None:
                return records
            return records[:limit]

        data = build_data_matrix(similarity.field, records)
        q = similarity.get_array()
        self.backend.create_index(data)
        neighbors = self.backend.query(q, limit)
        return [records[i] for i in neighbors]

    def insert(self, record: Collection) -> int:
        record.id = self.collection_id_counter[record.__class__.__name__] + 1
        self.records[record.__class__.__name__].append(record)
        self.collection_id_counter[record.__class__.__name__] = record.id

        return record.id

    def register_collection(self, collection_class: Type[Collection]) -> None:
        pass

    def _delete_by_id(self, collection: Type[Collection], id: str) -> None:
        collection_name = collection.__name__
        for r in self.records[collection_name]:
            if r.id == id:
                self.records[collection_name].remove(r)
                return
        raise ValueError(
            f"Record with id {id} not found in collection {collection_name}"
        )

    def get_elements_by_ids(
        self, collection: type, ids: list[int]
    ) -> list[Collection]:
        return [r for r in self.records[collection.__name__] if r.id in ids]

    def query(
        self, collection_class: Type[Collection], with_vectors: bool = True
    ) -> QueryObject:
        return super().query(collection_class, with_vectors=with_vectors)
