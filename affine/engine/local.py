import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Type

import numpy as np

from affine.collection import Collection, Filter, FilterSet, Metric, Similarity
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


class LocalBackend(ABC):
    @abstractmethod
    def create_index(self, data: np.ndarray, metric: Metric) -> None:
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
    def create_index(self, data: np.ndarray, metric: Metric) -> None:
        self.metric = metric
        self._index = data

    def query(self, q: np.ndarray, k: int) -> list[int]:
        if self.metric == Metric.COSINE:
            return np.argsort(
                -np.dot(self._index, q)
                / (
                    np.linalg.norm(
                        self._index, axis=1
                    )  # * np.linalg.norm(q) don't need to divide by this if not returning distances
                )
            )[:k].tolist()
        return np.linalg.norm(self._index - q, axis=1).argsort()[:k].tolist()


class KDTreeBackend(LocalBackend):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_index(self, data: np.ndarray, metric: Metric) -> None:
        try:
            from sklearn.neighbors import KDTree
        except ModuleNotFoundError:
            raise RuntimeError(
                "KDTree backend requires scikit-learn to be installed"
            )
        self._metric = metric
        if metric == Metric.COSINE:
            data = data / np.linalg.norm(data, axis=1).reshape(-1, 1)

        self.tree = KDTree(data, **self.kwargs)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        # q should be shape (N,)
        assert len(q.shape) == 1
        q = q.reshape(1, -1)

        if self._metric == Metric.COSINE:
            q = q / np.linalg.norm(q)
        return self.tree.query(q, k)[1][0].tolist()


class PyNNDescentBackend(LocalBackend):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_index(self, data: np.ndarray, metric: Metric) -> None:
        try:
            from pynndescent import NNDescent
        except ModuleNotFoundError:
            raise RuntimeError(
                "PyNNDescentBackend backend requires pynndescent to be installed"
            )
        self.index = NNDescent(data, metric=metric.value, **self.kwargs)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        idxs, _ = self.index.query(q, k)
        return idxs[0].tolist()


class AnnoyBackend(LocalBackend):
    def __init__(self, n_trees: int, n_jobs: int = -1):
        self.n_trees = n_trees
        self.n_jobs = n_jobs

    def create_index(self, data: np.ndarray, metric: Metric) -> None:
        try:
            from annoy import AnnoyIndex
        except ModuleNotFoundError:
            raise RuntimeError(
                "AnnoyBackend backend requires annoy to be installed"
            )

        annoy_metric = "angular" if metric == Metric.COSINE else "euclidean"
        self.index = AnnoyIndex(data.shape[1], metric=annoy_metric)
        for i, v in enumerate(data):
            self.index.add_item(i, v)
        self.index.build(self.n_trees, self.n_jobs)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        return self.index.get_nns_by_vector(q, k)


class FAISSBackend(LocalBackend):
    def __init__(self, index_factory_str: str):
        """
        Parameters
        ----------
        index_factory_str : str
            A string that specifies the index type to be created.
            See https://github.com/facebookresearch/faiss/wiki/The-index-factory for details.
        """
        self.index_factory_str = index_factory_str

    def create_index(self, data: np.ndarray, metric: Metric) -> None:
        try:
            import faiss
        except ModuleNotFoundError:
            raise RuntimeError(
                "FAISSBackend backend requires FAISS to be installed. See "
                "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md for installation instructions."
            )
        if metric == Metric.COSINE:
            data = data / np.linalg.norm(data, axis=1).reshape(-1, 1)
        self.index = faiss.index_factory(data.shape[1], self.index_factory_str)
        self.index.add(data)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        _, idxs = self.index.search(q.reshape(1, -1) / np.linalg.norm(q), k)
        assert idxs.shape[0] == 1
        return idxs[0].tolist()


class LocalEngine(Engine):
    def __init__(
        self, backend: LocalBackend | None = None
    ) -> None:  # maybe add option to the init for ANN algo
        self.records: dict[str, list[Collection]] = defaultdict(list)
        self.build_collection_id_counter()
        self.backend = backend or NumPyBackend()
        # maps collection class name and then field name to metric
        self.collection_name_to_field_to_metric: dict[
            str, dict[str, Metric]
        ] = {}

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
        metric = self.collection_name_to_field_to_metric[
            filter_set.collection
        ][similarity.field]
        self.backend.create_index(data, metric)
        neighbors = self.backend.query(q, limit)
        return [records[i] for i in neighbors]

    def insert(self, record: Collection) -> int:
        record.id = self.collection_id_counter[record.__class__.__name__] + 1
        self.records[record.__class__.__name__].append(record)
        self.collection_id_counter[record.__class__.__name__] = record.id

        return record.id

    def register_collection(self, collection_class: Type[Collection]) -> None:
        self.collection_name_to_field_to_metric[collection_class.__name__] = {
            field_name: metric
            for field_name, _, metric in collection_class.get_vector_fields()
        }

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
