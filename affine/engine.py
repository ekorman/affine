from abc import ABC, abstractmethod
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path, PosixPath

import numpy as np

from affine.collection import Collection, Filter, FilterSet


def apply_filter_to_record(filter_: Filter, record: Collection) -> bool:
    field = getattr(record, filter_.field)
    if filter_.operation == "eq":
        return field == filter_.value
    elif filter_.operation == "gte":
        return field >= filter_.value
    elif filter_.operation == "lte":
        return field <= filter_.value
    else:
        raise ValueError(f"Operation {filter_.operation} not supported")


def apply_non_topk_filters_to_records(
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


def apply_topk_filter_to_records(
    topk_filter: Filter, records: list[Collection]
) -> list[Collection]:
    vectors = np.stack([getattr(r, topk_filter.field).array for r in records])
    query_vector = topk_filter.value.vector.array
    distances = np.linalg.norm(vectors - query_vector, axis=1)
    topk_indices = distances.argsort()[: topk_filter.value.k]
    return [records[i] for i in topk_indices]


def apply_filters_to_records(
    filters: list[Filter], records: list[Collection]
) -> list[Collection]:
    # split out topk and other filters
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

    records = apply_non_topk_filters_to_records(non_topk_filters, records)

    if len(topk_filters) == 1:
        topk_filter = topk_filters[0]
        records = apply_topk_filter_to_records(topk_filter, records)

    return records


class Engine(ABC):
    @abstractmethod
    def query(self, filter_set: FilterSet) -> list[Collection]:
        pass

    @abstractmethod
    def insert(self, record: Collection) -> None:
        pass

    # @abstractmethod
    # def delete(self, record: Collection) -> None:
    #     pass


class InMemoryEngine(Engine):
    def __init__(self) -> None:  # maybe add option to the init for ANN algo
        self.records: dict[str, list[Collection]] = defaultdict(list)

    def query(self, filter_set: FilterSet = None) -> list[Collection]:
        records = self.records[filter_set.collection]
        if len(filter_set) == 0 or filter_set is None:
            return records

        return apply_filters_to_records(filter_set.filters, records)

    def insert(self, record: Collection) -> None:
        self.records[record.__class__.__name__].append(record)


class LocalStorageEngine(Engine):
    def __init__(self, base_dir: str | PosixPath) -> None:
        self.base_dir = Path(base_dir)
        self.file_paths: dict[str, TextIOWrapper] = {}
        self._collection_name_to_class: dict[str, type] = {}

    def _get_path_to_col(self, col_name: str) -> Path:
        return self.base_dir / f"{col_name}.jsonl"

    def load_records(self, col_name: str) -> list[Collection]:
        jsonl_path = self._get_path_to_col(col_name)
        if not jsonl_path.exists():
            raise RuntimeError(f"Collection {col_name} does not exist.")
        col_cls = self._collection_name_to_class.get(col_name)
        return [
            col_cls.from_json(line)
            for line in jsonl_path.read_text().split("\n")[:-1]
        ]

    def insert(self, record: Collection) -> None:
        col_name = record.__class__.__name__
        self._collection_name_to_class[col_name] = record.__class__
        if col_name not in self.file_paths:
            self.file_paths[col_name] = self._get_path_to_col(col_name)

        with open(self.file_paths[col_name], "a") as f:
            f.write(record.to_json() + "\n")

    def query(self, filter_set: FilterSet) -> list[Collection]:
        records = self.load_records(filter_set.collection)
        if len(filter_set) == 0 or filter_set is None:
            return records

        return apply_filters_to_records(filter_set.filters, records)
