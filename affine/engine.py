import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

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


class LocalEngine(Engine):
    def __init__(
        self, path: str | Path = None
    ) -> None:  # maybe add option to the init for ANN algo
        self.path = path
        self.records: dict[str, list[Collection]] = defaultdict(list)
        if self.path is not None:
            with open(self.path, "rb") as f:
                self.records = pickle.load(f)
        self.collection_id_counter: dict[str, int] = defaultdict(int)
        for k, recs in self.records.items():
            if len(recs) > 0:
                self.collection_id_counter[k] = max([r.id for r in recs])

    def save(self, path: str | Path) -> None:
        path = path or self.path
        with open(path, "wb") as f:
            pickle.dump(self.records, f)

    def query(self, filter_set: FilterSet = None) -> list[Collection]:
        records = self.records[filter_set.collection]
        if len(filter_set) == 0 or filter_set is None:
            return records

        return apply_filters_to_records(filter_set.filters, records)

    def insert(self, record: Collection) -> int:
        record.id = self.collection_id_counter[record.__class__.__name__] + 1
        self.records[record.__class__.__name__].append(record)
        self.collection_id_counter[record.__class__.__name__] = record.id

        if self.path is not None:
            self.save(self.path)

        return record.id
