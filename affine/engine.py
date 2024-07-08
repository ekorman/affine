from collections import defaultdict

import numpy as np

from affine.collection import Collection, Filter, FilterSet


class InMemoryEngine:
    def __init__(self):  # maybe add option to the init for ANN algo
        self.records: dict[str, list[Collection]] = defaultdict(list)

    @staticmethod
    def apply_filter(filter_: Filter, record: Collection) -> bool:
        field = getattr(record, filter_.field)
        if filter_.operation == "eq":
            return field == filter_.value
        elif filter_.operation == "gte":
            return field >= filter_.value
        elif filter_.operation == "lte":
            return field <= filter_.value
        else:
            raise ValueError(f"Operation {filter_.operation} not supported")

    def _apply_non_topk_filters(
        self, filters: list[Filter], records: list[Collection]
    ) -> list[Collection]:
        ret = []
        for record in records:
            record_match = True
            for f in filters:
                if not self.apply_filter(f, record):
                    record_match = False
                    break
            if record_match:
                ret.append(record)
        return ret

    def _apply_topk_filter(
        self, topk_filter: Filter, records: list[Collection]
    ) -> list[Collection]:
        vectors = np.stack(
            [getattr(r, topk_filter.field).array for r in records]
        )
        query_vector = topk_filter.value.vector.array
        distances = np.linalg.norm(vectors - query_vector, axis=1)
        topk_indices = distances.argsort()[: topk_filter.value.k]
        return [records[i] for i in topk_indices]

    def query(self, filter_set: FilterSet = None) -> list[Collection]:
        if len(filter_set) == 0 or filter_set is None:
            return self.records[filter_set.collection]

        # split out topk and other filters
        topk_filters = []
        non_topk_filters = []
        for f in filter_set.filters:
            if f.operation == "topk":
                topk_filters.append(f)
            else:
                non_topk_filters.append(f)

        if len(topk_filters) > 1:
            raise ValueError(
                f"Only one topk filter is allowed but got {len(topk_filters)}."
            )

        records = self._apply_non_topk_filters(
            non_topk_filters, self.records[filter_set.collection]
        )

        if len(topk_filters) == 1:
            topk_filter = topk_filters[0]
            records = self._apply_topk_filter(topk_filter, records)

        return records

    def insert(self, record: Collection) -> None:
        self.records[record.__class__.__name__].append(record)


class LocalStorageEnging:
    pass
