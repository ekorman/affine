from collections import defaultdict

from affine.collection import Collection, Filter, FilterSet


class InMemoryEngine:
    def __init__(self):  # maybe add option to the init for ANN algo
        self.records: dict[str, list[Collection]] = defaultdict(list)

    @staticmethod
    def apply_filter(filter: Filter, record: Collection) -> bool:
        field = getattr(record, filter.field)
        if filter.operation == "eq":
            return field == filter.value
        elif filter.operation == "gte":
            return field >= filter.value
        elif filter.operation == "lte":
            return field <= filter.value
        else:
            raise ValueError(f"Operation {filter.operation} not supported")

    def query(self, filter_set: FilterSet = None) -> list[Collection]:
        if len(filter_set) == 0 or filter_set is None:
            return self.records[filter_set.collection]
        ret = []
        for record in self.records[filter_set.collection]:
            record_match = True
            for f in filter_set.filters:
                if not self.apply_filter(f, record):
                    record_match = False
                    break
            if record_match:
                ret.append(record)
        return ret

    def insert(self, record: Collection) -> None:
        self.records[record.__class__.__name__].append(record)


class LocalStorageEnging:
    pass
