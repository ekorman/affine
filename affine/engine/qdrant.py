import uuid
from typing import Dict, List, Optional, Type, Union, get_origin

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from affine.collection import Collection, Filter, FilterSet, Vector
from affine.engine import Engine


def create_uuid() -> str:
    return str(uuid.uuid4())


class QdrantEngine(Engine):
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)
        self.created_collections = set()
        self.collection_classes: Dict[str, Type[Collection]] = {}

    def insert(self, record: Collection) -> int:
        collection_class = type(record)
        collection_name = collection_class.__name__
        self.register_collection(collection_class)
        self._ensure_collection_exists(collection_class)

        record.id = create_uuid()

        vector_fields = [
            f.name
            for f in collection_class.__dataclass_fields__.values()
            if get_origin(f.type) == Vector
        ]
        vector = (
            getattr(record, vector_fields[0]).array if vector_fields else [0.0]
        )  # Use a single-dimension vector if no vector field

        point = models.PointStruct(
            id=record.id,
            vector=vector,
            payload=self._convert_collection_to_payload(record),
        )

        self.client.upsert(collection_name=collection_name, points=[point])

        return record.id

    def _ensure_collection_exists(self, collection_class: Type[Collection]):
        collection_name = collection_class.__name__
        if collection_name not in self.created_collections:
            try:
                self.client.get_collection(collection_name)
            except UnexpectedResponse:
                vector_size = self._get_vector_size(collection_class)
                if vector_size == 0:
                    vector_size = 1  # Use a single-dimension vector for collections without vector fields

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
            self.created_collections.add(collection_name)

    def _get_vector_size(self, collection_class: Type[Collection]) -> int:
        vector_fields = [
            f
            for f in collection_class.__dataclass_fields__.values()
            if get_origin(f.type) == Vector
        ]
        if not vector_fields:
            return 0
        return vector_fields[0].type.__args__[0]

    def _convert_collection_to_payload(self, record: Collection) -> dict:
        return {
            f.name: getattr(record, f.name)
            for f in type(record).__dataclass_fields__.values()
            if get_origin(f.type) != Vector
        }

    def register_collection(self, collection_class: Type[Collection]) -> None:
        self.collection_classes[collection_class.__name__] = collection_class

    def _query(self, filter_set: FilterSet) -> List[Collection]:
        collection_name = filter_set.collection
        collection_class = self.collection_classes.get(collection_name)
        if not collection_class:
            raise ValueError(f"Collection {collection_name} not registered")

        self._ensure_collection_exists(collection_class)

        qdrant_filters = self._convert_filters_to_qdrant(filter_set.filters)

        search_params = models.SearchParams(hnsw_ef=128, exact=False)

        topk_filter = next(
            (f for f in filter_set.filters if f.operation == "topk"), None
        )
        if topk_filter:
            vector = topk_filter.value.vector.array
            limit = topk_filter.value.k
            results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=qdrant_filters,
                limit=limit,
                search_params=search_params,
            )
        else:
            limit = 100  # Default limit, adjust as needed
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_filters,
                limit=limit,
            )[
                0
            ]  # scroll returns a tuple (points, next_page_offset)

        return [
            self._convert_qdrant_point_to_collection(point, collection_class)
            for point in results
        ]

    def delete(self, record: Collection) -> None:
        collection_name = record.__class__.__name__
        self.register_collection(record.__class__)
        self._ensure_collection_exists(record.__class__)
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[record.id]),
        )

    def _convert_filters_to_qdrant(
        self, filters: List[Filter]
    ) -> Optional[models.Filter]:
        if not filters:
            return None

        qdrant_conditions = []
        for f in filters:
            if f.operation == "eq":
                if f.field == "id":
                    raise ValueError("Filtering by id is not supported")
                else:
                    qdrant_conditions.append(
                        models.FieldCondition(
                            key=f.field, match=models.MatchValue(value=f.value)
                        )
                    )
            elif f.operation == "gte":
                qdrant_conditions.append(
                    models.FieldCondition(
                        key=f.field, range=models.Range(gte=f.value)
                    )
                )
            elif f.operation == "lte":
                qdrant_conditions.append(
                    models.FieldCondition(
                        key=f.field, range=models.Range(lte=f.value)
                    )
                )
            elif f.operation == "topk":
                # topk is handled separately in the query method
                continue

        return (
            models.Filter(must=qdrant_conditions)
            if qdrant_conditions
            else None
        )

    def _convert_qdrant_point_to_collection(
        self,
        point: Union[models.ScoredPoint, models.Record],
        collection_class: Type[Collection],
    ) -> Collection:
        data = point.payload.copy() if point.payload else {}
        for field_name, field in collection_class.__dataclass_fields__.items():
            if get_origin(field.type) == Vector:
                if field_name in data and isinstance(data[field_name], dict):
                    # Convert the dictionary representation back to a Vector
                    vector_data = data[field_name].get("array", [])
                    data[field_name] = Vector(np.array(vector_data))
                elif hasattr(point, "vector") and point.vector is not None:
                    # If the vector is not in the payload, use the point's vector attribute
                    data[field_name] = Vector(np.array(point.vector))
                else:
                    # If we can't find the vector data, initialize with an empty vector
                    data[field_name] = None
        ret = collection_class(**data)
        ret.id = point.id
        return ret

    def get_elements_by_ids(
        self, collection: Type, ids: List[int]
    ) -> List[Collection]:
        # handle id filter separately uysing client.retrieve
        results = self.client.retrieve(
            collection_name=collection.__name__, ids=ids
        )
        return [
            self._convert_qdrant_point_to_collection(point, collection)
            for point in results
        ]
