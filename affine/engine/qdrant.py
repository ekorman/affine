import uuid
from typing import Dict, List, Optional, Type, Union, get_origin

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from affine.collection import (
    Collection,
    Filter,
    FilterSet,
    Metric,
    Similarity,
    Vector,
)
from affine.engine import Engine


def create_uuid() -> str:
    return str(uuid.uuid4())


def _convert_filters_to_qdrant(
    filters: List[Filter],
) -> Optional[models.Filter]:
    """Converts filters to Qdrant's format."""
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
        elif f.operation in ["gte", "lte", "gt", "lt"]:
            qdrant_conditions.append(
                models.FieldCondition(
                    key=f.field, range=models.Range(**{f.operation: f.value})
                )
            )
        else:
            raise ValueError(f"Unsupported filter operation: {f.operation}")

    return models.Filter(must=qdrant_conditions) if qdrant_conditions else None


class QdrantEngine(Engine):

    _RETURNS_NORMALIZED_FOR_COSINE = True

    qdrant_dists = {
        Metric.EUCLIDEAN: models.Distance.EUCLID,
        Metric.COSINE: models.Distance.COSINE,
    }

    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port)
        self.created_collections = set()
        self.collection_classes: Dict[str, Type[Collection]] = {}

    def insert(self, record: Collection) -> str:
        collection_class = type(record)
        collection_name = collection_class.__name__
        self.register_collection(collection_class)
        self._ensure_collection_exists(collection_class)

        record.id = create_uuid()

        vector = {
            name: getattr(record, name).array
            for name, _, _ in record.get_vector_fields()
        }

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
                vf_info = collection_class.get_vector_fields()
                vectors_config = {
                    name: models.VectorParams(
                        size=size,
                        distance=self.qdrant_dists[distance],
                    )
                    for name, size, distance in vf_info
                }

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config,
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

    def _query(
        self,
        filter_set: FilterSet,
        similarity: Similarity | None = None,
        limit: int | None = None,
        with_vectors: bool = False,
    ) -> list[Collection]:
        collection_name = filter_set.collection
        collection_class = self.collection_classes.get(collection_name)
        if not collection_class:
            raise ValueError(f"Collection {collection_name} not registered")

        self._ensure_collection_exists(collection_class)

        qdrant_filters = _convert_filters_to_qdrant(filter_set.filters)

        search_params = models.SearchParams(hnsw_ef=128, exact=False)
        if similarity:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(
                    name=similarity.field, vector=similarity.get_list()
                ),
                query_filter=qdrant_filters,
                limit=limit,
                with_vectors=with_vectors,
                search_params=search_params,
            )
        else:
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_filters,
                limit=limit,
                with_vectors=with_vectors,
            )[
                0
            ]  # scroll returns a tuple (points, next_page_offset)

        return [
            self._convert_qdrant_point_to_collection(point, collection_class)
            for point in results
        ]

    def _delete_by_id(self, collection: Type[Collection], id: str) -> None:
        collection_name = collection.__name__
        self.register_collection(collection)
        self._ensure_collection_exists(collection)
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[id]),
        )

    def _convert_qdrant_point_to_collection(
        self,
        point: Union[models.ScoredPoint, models.Record],
        collection_class: Type[Collection],
    ) -> Collection:
        kwargs = point.payload.copy() if point.payload else {}
        for name, _, _ in collection_class.get_vector_fields():
            if point.vector is None or name not in point.vector:
                kwargs[name] = None
            else:
                kwargs[name] = Vector(np.array(point.vector[name]))

        ret = collection_class(**kwargs)
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
