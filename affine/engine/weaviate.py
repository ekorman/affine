from typing import Dict, List, Type, get_origin

import weaviate
from weaviate.classes import query
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    VectorDistances,
)
from weaviate.collections import Collection as WeaviateCollection
from weaviate.collections.classes.filters import _FilterValue
from weaviate.collections.classes.internal import Object

from affine.collection import (
    Collection,
    Filter,
    FilterSet,
    Metric,
    Similarity,
    Vector,
)
from affine.engine import Engine


def _build_where_filter(filters: List[Filter]) -> _FilterValue:
    if len(filters) == 0:
        return None
    where_filters = []

    filter_op_to_weaviate_filter_method = {
        "eq": "equal",
        "gte": "greater_or_equal",
        "lte": "less_or_equal",
        "gt": "greater_than",
        "lt": "less_than",
    }

    for filter_ in filters:
        weaviate_filter = query.Filter.by_property(filter_.field)
        weaviate_filter = getattr(
            weaviate_filter,
            filter_op_to_weaviate_filter_method[filter_.operation],
        )(filter_.value)

        where_filters.append(weaviate_filter)

    ret = where_filters[0]
    for f in where_filters[1:]:
        ret = ret & f
    return ret


def weaviate_object_to_collection_object(
    obj: Object, collection_cls: Type[Collection]
) -> Collection:
    kwargs = obj.properties.copy()

    for vector_name, _, _ in collection_cls.get_vector_fields():
        kwargs[vector_name] = (
            Vector(obj.vector[vector_name])
            if vector_name in obj.vector
            else None
        )

    ret = collection_cls(**kwargs)
    ret.id = str(obj.uuid)
    return ret


class WeaviateEngine(Engine):

    weaviate_dists = {
        Metric.EUCLIDEAN: VectorDistances.L2_SQUARED,
        Metric.COSINE: VectorDistances.COSINE,
    }

    def __init__(self, host: str, port: int):
        self.client = weaviate.connect_to_local(host=host, port=port)
        self.collection_classes: Dict[str, Type[Collection]] = {}

    def register_collection(self, collection_class: Type[Collection]) -> None:
        collection_name = collection_class.__name__
        self.collection_classes[collection_name] = collection_class

        # Check if the class already exists in Weaviate
        if not self.client.collections.exists(collection_name):
            properties = []
            for (
                field_name,
                field,
            ) in collection_class.__dataclass_fields__.items():
                if field_name != "id":
                    if get_origin(field.type) != Vector:
                        data_type = (
                            DataType.TEXT
                            if field.type == str
                            else DataType.NUMBER
                        )
                        properties.append(
                            Property(name=field_name, data_type=data_type)
                        )
            if len(collection_class.get_vector_fields()) > 0:
                vectorizer_config = [
                    Configure.NamedVectors.none(
                        name,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=self.weaviate_dists[dist]
                        ),
                    )
                    for name, _, dist in collection_class.get_vector_fields()
                ]
            else:
                vectorizer_config = None

            self.client.collections.create(
                name=collection_name,
                properties=properties,
                vectorizer_config=vectorizer_config,
            )
        else:
            print(f"Class {collection_name} already exists in Weaviate")

    def insert(self, record: Collection) -> str:
        collection_class = type(record)
        collection_name = collection_class.__name__

        if collection_name not in self.collection_classes:
            self.register_collection(collection_class)

        col = self.client.collections.get(collection_name)

        data_object = {}
        vector = {}
        for field_name, field in collection_class.__dataclass_fields__.items():
            if field_name != "id":
                value = getattr(record, field_name)
                if get_origin(field.type) == Vector:
                    vector[field_name] = value.array
                else:
                    data_object[field_name] = value

        record.id = str(col.data.insert(data_object, vector=vector))

        return record.id

    def get_weaviate_collection_and_affine_collection_class(
        self, collection_name: str
    ) -> tuple[WeaviateCollection, Type[Collection]]:
        collection_class = self.collection_classes.get(collection_name)
        if not collection_class:
            raise ValueError(f"Collection {collection_name} not registered")
        col = self.client.collections.get(collection_name)
        return col, collection_class

    def _query(
        self,
        filter_set: FilterSet,
        with_vectors: bool = False,
        similarity: Similarity | None = None,
        limit: int | None = None,
    ) -> list[Collection]:
        (
            col,
            collection_class,
        ) = self.get_weaviate_collection_and_affine_collection_class(
            filter_set.collection
        )

        where_filter = _build_where_filter(filter_set.filters)
        if similarity:
            result = col.query.near_vector(
                similarity.get_list(),
                target_vector=similarity.field,
                filters=where_filter,
                include_vector=with_vectors,
                limit=limit,
            ).objects
        else:
            result = col.query.fetch_objects(
                filters=where_filter, include_vector=with_vectors
            ).objects

        return [
            weaviate_object_to_collection_object(obj, collection_class)
            for obj in result
        ]

    def _delete_by_id(self, collection: Type[Collection], id: str) -> None:
        col, _ = self.get_weaviate_collection_and_affine_collection_class(
            collection.__name__
        )
        col.data.delete_by_id(id)

    def get_elements_by_ids(
        self, collection: Type[Collection], ids: List[str]
    ) -> List[Collection]:
        (
            col,
            collection_class,
        ) = self.get_weaviate_collection_and_affine_collection_class(
            collection.__name__
        )

        return [
            weaviate_object_to_collection_object(
                col.query.fetch_object_by_id(id_), collection_class
            )
            for id_ in ids
        ]
