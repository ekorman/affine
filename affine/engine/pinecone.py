import os
import uuid
from typing import Any, Dict, Type

from pinecone import Index, Pinecone, PodSpec, ScoredVector, ServerlessSpec
from pinecone import Vector as PineconeVector

from affine.collection import Collection, Filter, FilterSet, Metric, Similarity
from affine.engine import Engine


def create_uuid() -> str:
    return str(uuid.uuid4())


def _convert_filters_to_pinecone(
    filters: list[Filter],
) -> dict[str, dict[str, Any]]:
    """Converts filters to Pinecone's format."""

    if not filters:
        return {}

    ret = {}
    for filter_ in filters:
        ret[filter_.field] = {f"${filter_.operation}": filter_.value}

    return ret


class PineconeEngine(Engine):
    def __init__(
        self, api_key: str = None, spec: ServerlessSpec | PodSpec | None = None
    ):
        """Engien for interacting with Pinecone.

        Parameters
        ----------
        api_key
            The Pinecone API key. If not provided, it will be read from the
            environment variable PINECONE_API_KEY.
        spec
            The PodSpec or ServerlessSpec object. If not provided, a `ServerlessSpec`
            will be created from the environment variables PINECONE_CLOUD and
            PINECONE_REGION.
        """
        # allow getting api_key from env variable
        if spec is None:
            spec = ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD"),
                region=os.getenv("PINECONE_REGION"),
            )
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.client = Pinecone(api_key=api_key)
        self.spec = spec
        self.collection_classes: Dict[str, Type[Collection]] = {}

    def _get_collections_vector_field_name_dim_and_metric(
        self, collection_class: Type[Collection]
    ) -> tuple[str, int, Metric]:
        vfs = collection_class.get_vector_fields()
        # TODO: try to handle 0 or multiple vfs?
        if len(vfs) != 1:
            raise ValueError(
                f"To use {self.__class__.__name__}, a collection {collection_class.__name__} must have exactly one vector field"
                f" but found {len(vfs)} in collection {collection_class.__name__}."
            )

        name, dim, metric = vfs[0]
        return name, dim, metric

    def _convert_pinecone_to_collection(
        self,
        pc_record: ScoredVector | PineconeVector,
        collection_class: Type[Collection],
    ) -> Collection:
        vf_name, _, _ = self._get_collections_vector_field_name_dim_and_metric(
            collection_class
        )
        kwargs = pc_record.metadata.copy()
        if pc_record.values:
            kwargs[vf_name] = pc_record.values
        else:
            kwargs[vf_name] = None

        ret = collection_class(**kwargs)
        ret.id = pc_record.id
        return ret

    def register_collection(
        self, collection_class: Type[Collection], exists_ok: bool = True
    ) -> None:
        (
            _,
            dim,
            metric,
        ) = self._get_collections_vector_field_name_dim_and_metric(
            collection_class
        )

        collection_name = collection_class.__name__.lower()

        current_collections = [
            idx["name"] for idx in self.client.list_indexes()
        ]
        if not exists_ok or collection_name not in current_collections:
            self.client.create_index(
                name=collection_name,
                spec=self.spec,
                dimension=dim,
                metric=metric.value,
            )
            self.collection_classes[collection_name] = collection_class
        else:
            self.collection_classes[collection_name] = collection_class

    def _get_index(self, collection_name: str) -> Index:
        return self.client.Index(collection_name.lower())

    def get_elements_by_ids(
        self, collection: Type[Collection], ids: list[int]
    ) -> list[Collection]:
        index = self._get_index(collection.__name__)

        return [
            self._convert_pinecone_to_collection(r, collection)
            for r in index.fetch(ids).vectors.values()
        ]

    def _query(
        self,
        filter_set: FilterSet,
        with_vectors: bool = False,
        similarity: Similarity | None = None,
        limit: int | None = None,
    ) -> list[Collection]:
        filter_ = _convert_filters_to_pinecone(filter_set.filters)
        index = self._get_index(filter_set.collection)
        if similarity is None:
            raise ValueError(
                "Pinecone queries require a vector in every query"
            )
        else:
            vector = similarity.get_array().tolist()

        ret = index.query(
            top_k=limit,
            vector=vector,
            filter=filter_,
            include_metadata=True,
            include_values=with_vectors,
        ).matches

        return [
            self._convert_pinecone_to_collection(
                r, self.collection_classes[filter_set.collection.lower()]
            )
            for r in ret
        ]

    def _delete_by_id(self, collection: Collection, id: str) -> None:
        index = self._get_index(collection.__name__)
        index.delete([id])

    def insert(self, record: Collection) -> str:
        index = self._get_index(record.__class__.__name__)

        vf_name, _, _ = self._get_collections_vector_field_name_dim_and_metric(
            record.__class__
        )
        uid = create_uuid()
        index.upsert(
            [
                PineconeVector(
                    id=uid,
                    values=getattr(record, vf_name).array.tolist(),
                    metadata=record.get_non_vector_dict(),
                )
            ]
        )
        return uid
