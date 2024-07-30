from typing import Dict, Type

from pinecone import Pinecone

from affine.collection import Collection
from affine.engine import Engine

# for pinecone will have to restrict to collections with a single index


class PineconeEngine(Engine):
    def __init__(self, api_key: str):
        self.client = Pinecone(api_key=api_key)
        self.collection_classes: Dict[str, Type[Collection]] = {}

    def register_collection(self, collection_class: Type[Collection]) -> None:
        vfs = collection_class.get_vector_fields()
        if len(vfs) != 1:
            raise ValueError(
                f"Collection {collection_class.__name__} must have exactly one vector field"
            )
        _, dim = vfs[0]

        collection_name = collection_class.__name__
        self.collection_classes[collection_name] = collection_class

        # self.client.create_index(collection_name, dim, metric=metric)
