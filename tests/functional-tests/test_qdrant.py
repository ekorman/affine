from typing import Type

import pytest
from qdrant_client import QdrantClient

from affine.collection import Collection
from affine.engine import QdrantEngine


@pytest.fixture(scope="module")
def qdrant_client():
    client = QdrantClient(host="localhost", port=6333)
    yield client
    # Clean up collections after tests
    for collection in client.get_collections().collections:
        client.delete_collection(collection.name)


@pytest.fixture(scope="function")
def db(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
    qdrant_client,
):
    engine = QdrantEngine(host="localhost", port=6333)
    engine.register_collection(PersonCollection)
    engine.register_collection(ProductCollection)
    return engine


def test_qdrant_engine(db: QdrantEngine, generic_test_engine):
    generic_test_engine(db)


def test_auto_creation(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
    db: QdrantEngine,
    qdrant_client: QdrantClient,
):
    # This should create the 'Person' collection if it doesn't exist
    db.query(PersonCollection).all()

    # Verify that the collection was created
    collections = qdrant_client.get_collections().collections
    assert any(c.name == PersonCollection.__name__ for c in collections)

    # This should create the 'Product' collection if it doesn't exist
    db.query(ProductCollection).all()

    # Verify that both collections exist
    collections = qdrant_client.get_collections().collections
    assert any(c.name == PersonCollection.__name__ for c in collections)
    assert any(c.name == ProductCollection.__name__ for c in collections)


def test_unregistered_collection(db: QdrantEngine):
    class UnregisteredCollection(Collection):
        name: str

    with pytest.raises(
        ValueError, match="Collection UnregisteredCollection not registered"
    ):
        db.query(UnregisteredCollection).all()
