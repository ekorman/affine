from typing import Type

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

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
    # Register collection classes
    engine.register_collection(PersonCollection)
    engine.register_collection(ProductCollection)
    yield engine
    # Clean up collections after each test
    for collection_name in engine.created_collections:
        try:
            qdrant_client.delete_collection(collection_name)
        except UnexpectedResponse:
            pass  # Collection might not exist, which is fine


def test_qdrant_engine(db: QdrantEngine, generic_test_engine):
    generic_test_engine(db)


def test_qdrant_engine_persistence(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
    db: QdrantEngine,
    data: list[Collection],
    qdrant_client: QdrantClient,
):
    # Insert data
    for rec in data:
        db.insert(rec)

    # Create a new engine instance
    db2 = QdrantEngine(host="localhost", port=6333)
    db2.register_collection(PersonCollection)
    db2.register_collection(ProductCollection)

    q1 = db2.query(PersonCollection.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}

    q2 = db2.query(ProductCollection.objects())
    assert len(q2) == 1
    assert q2[0].name == "Apple"


def test_auto_creation(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
    db: QdrantEngine,
    qdrant_client: QdrantClient,
):
    # This should create the 'Person' collection if it doesn't exist
    db.query(PersonCollection.objects())

    # Verify that the collection was created
    collections = qdrant_client.get_collections().collections
    assert any(c.name == PersonCollection.__name__ for c in collections)

    # This should create the 'Product' collection if it doesn't exist
    db.query(ProductCollection.objects())

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
        db.query(UnregisteredCollection.objects())
