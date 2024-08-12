from typing import Type

import pytest
import weaviate

from affine.collection import Collection
from affine.engine import WeaviateEngine


@pytest.fixture(scope="module")
def weaviate_client():
    client = weaviate.connect_to_local("localhost", "8080")
    collections = client.collections.list_all()
    if len(collections) > 0:
        raise RuntimeError(
            f"Tests should be run on an empty Weaviate instance but found collections {collections}"
        )

    yield client
    # Clean up collections after tests
    for class_name in client.collections.list_all():
        client.collections.delete(class_name)


@pytest.fixture(scope="function")
def db(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
    weaviate_client,
):
    engine = WeaviateEngine(host="localhost", port=8080)
    # Register collection classes
    engine.register_collection(PersonCollection)
    engine.register_collection(ProductCollection)
    return engine


def test_weaviate_engine(db: WeaviateEngine, generic_test_engine):
    generic_test_engine(db)


def test_similarity(db: WeaviateEngine, generic_test_similarity):
    generic_test_similarity(db)


def test_unregistered_collection(db: WeaviateEngine):
    class UnregisteredCollection(Collection):
        name: str

    with pytest.raises(
        ValueError, match="Collection UnregisteredCollection not registered"
    ):
        db.query(UnregisteredCollection).all()
