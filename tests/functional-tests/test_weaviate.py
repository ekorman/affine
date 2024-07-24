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


# def test_weaviate_engine_persistence(
#     Person: Type[Collection],
#     Product: Type[Collection],
#     db: WeaviateEngine,
#     data: list[Collection],
#     weaviate_client,
# ):
#     # Insert data
#     for rec in data:
#         db.insert(rec)

#     # Create a new engine instance
#     db2 = WeaviateEngine(host="localhost", port=8080)
#     db2.register_collection(Person)
#     db2.register_collection(Product)

#     q1 = db2.query(Person.objects())
#     assert len(q1) == 2
#     assert set([p.name for p in q1]) == {"John", "Jane"}

#     q2 = db2.query(Product.objects())
#     assert len(q2) == 1
#     assert q2[0].name == "Apple"


# def test_auto_creation(
#     Person: Type[Collection],
#     Product: Type[Collection],
#     db: WeaviateEngine,
#     weaviate_client,
# ):
#     # This should create the 'Person' class if it doesn't exist
#     db.query(Person.objects())

#     # Verify that the class was created
#     schema = weaviate_client.schema.get()
#     assert any(c["class"] == "Person" for c in schema["classes"])

#     # This should create the 'Product' class if it doesn't exist
#     db.query(Product.objects())

#     # Verify that both classes exist
#     schema = weaviate_client.schema.get()
#     assert any(c["class"] == "Person" for c in schema["classes"])
#     assert any(c["class"] == "Product" for c in schema["classes"])


# def test_unregistered_collection(db: WeaviateEngine):
#     class UnregisteredCollection(Collection):
#         name: str

#     with pytest.raises(
#         ValueError, match="Collection UnregisteredCollection not registered"
#     ):
#         db.query(UnregisteredCollection.objects())
