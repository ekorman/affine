import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from affine.collection import Collection, TopK, Vector
from affine.engine import QdrantEngine


class Person(Collection):
    name: str
    age: int
    embedding: Vector[2]


class Product(Collection):
    name: str
    price: float


@pytest.fixture(scope="module")
def qdrant_client():
    client = QdrantClient(host="localhost", port=6333)
    yield client
    # Clean up collections after tests
    for collection in client.get_collections().collections:
        client.delete_collection(collection.name)


@pytest.fixture(scope="function")
def db(qdrant_client):
    engine = QdrantEngine(host="localhost", port=6333)
    # Register collection classes
    engine.register_collection(Person)
    engine.register_collection(Product)
    yield engine
    # Clean up collections after each test
    for collection_name in engine.created_collections:
        try:
            qdrant_client.delete_collection(collection_name)
        except UnexpectedResponse:
            pass  # Collection might not exist, which is fine


@pytest.fixture
def data():
    return [
        Person(name="John", age=20, embedding=Vector([3.0, 0.0])),
        Person(name="Jane", age=30, embedding=Vector([1.0, 2.0])),
        Product(name="Apple", price=1.0),
    ]


def test_qdrant_engine(db: QdrantEngine, data: list[Collection]):
    assert len(db.query(Person.objects())) == 0

    for rec in data:
        db.insert(rec)

    q1 = db.query(Person.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}
    assert len(set([p.id for p in q1])) == 2

    q2 = db.query(Person.objects(name__eq="John"))
    assert len(q2) == 1
    assert q2[0].name == "John"

    q3 = db.query(Person.objects(age__gte=25))
    assert len(q3) == 1
    assert q3[0].name == "Jane"

    q4 = db.query(Person.objects(age__lte=25))
    assert len(q4) == 1
    assert q4[0].name == "John"

    q5 = db.query(Person.objects(age__lte=25, name__eq="Jane"))
    assert len(q5) == 0

    q6 = db.query(Person.objects(age__gte=25, name="Jane"))
    assert len(q6) == 1
    assert q6[0].name == "Jane"

    q7 = db.query(Person.objects(embedding__topk=TopK(Vector([1.8, 2.3]), 1)))
    assert len(q7) == 1
    assert q7[0].name == "Jane"

    q8 = db.query(Person.objects(embedding__topk=TopK(Vector([1.8, 2.3]), 2)))
    assert len(q8) == 2

    q9 = db.query(Product.objects())
    assert len(q9) == 1
    assert q9[0].name == "Apple"

    db.delete(Product, q9[0].id)
    assert db.query(Product.objects()) == []

    id_ = db.insert(Product(name="Banana", price=2.0))

    # check we can query by id
    assert db.get_element_by_id(Product, id_).name == "Banana"


def test_qdrant_engine_persistence(
    db: QdrantEngine, data: list[Collection], qdrant_client: QdrantClient
):
    # Insert data
    for rec in data:
        db.insert(rec)

    # Create a new engine instance
    db2 = QdrantEngine(host="localhost", port=6333)
    db2.register_collection(Person)
    db2.register_collection(Product)

    q1 = db2.query(Person.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}

    q2 = db2.query(Product.objects())
    assert len(q2) == 1
    assert q2[0].name == "Apple"


def test_auto_creation(db: QdrantEngine, qdrant_client: QdrantClient):
    # This should create the 'Person' collection if it doesn't exist
    db.query(Person.objects())

    # Verify that the collection was created
    collections = qdrant_client.get_collections().collections
    assert any(c.name == "Person" for c in collections)

    # This should create the 'Product' collection if it doesn't exist
    db.query(Product.objects())

    # Verify that both collections exist
    collections = qdrant_client.get_collections().collections
    assert any(c.name == "Person" for c in collections)
    assert any(c.name == "Product" for c in collections)


def test_unregistered_collection(db: QdrantEngine):
    class UnregisteredCollection(Collection):
        name: str

    with pytest.raises(
        ValueError, match="Collection UnregisteredCollection not registered"
    ):
        db.query(UnregisteredCollection.objects())
