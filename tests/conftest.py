import pytest

from affine.collection import Collection, TopK, Vector
from affine.engine import Engine


class Person(Collection):
    name: str
    age: int
    embedding: Vector[2]


class Product(Collection):
    name: str
    price: float


@pytest.fixture
def PersonCollection():
    return Person


@pytest.fixture
def ProductCollection():
    return Product


_data = [
    Person(name="John", age=20, embedding=Vector([3.0, 0.0])),
    Person(name="Jane", age=30, embedding=Vector([1.0, 2.0])),
    Product(name="Apple", price=1.0),
]


@pytest.fixture
def data():
    return _data


def _test_engine(db: Engine):
    assert len(db.query(Person.objects())) == 0

    for rec in _data:
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

    # check we can query by id
    assert db.get_element_by_id(Product, q9[0].id).name == "Apple"

    db.delete(Product, q9[0].id)
    assert db.query(Product.objects()) == []


@pytest.fixture
def generic_test_engine():
    return _test_engine
