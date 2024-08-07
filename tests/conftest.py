import pytest

from affine.collection import Collection, Metric, Vector
from affine.engine import Engine, LocalEngine


class Person(Collection):
    name: str
    age: int
    embedding: Vector[2, Metric.EUCLIDEAN]
    other_embedding: Vector[3, Metric.COSINE]


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
    Person(
        name="John",
        age=20,
        embedding=Vector([3.0, 0.0]),
        other_embedding=Vector([-1.0, 7.0, 0.0]),
    ),
    Person(
        name="Jane",
        age=30,
        embedding=Vector([1.0, 2.0]),
        other_embedding=Vector([10.7, 0.1, -5.0]),
    ),
    Product(name="Apple", price=1.0),
]


@pytest.fixture
def data():
    return _data


def _test_engine(db: Engine):
    assert len(db.query(Person).all()) == 0

    for rec in _data:
        db.insert(rec)

    q1 = db.query(Person).all()
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}
    # check unique ids
    assert len(set([p.id for p in q1])) == 2

    q2 = db.query(Person).filter(Person.name == "John").all()
    assert len(q2) == 1
    assert q2[0].name == "John"

    # for non-local engine vector fields should be none
    if not isinstance(db, LocalEngine):
        assert q2[0].embedding is None
        assert q2[0].other_embedding is None

    q3 = db.query(Person).filter(Person.age >= 25).all()
    assert len(q3) == 1
    assert q3[0].name == "Jane"

    q4 = db.query(Person).filter(Person.age <= 25).all()
    assert len(q4) == 1
    assert q4[0].name == "John"

    assert len(db.query(Person).filter(Person.age < 20).all()) == 0

    assert len(db.query(Person).filter(Person.age > 30).all()) == 0

    q5 = (
        db.query(Person)
        .filter((Person.age <= 25) & (Person.name == "Jane"))
        .all()
    )
    assert len(q5) == 0

    q6 = (
        db.query(Person)
        .filter((Person.age >= 25) & (Person.name == "Jane"))
        .all()
    )
    assert len(q6) == 1
    assert q6[0].name == "Jane"

    q7 = db.query(Person).similarity(Person.embedding == [1.8, 2.3]).limit(1)
    assert len(q7) == 1
    assert q7[0].name == "Jane"

    q8 = db.query(Person).similarity(Person.embedding == [1.8, 2.3]).all()
    assert len(q8) == 2

    q9 = db.query(Product).all()
    assert len(q9) == 1
    assert q9[0].name == "Apple"

    # check we can query by id
    assert db.query(Product).get_by_id(q9[0].id).name == "Apple"

    # check we can delete
    db.delete(record=q9[0])
    assert db.query(Product).all() == []

    # for non-local engines check `with_vector`
    if not isinstance(db, LocalEngine):
        q10 = (
            db.query(Person, with_vectors=True)
            .filter(Person.name == "Jane")
            .all()
        )
        assert len(q10) == 1
        assert q10[0].embedding == Vector([1.0, 2.0])

        if db._RETURNS_NORMALIZED_FOR_COSINE:
            assert (
                q10[0].other_embedding == Vector([10.7, 0.1, -5.0]).normalize()
            )
        else:
            assert q10[0].other_embedding == Vector([10.7, 0.1, -5.0])


@pytest.fixture
def generic_test_engine():
    return _test_engine
