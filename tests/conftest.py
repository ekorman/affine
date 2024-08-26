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
    db.register_collection(Person)
    db.register_collection(Product)
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


def _test_euclidean_similarity(db: Engine) -> list:
    class TestCol(Collection):
        a: float
        b: Vector[100, Metric.EUCLIDEAN]

    # generate 100 vectors to query against
    records = [
        TestCol(a=float(i), b=Vector([float(i + 1)] * 100)) for i in range(100)
    ]
    db.register_collection(TestCol)
    created_ids = []
    for record in records:
        created_ids.append(db.insert(record))

    # query each vector and check the result
    for i, record in enumerate(records):
        q = (
            db.query(TestCol, with_vectors=True)
            .similarity(TestCol.b == record.b)
            .limit(3)
        )
        assert len(q) == 3
        for j in [-1, 0, 1]:
            idx = i + j
            if idx >= 0 and idx < 100:
                assert records[i + j] in q

    return created_ids


def _test_cosine_similarity(db: Engine):
    # create vectors like [1, 1, 1, ...], [2, 2, 2, ...],
    # and [1 + eps, 1, ...] and make sure cosine is working
    class TestColCosine(Collection):
        a: float
        b: Vector[100, Metric.COSINE]

    db.register_collection(TestColCosine)

    created_ids = []
    for i in range(50):
        created_ids.append(
            db.insert(
                TestColCosine(a=float(2 * i), b=Vector([float(i + 1)] * 100))
            )
        )
        created_ids.append(
            db.insert(
                TestColCosine(
                    a=float(2 * i + 1),
                    b=Vector([float(i + 1) + 1] + [float(i + 1)] * 99),
                )
            )
        )

    for i in range(50):
        q = (
            db.query(TestColCosine, with_vectors=True)
            .similarity(TestColCosine.b == Vector([float(i + 1)] * 100))
            .limit(3)
        )
        assert len(q) == 3
        # all the vectors should have even index
        assert all([int(r.a) % 2 == 0 for r in q])

    return created_ids


@pytest.fixture
def generic_test_euclidean_similarity():
    return _test_euclidean_similarity


@pytest.fixture
def generic_test_cosine_similarity():
    return _test_cosine_similarity
