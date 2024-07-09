import pytest

from affine.collection import Collection, TopK, Vector
from affine.engine import InMemoryEngine, LocalStorageEngine


class Person(Collection):
    name: str
    age: int
    embedding: Vector[2]


@pytest.fixture
def data() -> list[Person]:
    return [
        Person(name="John", age=20, embedding=Vector([3.0, 0.0])),
        Person(name="Jane", age=30, embedding=Vector([1.0, 2.0])),
    ]


def _test_local_engine(data: list[Person], db):
    for person in data:
        db.insert(person)

    q1 = db.query(Person.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}

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

    q6 = db.query(Person.objects(age__gte=25, name__eq="Jane"))
    assert len(q6) == 1
    assert q6[0].name == "Jane"

    q7 = db.query(Person.objects(embedding__topk=TopK(Vector([1.8, 2.3]), 1)))
    assert len(q7) == 1
    assert q7[0].name == "Jane"

    q8 = db.query(Person.objects(embedding__topk=TopK(Vector([1.8, 2.3]), 2)))
    assert len(q8) == 2


def test_local_in_memory_engine(data: list[Person]):
    _test_local_engine(data, InMemoryEngine())


def test_local_storage_engine(data: list[Person], tmp_path):
    _test_local_engine(data, LocalStorageEngine(tmp_path))
