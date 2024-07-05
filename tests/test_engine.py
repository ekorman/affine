from affine.collection import Collection
from affine.engine import InMemoryEngine


def test_local_engine():
    db = InMemoryEngine()

    class Person(Collection):
        name: str
        age: int
        # embedding: Vector[2]

    db.insert(Person(name="John", age=20))
    db.insert(Person(name="Jane", age=30))

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

    # q4 = db.query(Person.objects(embedding__topk=TopK(Vector([1, 2]), 3))

    # Person.objects(embedding__topk=TopK(Vector([1, 2]), 3))
