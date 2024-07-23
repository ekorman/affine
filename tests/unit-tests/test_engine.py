import io
from typing import Type

import pytest

from affine.collection import Collection, TopK, Vector
from affine.engine import LocalEngine


def test_local_engine(
    Person: Type[Collection], Product: Type[Collection], data: list[Collection]
):
    db = LocalEngine()

    assert len(db.query(Person.objects())) == 0

    for rec in data:
        db.insert(rec)

    q1 = db.query(Person.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}
    assert set([p.id for p in q1]) == {1, 2}

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
    assert q9[0].id == 1

    db.delete(Product, 1)
    assert db.query(Product.objects()) == []

    with pytest.raises(ValueError) as exc_info:
        db.delete(Product, 1)
    assert "Record with id 1 not found in collection Product" in str(
        exc_info.value
    )

    # next id should be 2
    assert db.insert(Product(name="Banana", price=2.0)) == 2

    # check we can query by id
    q10 = db.query(Product.objects(id=2))
    assert len(q10) == 1
    assert q10[0].name == "Banana"

    assert db.get_element_by_id(Product, 2).name == "Banana"


def test_local_engine_save_load(
    Person: Type[Collection],
    Product: Type[Collection],
    data: list[Collection],
    tmp_path,
):
    db = LocalEngine()

    for rec in data:
        db.insert(rec)

    path = tmp_path / "db.affine"

    db.save(path)

    db2 = LocalEngine()
    db2.load(path)

    q1 = db2.query(Person.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}

    q2 = db2.query(Product.objects())
    assert len(q2) == 1
    assert q2[0].name == "Apple"

    # check that id counter was loaded correctly
    assert db2.insert(Product(name="Banana", price=2.0)) == 2


def test_save_load_from_buffer(
    Person: Type[Collection], Product: Type[Collection], data: list[Collection]
):
    f = io.BytesIO()

    db = LocalEngine()

    for rec in data:
        db.insert(rec)

    db.save(f)
    f.seek(0)

    db2 = LocalEngine()
    db2.load(f)
    assert len(db2.query(Person.objects())) == 2
    assert len(db2.query(Product.objects())) == 1