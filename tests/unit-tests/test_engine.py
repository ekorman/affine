import io
from typing import Type

from affine.collection import Collection
from affine.engine import LocalEngine


def test_local_engine(generic_test_engine):
    db = LocalEngine()
    generic_test_engine(db)


def test_local_engine_save_load(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
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

    q1 = db2.query(PersonCollection.objects())
    assert len(q1) == 2
    assert set([p.name for p in q1]) == {"John", "Jane"}

    q2 = db2.query(ProductCollection.objects())
    assert len(q2) == 1
    assert q2[0].name == "Apple"

    # check that id counter was loaded correctly
    assert db2.insert(ProductCollection(name="Banana", price=2.0)) == 2


def test_save_load_from_buffer(
    PersonCollection: Type[Collection],
    ProductCollection: Type[Collection],
    data: list[Collection],
):
    f = io.BytesIO()

    db = LocalEngine()

    for rec in data:
        db.insert(rec)

    db.save(f)
    f.seek(0)

    db2 = LocalEngine()
    db2.load(f)
    assert len(db2.query(PersonCollection.objects())) == 2
    assert len(db2.query(ProductCollection.objects())) == 1
