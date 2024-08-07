import os
import time

import pytest

from affine.collection import Collection, Metric, Vector
from affine.engine.pinecone import PineconeEngine


class C(Collection):
    a: str
    b: int
    embedding: Vector[2, Metric.EUCLIDEAN]


@pytest.fixture
def db() -> PineconeEngine:
    return PineconeEngine()


@pytest.fixture
def data() -> list[C]:
    return [
        C(
            a="str1",
            b=7,
            embedding=Vector([1.0, 0.2]),
        ),
        C(
            a="str2",
            b=3,
            embedding=Vector([0.1, 0.3]),
        ),
    ]


@pytest.fixture
def created_ids(db: PineconeEngine):
    _created_ids = []
    yield _created_ids

    for id_ in _created_ids:
        db.delete(collection=C, id=id_)


@pytest.mark.skipif(
    os.getenv("PINECONE_API_KEY") is None, reason="No pinecone API key"
)
def test_pinecone_engine(
    db: PineconeEngine, data: list[C], created_ids: list[str]
):
    # try:
    db.register_collection(C)
    for record in data:
        created_ids.append(db.insert(record))

    timeout = 20
    start = time.time()
    while True:
        if len(db.get_elements_by_ids(C, created_ids)) == 2:
            break
        if time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for records to be inserted")

    q1 = (
        db.query(C, with_vectors=True)
        .similarity(C.embedding == [1.1, 0.2])
        .limit(1)
    )
    assert len(q1) == 1
    assert q1[0].a == "str1"

    q2 = (
        db.query(C, with_vectors=True)
        .filter(C.b == 3)
        .similarity(C.embedding == [1.1, 0.2])
        .limit(1)
    )
    assert len(q2) == 1
    assert q2[0].a == "str2"

    assert (
        len(
            db.query(C, with_vectors=True)
            .filter(C.b < 3)
            .similarity(C.embedding == [1.1, 0.2])
            .limit(1)
        )
        == 0
    )
