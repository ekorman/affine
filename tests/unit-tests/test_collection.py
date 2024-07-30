import pytest

from affine.collection import Collection, Metric, Vector


def test_vector_validation():
    class C(Collection):
        x: Vector[3, Metric.EUCLIDEAN]

    with pytest.raises(ValueError) as exc_info:
        C(x=[1, 2])
    assert "Expected vector of length 3, got 2" in str(exc_info.value)

    try:
        C(x=[1, 2, 3])
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_vector_fields_method():
    class C(Collection):
        x: Vector[3, Metric.COSINE]
        y: Vector[2, Metric.EUCLIDEAN]
        z: str

    assert set(C.get_vector_fields()) == {
        ("x", 3, Metric.COSINE),
        ("y", 2, Metric.EUCLIDEAN),
    }


def test_collection_equality():
    class C(Collection):
        x: int
        y: str
        z: Vector[2, Metric.COSINE]

    c1 = C(x=1, y="a", z=Vector([1.0, 2.0]))
    c2 = C(x=1, y="a", z=Vector([1.0, 2.00000000000001]))
    c3 = C(x=1, y="b", z=Vector([1.0, 2.001]))
    c4 = C(x=1, y="a", z=Vector([1.0, 3.0]))
    c5 = C(x=1, y="b", z=Vector([1.0, 2.0]))

    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert c1 != c5
    assert c3 != c4
