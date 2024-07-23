import pytest

from affine.collection import Collection, Vector


def test_vector_validation():
    class C(Collection):
        x: Vector[3]

    with pytest.raises(ValueError) as exc_info:
        C(x=[1, 2])
    assert "Expected vector of length 3, got 2" in str(exc_info.value)

    try:
        C(x=[1, 2, 3])
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
