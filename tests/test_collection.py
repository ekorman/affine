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


def test_json_encoder_decoder():
    class Col(Collection):
        x: int
        y: str
        z: Vector[2]

    c = Col(x=1, y="hello", z=Vector([1.0, 2.0]))

    expected_json = '{"x": 1, "y": "hello", "z": [1.0, 2.0]}'

    assert c.to_json() == expected_json

    reloaded = Col.from_json(expected_json)
    assert reloaded.x == c.x
    assert reloaded.y == c.y
    assert isinstance(reloaded.z, Vector)
    assert reloaded.z == c.z
