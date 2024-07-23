import pytest

from affine.collection import Collection, Vector


class PersonCollection(Collection):
    name: str
    age: int
    embedding: Vector[2]


class ProductCollection(Collection):
    name: str
    price: float


@pytest.fixture
def Person():
    return PersonCollection


@pytest.fixture
def Product():
    return ProductCollection


@pytest.fixture
def data(Person, Product):
    return [
        Person(name="John", age=20, embedding=Vector([3.0, 0.0])),
        Person(name="Jane", age=30, embedding=Vector([1.0, 2.0])),
        Product(name="Apple", price=1.0),
    ]
