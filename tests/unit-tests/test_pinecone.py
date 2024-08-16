from unittest.mock import patch

import pytest
from pinecone import ScoredVector

from affine.collection import (
    Collection,
    Filter,
    FilterSet,
    Metric,
    Similarity,
    Vector,
)
from affine.engine.pinecone import (
    PineconeEngine,
    _convert_filters_to_pinecone,
    create_uuid,
)


@pytest.fixture
def mock_pinecone_client():
    with patch("affine.engine.pinecone.Pinecone") as MockPinecone:
        yield MockPinecone


@pytest.fixture
def engine(mock_pinecone_client):
    return PineconeEngine()


def test_create_uuid():
    uuid_str = create_uuid()
    assert isinstance(uuid_str, str)


def test_convert_filters_to_pinecone():
    filters = [
        Filter(collection="C", field="field1", operation="eq", value="value1")
    ]

    assert _convert_filters_to_pinecone(filters) == {
        "field1": {"$eq": "value1"}
    }


def test_register_collection(engine, mock_pinecone_client):
    class C(Collection):
        field1: str
        embedding: Vector[32, Metric.EUCLIDEAN]

    mock_client_instance = mock_pinecone_client.return_value
    mock_client_instance.list_indexes.return_value = []

    engine.register_collection(C)

    mock_client_instance.create_index.assert_called_once_with(
        name="c",
        spec=engine.spec,
        dimension=32,
        metric="euclidean",
    )

    # check we get an error if trying to register collections without
    # exactly one vector field
    class D(Collection):
        field1: str
        field2: str

    with pytest.raises(ValueError) as exc_info:
        engine.register_collection(D)
    assert "must have exactly one vector field" in str(exc_info)

    class E(Collection):
        field1: str
        field2: Vector[2, Metric.EUCLIDEAN]
        field3: Vector[18, Metric.COSINE]

    with pytest.raises(ValueError) as exc_info:
        engine.register_collection(E)
    assert "must have exactly one vector field" in str(exc_info)


def test_get_collections_vector_field_name_dim_and_metric(engine):
    class C(Collection):
        vector: Vector[128, Metric.COSINE]

    (
        name,
        dim,
        metric,
    ) = engine._get_collections_vector_field_name_dim_and_metric(C)
    assert name == "vector"
    assert dim == 128
    assert metric == Metric.COSINE


@patch.object(PineconeEngine, "_get_index")
def test_insert(mock_get_index, engine):
    class C(Collection):
        field1: str
        vector: Vector[128, Metric.COSINE]

    mock_index = mock_get_index.return_value
    record = C(vector=Vector([1.0] * 128), field1="value1")

    uid = engine.insert(record)

    assert isinstance(uid, str)
    mock_index.upsert.assert_called_once()


@patch.object(PineconeEngine, "_get_index")
def test_delete_by_id(mock_get_index, engine):
    class C(Collection):
        pass

    mock_index = mock_get_index.return_value

    collection = C()
    collection.id = "test_id"
    engine._delete_by_id(C, "test_id")

    mock_index.delete.assert_called_once_with(["test_id"])


@patch.object(PineconeEngine, "_get_index")
def test_query(mock_get_index, engine):
    class C(Collection):
        vector: Vector[128, Metric.COSINE]
        field1: str

    mock_index = mock_get_index.return_value
    mock_index.query.return_value.matches = [
        ScoredVector(
            id="1",
            score=0.9,
            values=[1.0] * 128,
            metadata={"field1": "value1"},
        )
    ]

    filter_set = FilterSet(collection="C", filters=[])
    similarity = Similarity(
        collection="C", field="vector", value=Vector([1.0] * 128)
    )

    with patch.object(engine, "collection_classes", {"c": C}):
        results = engine._query(filter_set, similarity=similarity, limit=10)

    assert len(results) == 1
    assert results[0].id == "1"
    assert results[0].vector == Vector([1.0] * 128)
