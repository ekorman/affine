# affine

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/ekorman/7fbb57e6d6a2c8b69617ddf141043b98/raw/affine-coverage.json)

Affine is a Python library for providing a uniform and structured interface to various backing vector databases and approximate nearest neighbor libraries. It allows simple dataclass-like objects to describe collections together with a high-level query syntax for doing filtered vector search.

For vector databases, it currently supports:

- qdrant
- weaviate
- pinecone

For local mode, the following approximate nearest neighbor libraries are supported:

- FAISS
- annoy
- pynndescent
- scikit-learn KDTree
- naive/NumPy

Note: this project is very similar to [vectordb-orm](https://github.com/piercefreeman/vectordb-orm), which looks to be no longer maintained.

## Installation

```bash
pip install affine
# or `pip install affine[qdrant]` for qdrant support
# `pip install affine[weaviate]` for weaviate support
# `pip install affine[pinecone]` for pinecone support
```

## Basic Usage

```python
from affine import Collection, Vector, Filter, Query

# Define a collection
class MyCollection(Collection):
    vec: Vector[3] # declare a 3-dimensional vector

    # support for additional fields for filtering
    a: int
    b: str

db = LocalEngine()

# Insert vectors
db.insert(MyCollection(vec=[0.1, 0.0, -0.5], a=1, b="foo"))
db.insert(MyCollection(vec=[1.3, 2.1, 3.6], a=2, b="bar"))
db.insert(MyCollection(vec=[-0.1, 0.2, 0.3], a=3, b="foo"))

# Query vectors
result: list[MyCollection] = (
    db.query(MyCollection)
    .filter(MyCollection.b == "foo")
    .similarity([2.8, 1.8, -4.5])
    .limit(1)
)
```

## Engines

A fundamental notion of _affine_ are `Engine` classes. All such classes conform to the same API for interchangeabillity (with the exception of a few engine-specific restrictions which are be mentioned below). There are two broad types of engines

1. `LocalEngine`: this does nearest neighbor search on the executing machine, and supports a variety of libraries for the backing nearest neighborsearch (these are called the _backend_ of the local engine).

2. Vector database engines: these are engines that connect to a vector database service, such as QDrant, Weaviate, or Pinecone.

### Vector Databases

The currently supported vector databases are:

| Database | Class                          | Constructor arguments                                                                                                                                                                                                                                                                                                                          | Notes                                                                                                    |
| -------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Qdrant   | `affine.engine.QdrantEngine`   | `host: str` hostname to use<br><br>`port: int` port to use                                                                                                                                                                                                                                                                                     | -                                                                                                        |
| Weaviate | `affine.engine.WeaviateEngine` | `host: str` hostname to use<br><br>`port: int` port to use                                                                                                                                                                                                                                                                                     | -                                                                                                        |
| Pinecone | `affine.engine.PineconeEngine` | `api_key: Union[str, None]` pinecone API key. if not provided, it will be read from the environment variable PINECONE_API_KEY.<br><br>`spec: Union[ServerlessSpec, PodSpec, None]` the PodSpec or ServerlessSpec object. If not provided, a`ServerlessSpec` will be created from the environment variables PINECONE_CLOUD and PINECONE_REGION. | the Pinecone engine has the restriction that every collection must contain exactly one vector attribute. |

### Approximate Nearest Neighbor Libraries

The `LocalEngine` class provides an interface for doing nearest neighbor search on the executing machine, supporting a variety of libraries for the backing nearest neighborsearch. Which one is specified by the `backend` argument to the constructor. For example, to use `annoy`:

```python
from affine.engine.local import LocalEngine, AnnoyBackend

db = LocalEngine(backend=AnnoyBackend(n_tress=10))
```

The options and settings for the various supported backends are as follows:

| Library             | Class                                    | Constructor arguments                                                    | Notes |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------------ | ----- |
| naive/numpy         | `affine.engine.local.NumPyBackend`       | -                                                                        | -     |
| scikit-learn KDTree | `affine.engine.local.KDTreeBackend`      | keyword arguments that get passed directly to `sklearn.neighbors.KDTree` | -     |
| annoy               | `affine.engine.local.AnnoyBackend`       | `n_trees: int` number of trees to use<br>`n_jobs: int` defaults to -1    | -     |
| FAISS               | `affine.engine.local.FAISSBackend`       | `index_factory_str: str`                                                 | -     |
| PyNNDescent         | `affine.engine.local.PyNNDescentBackend` | keyword arguments that get passed directly to `pynndescent.NNDescent`    | -     |
