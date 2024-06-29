from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

Operation = Literal["eq", "lte", "gte"]


class Filter(BaseModel):
    field: str
    operation: Operation
    value: Any


class NumPyEngine:
    def __init__(
        self, dim: int, name: str = None
    ):  # maybe add option to the init for ANN algo
        self.data = np.empty((0, dim))

    def query(
        self, vector: np.ndarray, top_k: int, filters: list[Filter] = None
    ):
        pass
