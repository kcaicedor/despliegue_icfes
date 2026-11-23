from typing import Any
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Any = None
    version: str
    predictions: list[float]
    importancia_global: list[dict[str, Any]]
    contribuciones_individuales: list[list[dict[str, Any]]]