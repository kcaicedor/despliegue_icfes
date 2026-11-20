from typing import Any, List, Optional

from pydantic import BaseModel

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any] = None
    version: str
    predictions: Optional[List[float]] = None