import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from io import BytesIO
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model_icfes import predecir

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()

model_version = "0.0.1"

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.model_dump()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(file: UploadFile = File(...)):

    try:
        # 1. Leer el archivo recibido
        contenido = await file.read()

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="El archivo está vacío."
            )

        # 2. Identificar y leer el formato
        nombre_archivo = (file.filename or "").lower()

        if nombre_archivo.endswith(".csv"):
            input_df = pd.read_csv(BytesIO(contenido))

        elif nombre_archivo.endswith((".xlsx", ".xls")):
            input_df = pd.read_excel(BytesIO(contenido))

        else:
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Utilice CSV o Excel."
            )

        # 3. Validar que existan registros
        if input_df.empty:
            raise HTTPException(
                status_code=400,
                detail="El archivo no contiene registros para predecir."
            )

        logger.info(
            f"Archivo recibido: {file.filename}. "
            f"Registros: {len(input_df)}"
        )

        # 4. Ejecutar el modelo empaquetado
        predicciones = predecir(input_df)

        # 5. Convertir las predicciones a una lista
        if isinstance(predicciones, np.ndarray):
            predicciones = predicciones.tolist()
        elif isinstance(predicciones, pd.Series):
            predicciones = predicciones.tolist()
        else:
            predicciones = list(predicciones)

        # 6. Construir respuesta
        return {
            "errors": None,
            "version": model_version,
            "predictions": [float(valor) for valor in predicciones]
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error realizando predicciones: {str(e)}")

        raise HTTPException(
            status_code=400,
            detail=f"Error procesando el archivo: {str(e)}"
        )