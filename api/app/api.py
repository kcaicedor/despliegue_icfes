import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from io import BytesIO
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model_icfes import (
    cargar_modelo,
    preparar_datos,
    obtener_explicabilidad_lineal
)

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
@api_router.post(
    "/predict",
    response_model=schemas.PredictionResults,
    status_code=200
)
async def predict(file: UploadFile = File(...)):

    try:
        # 1. Leer archivo
        contenido = await file.read()

        if not contenido:
            raise HTTPException(
                status_code=400,
                detail="El archivo está vacío."
            )

        # 2. Identificar formato
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

        # 3. Validar registros
        if input_df.empty:
            raise HTTPException(
                status_code=400,
                detail="El archivo no contiene registros para predecir."
            )

        logger.info(
            f"Archivo recibido: {file.filename}. "
            f"Registros: {len(input_df)}"
        )

        # 4. Cargar modelo entrenado
        modelo = cargar_modelo()

        # 5. Preparar datos
        df_modelo = preparar_datos(input_df)

        # 6. Generar predicciones
        predicciones = modelo.predict(df_modelo)

        predicciones = np.asarray(
            predicciones
        ).reshape(-1).tolist()

        # 7. Calcular importancia global y contribuciones
        importancia_global, contribuciones_individuales = (
            obtener_explicabilidad_lineal(
                modelo,
                df_modelo,
                top_n=5
            )
        )

        # 8. Validar cantidad de resultados
        if len(predicciones) != len(df_modelo):
            raise ValueError(
                "La cantidad de predicciones no coincide "
                "con la cantidad de registros."
            )

        if len(contribuciones_individuales) != len(df_modelo):
            raise ValueError(
                "La cantidad de contribuciones no coincide "
                "con la cantidad de registros."
            )

        logger.info(
            "Predicciones y explicabilidad calculadas correctamente."
        )

        # 9. Construir respuesta
        return {
            "errors": None,
            "version": model_version,
            "predictions": [
                float(valor) for valor in predicciones
            ],
            "importancia_global": importancia_global,
            "contribuciones_individuales": contribuciones_individuales
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            f"Error realizando predicciones: {str(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Error procesando el archivo: {str(e)}"
        )