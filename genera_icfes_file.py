
# ============================================================
# GENERACIÓN DEL MODELO ICFES Y PAQUETE .WHL
# Modelo seleccionado: Regresión Lineal
# ============================================================

import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parent

RUTA_DATASET = RUTA_PROYECTO / "Saber11_2022.csv"

NOMBRE_PAQUETE = "model_icfes"
VERSION_PAQUETE = "0.0.1"

NOMBRE_MODELO = "modelo_regresion_lineal.joblib"

FECHA_EXAMEN = pd.Timestamp("2022-09-02")


# ============================================================
# 2. COLUMNAS QUE SE ELIMINAN
# ============================================================

COLUMNAS_BORRAR = [
    "periodo",
    "estu_tipodocumento",
    "estu_consecutivo",
    "cole_codigo_icfes",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion",
    "cole_cod_mcpio_ubicacion",
    "cole_depto_ubicacion",
    "cole_mcpio_ubicacion",
    "estu_cod_depto_presentacion",
    "estu_cod_mcpio_presentacion",
    "estu_cod_reside_depto",
    "estu_cod_reside_mcpio",
    "estu_estudiante",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica",
    "punt_ingles",
    "desemp_ingles"
]


# ============================================================
# 3. PREPARACIÓN DE LOS DATOS
# ============================================================

def preparar_datos(df):

    df = df.copy()

    # Eliminar columnas innecesarias
    df = df.drop(
        columns=COLUMNAS_BORRAR,
        errors="ignore"
    )

    # Eliminar registros sin puntaje válido (solo aplica
    # cuando se procesa el dataset de entrenamiento).
    if "punt_global" in df.columns:
        df = df[df["punt_global"] != 0].copy()

    # Calcular edad a partir de fecha de nacimiento
    if "estu_fechanacimiento" in df.columns:

        df["estu_fechanacimiento"] = pd.to_datetime(
            df["estu_fechanacimiento"],
            format="%d/%m/%Y",
            errors="coerce"
        )

        df["edad"] = (
            FECHA_EXAMEN.year
            - df["estu_fechanacimiento"].dt.year
            - (
                (
                    df["estu_fechanacimiento"].dt.month
                    > FECHA_EXAMEN.month
                )
                |
                (
                    (
                        df["estu_fechanacimiento"].dt.month
                        == FECHA_EXAMEN.month
                    )
                    &
                    (
                        df["estu_fechanacimiento"].dt.day
                        > FECHA_EXAMEN.day
                    )
                )
            ).astype(int)
        )

        df = df.drop(
            columns=["estu_fechanacimiento"]
        )

    return df


# ============================================================
# 4. CARGA DEL DATASET
# ============================================================

def cargar_dataset():

    print("\n" + "=" * 60)
    print("CARGANDO DATASET SABER 11")
    print("=" * 60)

    if not RUTA_DATASET.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {RUTA_DATASET}"
        )

    df = pd.read_csv(RUTA_DATASET)

    print(f"Registros originales: {len(df):,}")

    # Aplicar preparación de datos
    df = preparar_datos(df)

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Eliminar registros sin variable objetivo
    df = df.dropna(subset=["punt_global"])

    print(f"Registros finales: {len(df):,}")
    print(f"Columnas disponibles: {df.shape[1]}")

    return df


# ============================================================
# 5. CONSTRUCCIÓN DEL PREPROCESADOR
# ============================================================

def crear_preprocesador(X):

    numericas = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categoricas = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numericas",
                Pipeline([
                    (
                        "imputacion",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "escalado",
                        StandardScaler()
                    )
                ]),
                numericas
            ),

            (
                "categoricas",
                Pipeline([
                    (
                        "imputacion",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Desconocido"
                        )
                    ),
                    (
                        "encoding",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categoricas
            )
        ]
    )

    return preprocessor


# ============================================================
# 6. ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def entrenar_modelo(df):

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO: REGRESIÓN LINEAL")
    print("=" * 60)

    X = df.drop(columns=["punt_global"])
    y = df["punt_global"]

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42
    )

    print(f"Datos de entrenamiento: {len(X_train):,}")
    print(f"Datos de prueba: {len(X_test):,}")

    # Crear preprocesador utilizando únicamente
    # los datos de entrenamiento
    preprocessor = crear_preprocesador(X_train)

    # Regresión Lineal con parámetros por defecto
    modelo_evaluacion = Pipeline([
        ("preprocesamiento", preprocessor),
        ("modelo", LinearRegression())
    ])

    # Entrenar
    modelo_evaluacion.fit(
        X_train,
        y_train
    )

    # Predicciones de prueba
    predicciones = modelo_evaluacion.predict(X_test)

    # Métricas
    mse = mean_squared_error(
        y_test,
        predicciones
    )

    mae = mean_absolute_error(
        y_test,
        predicciones
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predicciones
    )

    print("\nRESULTADOS DEL MODELO")
    print("-" * 40)
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    metricas = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    # ========================================================
    # 7. ENTRENAMIENTO FINAL CON TODOS LOS DATOS
    # ========================================================

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINAL CON TODOS LOS REGISTROS")
    print("=" * 60)

    preprocessor_final = crear_preprocesador(X)

    modelo_final = Pipeline([
        ("preprocesamiento", preprocessor_final),
        ("modelo", LinearRegression())
    ])

    modelo_final.fit(X, y)

    print("Modelo final entrenado correctamente.")

    return modelo_final, metricas


# ============================================================
# 8. CÓDIGO PREDICTIVO QUE QUEDARÁ DENTRO DEL PAQUETE
# ============================================================

CODIGO_PREDICTOR = r'''
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
import numpy as np


FECHA_EXAMEN = pd.Timestamp("2022-09-02")

COLUMNAS_BORRAR = [
    "periodo",
    "estu_tipodocumento",
    "estu_consecutivo",
    "cole_codigo_icfes",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion",
    "cole_cod_mcpio_ubicacion",
    "cole_depto_ubicacion",
    "cole_mcpio_ubicacion",
    "estu_cod_depto_presentacion",
    "estu_cod_mcpio_presentacion",
    "estu_cod_reside_depto",
    "estu_cod_reside_mcpio",
    "estu_estudiante",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica",
    "punt_ingles",
    "desemp_ingles"
]


RUTA_MODELO = (
    Path(__file__).parent
    / "artifacts"
    / "modelo_regresion_lineal.joblib"
)


@lru_cache(maxsize=1)
def cargar_modelo():

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    return joblib.load(RUTA_MODELO)


def preparar_datos(datos):

    df = pd.DataFrame(datos).copy()

    # Eliminar columnas que no utiliza el modelo
    df = df.drop(
        columns=COLUMNAS_BORRAR,
        errors="ignore"
    )

    # Si viene el puntaje real, no debe utilizarse
    # como variable de entrada.
    df = df.drop(
        columns=["punt_global"],
        errors="ignore"
    )

    # Calcular edad
    if "estu_fechanacimiento" in df.columns:

        df["estu_fechanacimiento"] = pd.to_datetime(
            df["estu_fechanacimiento"],
            format="%d/%m/%Y",
            errors="coerce"
        )

        df["edad"] = (
            FECHA_EXAMEN.year
            - df["estu_fechanacimiento"].dt.year
            - (
                (
                    df["estu_fechanacimiento"].dt.month
                    > FECHA_EXAMEN.month
                )
                |
                (
                    (
                        df["estu_fechanacimiento"].dt.month
                        == FECHA_EXAMEN.month
                    )
                    &
                    (
                        df["estu_fechanacimiento"].dt.day
                        > FECHA_EXAMEN.day
                    )
                )
            ).astype(int)
        )

        df = df.drop(
            columns=["estu_fechanacimiento"]
        )

    return df

def obtener_explicabilidad_lineal(
    pipeline,
    df_inputs: pd.DataFrame,
    top_n: int = 5
):
    # Obtener los componentes del pipeline
    preprocesador = pipeline.named_steps["preprocesamiento"]
    regresor = pipeline.named_steps["modelo"]

    # Transformar los datos
    X_trans = preprocesador.transform(df_inputs)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    X_trans = np.asarray(X_trans)

    # Obtener nombres de variables transformadas
    nombres_trans = preprocesador.get_feature_names_out()

    coefs = np.asarray(regresor.coef_).reshape(-1)

    # Validar dimensiones
    if X_trans.shape[1] != len(coefs):
        raise ValueError(
            "Las dimensiones de los datos transformados "
            "no coinciden con los coeficientes del modelo."
        )

    # Mapear variables transformadas a variables originales
    variables_originales = df_inputs.columns.tolist()
    mapa_origen = []

    for nombre in nombres_trans:

        limpio = nombre.split("__", 1)[-1]

        coincidencias = [
            col for col in variables_originales
            if limpio == col or limpio.startswith(col + "_")
        ]

        origen = (
            max(coincidencias, key=len)
            if coincidencias
            else limpio
        )

        mapa_origen.append(origen)

    # ----------------------------------------
    # 1. IMPORTANCIA GLOBAL
    # ----------------------------------------

    df_global = pd.DataFrame({
        "variable": mapa_origen,
        "peso": np.abs(coefs)
    })

    top_global = (
        df_global.groupby("variable")["peso"].max()
        .sort_values(ascending=False)
        .head(top_n)
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    # ----------------------------------------
    # 2. CONTRIBUCIONES INDIVIDUALES
    # ----------------------------------------

    contribuciones_matriz = X_trans * coefs

    contribuciones_lista = []

    for fila in contribuciones_matriz:

        df_ind = pd.DataFrame({
            "variable": mapa_origen,
            "contribucion": fila
        })

        agrupado = (
            df_ind.groupby("variable")["contribucion"].sum()
        )

        top_ind = agrupado.reindex(
            agrupado.abs().sort_values(ascending=False).index
        ).head(top_n)

        contribuciones_lista.append([
            {
                "variable": var,
                "contribucion": round(float(val), 2)
            }
            for var, val in top_ind.items()
        ])

    return top_global, contribuciones_lista

def predecir(datos):

    modelo = cargar_modelo()

    df = preparar_datos(datos)

    predicciones = modelo.predict(df)

    return predicciones.tolist()
'''


# ============================================================
# 9. GENERAR EL PAQUETE WHEEL EN UNA CARPETA TEMPORAL
# ============================================================

def generar_wheel(modelo):

    print("\n" + "=" * 60)
    print("GENERANDO PAQUETE .WHL")
    print("=" * 60)

    # Crear estructura temporal
    with tempfile.TemporaryDirectory(
        prefix="build_icfes_"
    ) as temp_dir:

        ruta_temp = Path(temp_dir)

        ruta_paquete = (
            ruta_temp
            / NOMBRE_PAQUETE
        )

        ruta_artifacts = (
            ruta_paquete
            / "artifacts"
        )

        ruta_artifacts.mkdir(
            parents=True,
            exist_ok=True
        )

        # Guardar modelo dentro del paquete
        joblib.dump(
            modelo,
            ruta_artifacts / NOMBRE_MODELO
        )

        # Crear archivo predictor.py
        (ruta_paquete / "predictor.py").write_text(
            CODIGO_PREDICTOR,
            encoding="utf-8"
        )

        # Crear __init__.py
        (ruta_paquete / "__init__.py").write_text(
            "from .predictor import cargar_modelo, predecir, preparar_datos, obtener_explicabilidad_lineal\n",
            encoding="utf-8"
        )

        # Crear setup.py temporal
        setup_code = f'''
from setuptools import setup, find_packages

setup(
    name="{NOMBRE_PAQUETE}",
    version="{VERSION_PAQUETE}",
    description="Modelo predictivo Saber 11 - Regresión Lineal",
    packages=find_packages(),
    include_package_data=True,
    package_data={{
        "{NOMBRE_PAQUETE}": ["artifacts/*.joblib"]
    }},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib"
    ],
    python_requires=">=3.10",
    zip_safe=False
)
'''

        (ruta_temp / "setup.py").write_text(
            setup_code,
            encoding="utf-8"
        )

        # Construir el Wheel directamente en la ruta
        # donde se encuentra este script
        ruta_build = ruta_temp / "build"

        comando = [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(RUTA_PROYECTO),
            "--bdist-dir",
            str(ruta_build)
        ]

        resultado = subprocess.run(
            comando,
            cwd=ruta_temp,
            capture_output=True,
            text=True
        )

        if resultado.returncode != 0:

            print(resultado.stdout)
            print(resultado.stderr)

            raise RuntimeError(
                "No fue posible construir el paquete .whl"
            )

        print(resultado.stdout)

    ruta_wheel = (
        RUTA_PROYECTO
        / f"{NOMBRE_PAQUETE}-{VERSION_PAQUETE}-py3-none-any.whl"
    )

    if not ruta_wheel.exists():
        raise FileNotFoundError(
            "El archivo .whl no fue generado."
        )

    print("\nPAQUETE GENERADO CORRECTAMENTE")
    print(f"Ubicación: {ruta_wheel}")

    return ruta_wheel


# ============================================================
# 10. EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("PROYECTO ICFES - GENERADOR DE MODELO")
    print("MODELO: REGRESIÓN LINEAL")
    print("=" * 60)

    # Verificar herramientas para construir Wheel
    try:
        import setuptools
        import wheel
    except ImportError as e:
        raise RuntimeError(
            "Faltan dependencias de construcción. "
            "Ejecuta: py -m pip install setuptools wheel"
        ) from e

    # 1. Cargar datos
    df = cargar_dataset()

    # 2. Entrenar y evaluar
    modelo_final, metricas = entrenar_modelo(df)

    # 3. Generar paquete
    ruta_wheel = generar_wheel(modelo_final)

    # 4. Resumen
    print("\n" + "=" * 60)
    print("PROCESO FINALIZADO")
    print("=" * 60)

    print("\nModelo: Regresión Lineal")
    print(f"Versión del paquete: {VERSION_PAQUETE}")

    print("\nMétricas de evaluación:")

    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")

    print("\nArchivo generado:")
    print(ruta_wheel)

    print("\n¡Modelo y paquete generados exitosamente!")


if __name__ == "__main__":
    main()
