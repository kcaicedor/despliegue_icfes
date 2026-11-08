import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#pd.set_option('print.max_columns', None)

df = pd.read_csv("Saber11_2022.csv")
#df.head(5)

columnas_borrar1 = [
    "periodo", "estu_tipodocumento", "estu_consecutivo", "cole_codigo_icfes",
    "cole_cod_dane_establecimiento", "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion", "cole_cod_mcpio_ubicacion",
    "cole_depto_ubicacion", "cole_mcpio_ubicacion",
    "estu_cod_depto_presentacion", "estu_cod_mcpio_presentacion",
    "estu_cod_reside_depto", "estu_cod_reside_mcpio",
    "estu_estudiante", "punt_matematicas", "punt_sociales_ciudadanas", 
    "punt_c_naturales", "punt_lectura_critica", "punt_ingles", "desemp_ingles"
]

#depuracion definida en el experimento del notebook
df = df.drop(columns=columnas_borrar1)
df = df[df["punt_global"] != 0].copy()
df = df.drop_duplicates()
print("Dimensiones del DataFrame:", df.shape)

# Cálculo de edad
df["estu_fechanacimiento"] = pd.to_datetime(df["estu_fechanacimiento"], format="%d/%m/%Y", errors="coerce")
fecha_examen = pd.Timestamp("2022-09-02")
df["edad"] = (
    fecha_examen.year - df["estu_fechanacimiento"].dt.year - (
        (df["estu_fechanacimiento"].dt.month > fecha_examen.month) |
        ((df["estu_fechanacimiento"].dt.month == fecha_examen.month) & (df["estu_fechanacimiento"].dt.day > fecha_examen.day))
    ).astype(int)
)
df = df.drop(columns=["estu_fechanacimiento"])

# Separación de variables
X = df.drop(columns=["punt_global"])
y = df["punt_global"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ==========================================
# CONFIGURACIÓN DEL BASELINE
# ==========================================
X_train_base = X_train.copy()
X_test_base = X_test.copy()

numericas_base = X_train_base.select_dtypes(include=["number"]).columns.tolist()
categoricas_base = X_train_base.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

preprocessor_base = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler())
    ]), numericas_base),
    ("categoricas", Pipeline([
        ("imputacion", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("encoding", OneHotEncoder(handle_unknown="ignore"))
    ]), categoricas_base)
])

# ==========================================
# FEATURE ENGINEERING
# ==========================================
#1: Ordinal Encoding
ordenes = {
    "fami_estratovivienda": ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"],
    "fami_cuartoshogar": ["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis o mas"],
    "fami_personashogar": ["1 a 2", "3 a 4", "5 a 6", "7 a 8", "9 o más"],
    "fami_educacionmadre": ["Ninguno", "Primaria incompleta", "Primaria completa", "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa", "Técnica o tecnológica incompleta", "Técnica o tecnológica completa", "Educación profesional incompleta", "Educación profesional completa", "Postgrado"],
    "fami_educacionpadre": ["Ninguno", "Primaria incompleta", "Primaria completa", "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa", "Técnica o tecnológica incompleta", "Técnica o tecnológica completa", "Educación profesional incompleta", "Educación profesional completa", "Postgrado"]
}

ordinal_encoder = OrdinalEncoder(categories=[ordenes[col] for col in ordenes], handle_unknown="use_encoded_value", unknown_value=np.nan)
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()
X_train_fe[list(ordenes.keys())] = ordinal_encoder.fit_transform(X_train_fe[list(ordenes.keys())])
X_test_fe[list(ordenes.keys())] = ordinal_encoder.transform(X_test_fe[list(ordenes.keys())])

#2: Eliminar variables redundantes
variables_eliminar = ["cole_nombre_sede", "estu_mcpio_presentacion", "estu_depto_presentacion"]
X_train_fe2 = X_train_fe.drop(columns=variables_eliminar)
X_test_fe2 = X_test_fe.drop(columns=variables_eliminar)

#3: Frequency Encoding para alta cardinalidad
variables_frecuencia = ["cole_nombre_establecimiento", "estu_mcpio_reside", "estu_depto_reside"]
X_train_fe3 = X_train_fe2.copy()
X_test_fe3 = X_test_fe2.copy()
frecuencias = {}

for col in variables_frecuencia:
    frecuencias[col] = X_train_fe3[col].value_counts(normalize=True)
    X_train_fe3[col + "_FE"] = X_train_fe3[col].map(frecuencias[col])
    X_test_fe3[col + "_FE"] = X_test_fe3[col].map(frecuencias[col]).fillna(0)
    X_train_fe3.drop(columns=[col], inplace=True)
    X_test_fe3.drop(columns=[col], inplace=True)

#4: Indicadores de valores faltantes
variables_faltantes = ["cole_bilingue", "cole_caracter", "fami_estratovivienda", "fami_tieneinternet", "fami_educacionpadre", "fami_educacionmadre", "fami_tieneautomovil", "fami_cuartoshogar", "fami_tienelavadora", "fami_tienecomputador", "fami_personashogar"]
X_train_fe4 = X_train_fe3.copy()
X_test_fe4 = X_test_fe3.copy()

for col in variables_faltantes:
    X_train_fe4[col + "_missing"] = X_train_fe4[col].isna().astype(int)
    X_test_fe4[col + "_missing"] = X_test_fe4[col].isna().astype(int)

#5: Preprocesador para el dataset con Feature Engineering
# Agrupamos las variables resultantes para inyectarlas al Pipeline
numericas_fe = X_train_fe4.select_dtypes(include=["number"]).columns.tolist()
categoricas_fe = X_train_fe4.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

indicadores_missing = [col for col in numericas_fe if col.endswith("_missing")]
numericas_continuas = [col for col in numericas_fe if not col.endswith("_missing")]

preprocessor_fe = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler())
    ]), numericas_continuas),
    ("indicadores", "passthrough", indicadores_missing),
    ("categoricas", Pipeline([
        ("imputacion", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("encoding", OneHotEncoder(handle_unknown="ignore"))
    ]), categoricas_fe)
])

# ==========================================
# CONFIGURACIÓN DE EXPERIMENTOS EN MLFLOW
# ==========================================
def configure_experiment(name_experiment, modelos, option):
    experiment = mlflow.set_experiment(name_experiment)

    if option == 0:
        rf = modelos["Baseline"]
        nombre_modelo = "baseline-model"
        X_train_model, y_train_model = X_train_base, y_train
        X_test_model, y_test_model = X_test_base, y_test

    elif option == 1:
        rf = modelos["Regresion_Lineal"]
        nombre_modelo = "regresion-lineal-model"
        X_train_model, y_train_model = X_train_base, y_train
        X_test_model, y_test_model = X_test_base, y_test

    elif option == 2:
        rf = modelos["Random_Forest"]
        nombre_modelo = "random-forest-model"
        X_train_model, y_train_model = X_train_base, y_train
        X_test_model, y_test_model = X_test_base, y_test

    elif option == 3:
        rf = modelos["Gradient_Boosting"]
        nombre_modelo = "gradient-boosting-model"
        X_train_model, y_train_model = X_train_base, y_train
        X_test_model, y_test_model = X_test_base, y_test
        
    elif option == 4:
        rf = modelos["Random_Forest_FE"]
        nombre_modelo = "random-forest-feature-eng"
        # Usamos los datasets con todo el feature engineering manual aplicado
        X_train_model, y_train_model = X_train_fe4, y_train
        X_test_model, y_test_model = X_test_fe4, y_test

    else:
        raise ValueError("Opción no válida")

    # El baseline ya contiene su propio preprocesamiento.
    # Los demás modelos necesitan que se lo agreguemos.
    if option != 0:
        preprocesador_actual = preprocessor_base if option != 4 else preprocessor_fe
        rf = Pipeline([
            ("preprocesamiento", clone(preprocesador_actual)),
            ("modelo", rf)
        ])

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=nombre_modelo
    ):
        mlflow.set_tag("modelo", nombre_modelo)

        rf.fit(X_train_model, y_train_model)
        predictions = rf.predict(X_test_model)

        mse = mean_squared_error(y_test_model, predictions)
        mae = mean_absolute_error(y_test_model, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_model, predictions)

        mlflow.log_metrics({
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        # Registra los parámetros del estimador, no los del preprocesador.
        if option != 0:
            params = rf.named_steps["modelo"].get_params()
            mlflow.log_params({f"modelo_{k}": str(v) for k, v in params.items()})

        mlflow.sklearn.log_model(sk_model=rf, artifact_path="modelo")

        print(f"Modelo: {nombre_modelo}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

# ==========================================
# DICCIONARIO DE MODELOS PARA EJECUCIÓN
# ==========================================
modelos = {
    "Baseline": Pipeline([("preprocesamiento", preprocessor_base), ("regresion", LinearRegression())]),
    "Regresion_Lineal": LinearRegression(fit_intercept=True),
    "Random_Forest": RandomForestRegressor(n_estimators=200, max_depth=6, max_features=4, random_state=42),
    "Gradient_Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
    # modelo para el dataset con Feature Engineering
    "Random_Forest_FE": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
}

# Ejecutar el experimento con la opción del 0 al 4
configure_experiment("Proyecto_ICFES", modelos, 4)