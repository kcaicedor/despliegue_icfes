from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone


from scipy.stats import (
    pearsonr,
    spearmanr,
    f_oneway
)

from sklearn.feature_selection import mutual_info_regression

#pd.set_option('print.max_columns', None)

df = pd.read_csv("Saber11_2022.csv")
#df.head(5)

columnas_borrar1 = [
    "periodo","estu_tipodocumento","estu_consecutivo","cole_codigo_icfes",
    "cole_cod_dane_establecimiento",
    "cole_cod_dane_sede",
    "cole_cod_depto_ubicacion",
    "cole_cod_mcpio_ubicacion",
    "cole_depto_ubicacion", "cole_mcpio_ubicacion",
    "estu_cod_depto_presentacion",
    "estu_cod_mcpio_presentacion",
    "estu_cod_reside_depto",
    "estu_cod_reside_mcpio",
    "estu_estudiante", 'punt_matematicas', 'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_ingles', "desemp_ingles"
]

df = df.drop(columns=columnas_borrar1)
df = df.drop([], axis = 1)
print(df.shape)
df.columns.tolist()

# Porcentaje de valores faltantes
print(
    (df.isnull().mean() * 100)
    .sort_values(ascending=False)
    .round(2)
)

# Distribución del puntaje global
print(df["punt_global"].describe())

# Identificar registros con puntaje cero
print("Puntajes iguales a cero:",
      (df["punt_global"] == 0).sum())

# Identificar registros con fechas inválidas
print("Fechas inválidas:",
      df["estu_fechanacimiento"].isna().sum())

# Revisar posibles edades extremas posteriormente
print(df["estu_fechanacimiento"].describe())

print(
    df[df["punt_global"] == 0].T
)

df = df[df["punt_global"] != 0].copy()

#duplicados
print("Duplicados:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicados restantes:", df.duplicated().sum())
print("Dimensiones del DataFrame:", df.shape)

# cardinalidad
cardinalidad = pd.DataFrame({
    "Cardinalidad": df.nunique(),
    "Porcentaje_unicos": (df.nunique() / len(df) * 100).round(2),
    "Tipo_dato": df.dtypes
})

cardinalidad = cardinalidad.sort_values(
    by="Cardinalidad", ascending=False
)

print(cardinalidad)

col = "cole_nombre_establecimiento"

# Cantidad de establecimientos diferentes
print("Establecimientos únicos:", df[col].nunique())

# Cantidad de registros
print("Total registros:", len(df))

# Frecuencia de los establecimientos
print(df[col].value_counts().head(15))

# Porcentaje de establecimientos únicos respecto al total
print(
    "Cardinalidad:",
    df[col].nunique() / len(df) * 100
)

for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}: {df[col].nunique()} categorías")
    print(df[col].dropna().unique()[:10])

# Convertir fecha de nacimiento
df["estu_fechanacimiento"] = pd.to_datetime(
    df["estu_fechanacimiento"],
    format="%d/%m/%Y",
    errors="coerce"
)

# Fecha del examen
fecha_examen = pd.Timestamp("2022-09-02")

# Calcular edad
df["edad"] = (
    fecha_examen.year
    - df["estu_fechanacimiento"].dt.year
    - (
        (
            df["estu_fechanacimiento"].dt.month > fecha_examen.month
        ) |
        (
            (df["estu_fechanacimiento"].dt.month == fecha_examen.month) &
            (df["estu_fechanacimiento"].dt.day > fecha_examen.day)
        )
    ).astype(int)
)
#elimino fecha nacimiento
df = df.drop(columns=["estu_fechanacimiento"])

print(df["edad"].describe())

print("Edades menores de 10:",
      (df["edad"] < 10).sum())

print("Edades mayores de 20:",
      (df["edad"] > 20).sum())

X = df.drop(columns=["punt_global"])
y = df["punt_global"]
#dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

# dimensiones de los datos
print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)

# Identificar variables numéricas
numericas = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Identificar variables categóricas
categoricas = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numéricas:", numericas)
print("Categóricas:", categoricas)

#separamos x tipo de variable
numeric_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("Variables numéricas:", len(numeric_cols))
print("Variables categóricas:", len(categorical_cols))

missing_analysis = pd.DataFrame({
    "missing_count": X_train.isna().sum(),
    "missing_percentage": X_train.isna().mean() * 100
})

missing_analysis = missing_analysis[
    missing_analysis["missing_count"] > 0
].sort_values(
    "missing_percentage",
    ascending=False
)

print(missing_analysis)


if numeric_cols:

    print(
        X_train[numeric_cols].describe().T
    )

numeric_data = X_train[numeric_cols].copy()

numeric_data["punt_global"] = y_train

pearson_corr = numeric_data.corr(
    method="pearson"
)

target_pearson = (
    pearson_corr["punt_global"]
    .drop("punt_global")
    .sort_values(
        key=abs,
        ascending=False
    )
)

spearman_corr = numeric_data.corr(
    method="spearman"
)

target_spearman = (
    spearman_corr["punt_global"]
    .drop("punt_global")
    .sort_values(
        key=abs,
        ascending=False
    )
)

correlation_comparison = pd.DataFrame({
    "pearson": target_pearson,
    "spearman": target_spearman
})

print(
    correlation_comparison.sort_values(
        "spearman",
        key=abs,
        ascending=False
    )
)

anova_results = []

for col in categorical_cols:

    temp = pd.DataFrame({
        "feature": X_train[col],
        "target": y_train
    }).dropna()

    groups = [
        group["target"].values
        for _, group in temp.groupby("feature")
    ]

    if len(groups) < 2:
        continue

    if any(len(group) < 2 for group in groups):
        continue

    statistic, p_value = f_oneway(*groups)

    anova_results.append({
        "variable": col,
        "categories": len(groups),
        "f_statistic": statistic,
        "p_value": p_value
    })

anova_df = pd.DataFrame(anova_results)

anova_df = anova_df.sort_values(
    "f_statistic",
    ascending=False
)

print(anova_df)

X_mi = X_train[categorical_cols].copy()

for col in categorical_cols:

    X_mi[col] = (
        X_mi[col]
        .astype("category")
        .cat.codes
    )

    # -1 representa valores faltantes
    X_mi[col] = X_mi[col].replace(-1, np.nan)

# Para evitar errores por valores faltantes, imputamos temporalmente la moda calculada en Train:



X_mi = X_mi.fillna(
    X_mi.mode().iloc[0]
)

mi_scores = mutual_info_regression(
    X_mi,
    y_train,
    discrete_features=True,
    random_state=42
)

mi_results = pd.DataFrame({
    "variable": categorical_cols,
    "mutual_information": mi_scores
})

mi_results = mi_results.sort_values(
    "mutual_information",
    ascending=False
)

print(mi_results)

association_summary = mi_results.merge(
    anova_df[["variable", "f_statistic", "p_value"]],
    on="variable",
    how="left"
)

print(
    association_summary.sort_values(
        "mutual_information",
        ascending=False
    )
)


X_train.head(5)

from sklearn.preprocessing import OrdinalEncoder

# Variables ordinales
ordenes = {
    "fami_estratovivienda": [
        "Estrato 1",
        "Estrato 2",
        "Estrato 3",
        "Estrato 4",
        "Estrato 5",
        "Estrato 6"
    ],

    "fami_cuartoshogar": [
        "Uno",
        "Dos",
        "Tres",
        "Cuatro",
        "Cinco",
        "Seis o mas"
    ],

    "fami_personashogar": [
        "1 a 2",
        "3 a 4",
        "5 a 6",
        "7 a 8",
        "9 o más"
    ],

    "fami_educacionmadre": [
        "Ninguno",
        "Primaria incompleta",
        "Primaria completa",
        "Secundaria (Bachillerato) incompleta",
        "Secundaria (Bachillerato) completa",
        "Técnica o tecnológica incompleta",
        "Técnica o tecnológica completa",
        "Educación profesional incompleta",
        "Educación profesional completa",
        "Postgrado"
    ],

    "fami_educacionpadre": [
        "Ninguno",
        "Primaria incompleta",
        "Primaria completa",
        "Secundaria (Bachillerato) incompleta",
        "Secundaria (Bachillerato) completa",
        "Técnica o tecnológica incompleta",
        "Técnica o tecnológica completa",
        "Educación profesional incompleta",
        "Educación profesional completa",
        "Postgrado"
    ]
}

# aplicar encoding
X_train_fe = X_train.copy()

ordinal_encoder = OrdinalEncoder(
    categories=[ordenes[col] for col in ordenes],
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)

X_train_fe[list(ordenes.keys())] = ordinal_encoder.fit_transform(
    X_train_fe[list(ordenes.keys())]
)

# transformar test con el mismo encoder
X_test_fe = X_test.copy()

X_test_fe[list(ordenes.keys())] = ordinal_encoder.transform(
    X_test_fe[list(ordenes.keys())]
)

print(X_train_fe[list(ordenes.keys())].head())

print(X_train_fe[list(ordenes.keys())].dtypes)


variables_geograficas = [
    "cole_nombre_establecimiento",
    "cole_nombre_sede",
    "estu_mcpio_presentacion",
    "estu_mcpio_reside",
    "estu_depto_presentacion",
    "estu_depto_reside"
]

cardinalidad = pd.DataFrame({
    "variable": variables_geograficas,
    "categorias": [
        X_train[col].nunique() for col in variables_geograficas
    ],
    "valores_nulos": [
        X_train[col].isna().sum() for col in variables_geograficas
    ]
})

print(cardinalidad.sort_values("categorias", ascending=False))

correspondencia = (
    X_train.groupby("cole_nombre_establecimiento")[
        "cole_nombre_sede"
    ].nunique()
)

print(
    "Establecimientos con una sola sede:",
    (correspondencia == 1).mean() * 100,
    "%"
)

print(
    "Coincidencia departamento:",
    (
        X_train["estu_depto_reside"] ==
        X_train["estu_depto_presentacion"]
    ).mean() * 100
)

print(
    "Coincidencia municipio:",
    (
        X_train["estu_mcpio_reside"] ==
        X_train["estu_mcpio_presentacion"]
    ).mean() * 100
)

# Estudiantes que presentan el examen
# en un municipio diferente al de residencia

municipios_diferentes = X_train[
    X_train["estu_mcpio_reside"] !=
    X_train["estu_mcpio_presentacion"]
]

#print("Registros diferentes:", len(municipios_diferentes))

#colegios_multisede = sedes_por_colegio[sedes_por_colegio > 1]

#print("Establecimientos con varias sedes:", len(colegios_multisede))

# Copia del dataset transformado
X_train_fe2 = X_train_fe.copy()

# Variables redundantes
variables_eliminar = [
    "cole_nombre_sede",
    "estu_mcpio_presentacion",
    "estu_depto_presentacion"
]

# Eliminar variables
X_train_fe2 = X_train_fe2.drop(
    columns=variables_eliminar
)

print("Dimensiones originales:", X_train.shape)
print("Dimensiones después:", X_train_fe2.shape)

# Copia del dataset transformado
X_train_fe2 = X_train_fe.copy()

# Variables redundantes
variables_eliminar = [
    "cole_nombre_sede",
    "estu_mcpio_presentacion",
    "estu_depto_presentacion"
]

# Eliminar variables
X_train_fe2 = X_train_fe2.drop(
    columns=variables_eliminar
)

print("Dimensiones originales:", X_train.shape)
print("Dimensiones después:", X_train_fe2.shape)

# usamos train y test original 
X_train_base = X_train.copy()
X_test_base = X_test.copy()

y_train_base = y_train.copy()
y_test_base = y_test.copy()

numericas = X_train_base.select_dtypes(
    include=["number"]
).columns.tolist()

categoricas = X_train_base.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pipeline numérico
pipeline_numerico = Pipeline([
    ("imputacion", SimpleImputer(strategy="median")),
    ("escalado", StandardScaler())
])

# Pipeline categórico
pipeline_categorico = Pipeline([
    ("imputacion", SimpleImputer(
        strategy="constant",
        fill_value="Desconocido"
    )),
    ("encoding", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

# Transformador general
preprocessor_base = ColumnTransformer([
    ("numericas", pipeline_numerico, numericas),
    ("categoricas", pipeline_categorico, categoricas)
])

from sklearn.linear_model import LinearRegression

modelo_base = Pipeline([
    ("preprocesamiento", preprocessor_base),
    ("regresion", LinearRegression())
])

modelo_base.fit(X_train_base, y_train_base)

y_pred_base = modelo_base.predict(X_test_base)


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae_base = mean_absolute_error(y_test_base, y_pred_base)

rmse_base = np.sqrt(
    mean_squared_error(y_test_base, y_pred_base)
)

r2_base = r2_score(y_test_base, y_pred_base)

print("RESULTADOS MODELO BASELINE")
print("---------------------------")
print(f"MAE:  {mae_base:.4f}")
print(f"RMSE: {rmse_base:.4f}")
print(f"R²:   {r2_base:.4f}")

# guardar 
resultados_modelos = pd.DataFrame([
    {
        "Modelo": "Regresion Lineal Baseline",
        "MAE": mae_base,
        "RMSE": rmse_base,
        "R2": r2_base
    }
])

print(resultados_modelos)


#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#mlflow.set_tracking_uri("http://127.0.0.1:8050")

from sklearn.base import clone
from sklearn.pipeline import Pipeline

def configure_experiment(name_experiment, modelos, option):
    experiment = mlflow.set_experiment(name_experiment)

    if option == 0:
        rf = modelos["Baseline"]
        nombre_modelo = "baseline-model"
        X_train_model, y_train_model = X_train_base, y_train_base
        X_test_model, y_test_model = X_test_base, y_test_base

    elif option == 1:
        rf = modelos["Regresion_Lineal"]
        nombre_modelo = "regresion-lineal-model"
        X_train_model, y_train_model = X_train, y_train
        X_test_model, y_test_model = X_test, y_test

    elif option == 2:
        rf = modelos["Random_Forest"]
        nombre_modelo = "random-forest-model"
        X_train_model, y_train_model = X_train, y_train
        X_test_model, y_test_model = X_test, y_test

    elif option == 3:
        rf = modelos["Gradient_Boosting"]
        nombre_modelo = "gradient-boosting-model"
        X_train_model, y_train_model = X_train, y_train
        X_test_model, y_test_model = X_test, y_test

    else:
        raise ValueError("Opción no válida")

    # El baseline ya contiene su propio preprocesamiento.
    # Los demás modelos necesitan que se lo agreguemos.
    if option != 0:
        rf = Pipeline([
            ("preprocesamiento", clone(preprocessor_base)),
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
            mlflow.log_params({
                f"modelo_{key}": str(value)
                for key, value in params.items()
            })

        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="modelo"
        )

        print(f"Modelo: {nombre_modelo}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

modelos = {
    "Baseline": modelo_base,
    "Regresion_Lineal": LinearRegression(
        
    ),
    "Random_Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        max_features=4,
        random_state=42
    ),
    "Gradient_Boosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
}

#ejecutar el baseline
configure_experiment("Proyecto_ICFES", modelos, 0)
