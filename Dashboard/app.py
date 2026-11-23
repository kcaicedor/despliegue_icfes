
# IMPORTACION LIBRERIAS
import os
import io
import base64
import requests
import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, html, dcc, Input, Output, State, no_update


# ============================================================
# 1. CONFIGURACIÓN DE CONEXIÓN A LA API
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001/api/v1/predict"
)

MAE_MODELO = 31.2913
RMSE_MODELO = 38.6765
R2_MODELO = 0.3745

IMPORTANCIA_INICIAL = [
    {"variable": "Educación de la madre", "peso": 0.0},
    {"variable": "Estrato socioeconómico", "peso": 0.0},
    {"variable": "Educación del padre", "peso": 0.0},
    {"variable": "Acceso a Internet", "peso": 0.0},
    {"variable": "Disponibilidad de computador", "peso": 0.0},
    {"variable": "Jornada escolar", "peso": 0.0}
]


# ============================================================
# 2. DATOS INICIALES
# ============================================================

def generar_datos_iniciales():
    """Genera datos demostrativos mientras se carga un archivo."""

    np.random.seed(42)

    puntajes = np.clip(
        np.random.normal(loc=255, scale=45, size=200),
        0,
        500
    )

    df = pd.DataFrame({
        "ID estudiante": [
            f"Estudiante {i:03d}" for i in range(1, 201)
        ],
        "Estrato": np.random.choice(
            [1, 2, 3, 4, 5, 6],
            size=200,
            p=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04]
        ),
        "Internet": np.random.choice(
            ["Sí", "No"], size=200, p=[0.61, 0.39]
        ),
        "Computador": np.random.choice(
            ["Sí", "No"], size=200, p=[0.61, 0.39]
        ),
        "Puntaje estimado": np.round(puntajes, 2)
    })

    df["Estado"] = np.where(
        df["Puntaje estimado"] < 300,
        "Requiere acompañamiento",
        "No requiere acompañamiento"
    )

    return df


df_inicial = generar_datos_iniciales()


# ============================================================
# 3. FUNCIONES DE NORMALIZACIÓN Y PROCESAMIENTO
# ============================================================

def normalizar_si_no(serie):
    """Normaliza respuestas de variables binarias."""

    valores = serie.astype("string").str.strip().str.lower()

    equivalencias = {
        "si": "Sí",
        "sí": "Sí",
        "s": "Sí",
        "1": "Sí",
        "true": "Sí",
        "yes": "Sí",
        "no": "No",
        "n": "No",
        "2": "No",
        "false": "No"
    }

    resultado = valores.map(equivalencias)

    return resultado.fillna("No informado")


def preparar_dataframe_tablero(
    df_raw: pd.DataFrame,
    predicciones: list,
    contribuciones: list = None
) -> pd.DataFrame:
    """
    Combina los datos originales con las predicciones de la API
    y prepara las columnas utilizadas por el Dashboard.
    """

    df = df_raw.copy()

    if len(predicciones) != len(df):
        raise ValueError(
            "La cantidad de predicciones no coincide "
            "con la cantidad de registros."
        )

    # Agregar predicciones
    df["Puntaje estimado"] = np.round(
        np.clip(
            pd.to_numeric(predicciones, errors="coerce"),
            0,
            500
        ),
        2
    )

    # Clasificar estudiantes
    df["Estado"] = np.where(
        df["Puntaje estimado"] < 300,
        "Requiere acompañamiento",
        "No requiere acompañamiento"
    )

    # ID de estudiante para el Dashboard
    # No utilizamos identificadores personales del archivo.
    df["ID estudiante"] = [
        f"Estudiante {i:04d}"
        for i in range(1, len(df) + 1)
    ]

    # Estrato
    if "Estrato" not in df.columns:

        if "fami_estratovivienda" in df.columns:
            df["Estrato"] = (
                df["fami_estratovivienda"]
                .astype("string")
                .str.extract(r"(\d+)")[0]
            )

        else:
            df["Estrato"] = 0

    df["Estrato"] = pd.to_numeric(
        df["Estrato"],
        errors="coerce"
    ).fillna(0)

    # Internet
    if "Internet" not in df.columns:

        if "fami_tieneinternet" in df.columns:
            df["Internet"] = normalizar_si_no(
                df["fami_tieneinternet"]
            )
        else:
            df["Internet"] = "No informado"

    else:
        df["Internet"] = normalizar_si_no(df["Internet"])

    # Computador
    if "Computador" not in df.columns:

        if "fami_tienecomputador" in df.columns:
            df["Computador"] = normalizar_si_no(
                df["fami_tienecomputador"]
            )
        else:
            df["Computador"] = "No informado"

    else:
        df["Computador"] = normalizar_si_no(df["Computador"])

    # Guardar las contribuciones individuales recibidas desde la API
    if contribuciones is not None:

        if len(contribuciones) != len(df):
            raise ValueError(
                "La cantidad de contribuciones no coincide "
                "con la cantidad de estudiantes."
            )

        df["_contribuciones_api"] = contribuciones

    else:
        df["_contribuciones_api"] = [None] * len(df)

    return df


# ============================================================
# 4. FUNCIONES PARA LAS TARJETAS
# ============================================================

def crear_tarjeta(titulo, valor, identificador):

    return html.Div([
        html.P(titulo, className="card-title"),
        html.H3(
            valor,
            className="card-value",
            id=f"{identificador}-valor"
        )
    ], className="indicator-card", id=identificador)


# ============================================================
# 5. FUNCIONES DE GRÁFICAS
# ============================================================

def crear_histograma_puntajes(df):

    fig = px.histogram(
        df,
        x="Puntaje estimado",
        nbins=25,
        title="Distribución de puntajes globales estimados",
        labels={
            "Puntaje estimado": "Puntaje global estimado",
            "count": "Cantidad de estudiantes"
        },
        color_discrete_sequence=["#4E79A7"]
    )

    fig.add_vline(
        x=300,
        line_dash="dash",
        line_color="#D9534F",
        line_width=2,
        annotation_text="Umbral: 300",
        annotation_position="top right"
    )

    promedio = df["Puntaje estimado"].mean()

    fig.add_vline(
        x=promedio,
        line_dash="dot",
        line_color="#2E8B57",
        line_width=2,
        annotation_text=f"Promedio: {promedio:.1f}",
        annotation_position="top left"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=16
        ),
        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=22,
            color="#304F70",
            weight="bold"
        ),
        title_x=0.5,
        margin=dict(l=45, r=25, t=65, b=45),
        showlegend=False
    )

    fig.update_xaxes(range=[0, 500], showgrid=False)
    fig.update_yaxes(
        title="Cantidad de estudiantes",
        gridcolor="#E8EAF0"
    )

    return fig


def crear_grafica_acompanamiento(df):

    df_estado = df["Estado"].value_counts().reset_index()
    df_estado.columns = ["Estado", "Cantidad"]

    fig = px.pie(
        df_estado,
        names="Estado",
        values="Cantidad",
        hole=0,
        title="Estudiantes según necesidad de acompañamiento",
        color="Estado",
        color_discrete_map={
            "Requiere acompañamiento": "#4E79A7",
            "No requiere acompañamiento": "#8BCF7A"
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Estudiantes: %{value}<br>"
            "Porcentaje: %{percent}<extra></extra>"
        )
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=16
        ),
        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=20,
            color="#304F70",
            weight="bold"
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.10,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def crear_grafica_estrato(df):

    df_estratos = (
        df[df["Estrato"] > 0]
        .groupby("Estrato")["Puntaje estimado"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        df_estratos,
        x="Estrato",
        y="Puntaje estimado",
        title="Puntaje global estimado promedio por estrato",
        labels={
            "Estrato": "Estrato social",
            "Puntaje estimado": "Puntaje promedio estimado"
        },
        color_discrete_sequence=["#4E79A7"]
    )

    fig.add_hline(
        y=300,
        line_dash="dash",
        line_color="#D9534F",
        line_width=2,
        annotation_text="Umbral: 300",
        annotation_position="top right"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=16
        ),
        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=20,
            color="#304F70",
            weight="bold"
        ),
        margin=dict(l=45, r=25, t=65, b=45),
        showlegend=False
    )

    fig.update_xaxes(title="Estrato social", dtick=1)
    fig.update_yaxes(
        title="Puntaje promedio estimado",
        range=[0, 500],
        gridcolor="#E8EAF0"
    )

    return fig


def crear_grafica_recursos(df, variable, titulo):

    df_recursos = df[variable].value_counts().reset_index()
    df_recursos.columns = ["Respuesta", "Cantidad"]

    fig = px.pie(
        df_recursos,
        names="Respuesta",
        values="Cantidad",
        hole=0.55,
        title=titulo,
        color="Respuesta",
        color_discrete_map={
            "Sí": "#4E79A7",
            "No": "#8BCF7A",
            "No informado": "#BDBDBD"
        }
    )

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Estudiantes: %{value}<br>"
            "Porcentaje: %{percent}<extra></extra>"
        )
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=16
        ),
        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=20,
            color="#304F70",
            weight="bold"
        ),
        margin=dict(l=35, r=35, t=60, b=35),
        showlegend=False
    )

    return fig


def crear_grafica_importancia(lista_importancia):

    if not lista_importancia:
        fig = px.bar(
            title="Importancia global de las variables"
        )

        fig.add_annotation(
            text=(
                "La API todavía no devuelve "
                "la importancia de las variables."
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=15, color="#304F70")
        )

        fig.update_layout(template="plotly_white")

        return fig

    df_imp = pd.DataFrame(lista_importancia)

    fig = px.bar(
        df_imp.sort_values("peso", ascending=True),
        x="peso",
        y="variable",
        orientation="h",
        title="Importancia global de las variables",
        text="peso"
    )

    fig.update_traces(
        marker_color="#527BAA",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Magnitud de impacto",
        yaxis_title="",
        template="plotly_white",
        showlegend=False,
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=16
        ),
        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=22,
            color="#304F70",
            weight="bold"
        ),
        title_x=0.5
    )

    return fig


# ============================================================
# 6. FUNCIÓN PARA LEER EL ARCHIVO EN EL DASHBOARD
# ============================================================

def leer_archivo_dashboard(decoded, filename):

    nombre = filename.lower()

    if nombre.endswith(".csv"):

        try:
            return pd.read_csv(
                io.BytesIO(decoded),
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                io.BytesIO(decoded),
                encoding="latin-1"
            )

    elif nombre.endswith((".xls", ".xlsx")):

        return pd.read_excel(
            io.BytesIO(decoded)
        )

    raise ValueError(
        "Formato no válido. Sube un archivo CSV o Excel."
    )


# ============================================================
# 7. LAYOUT DE LA APLICACIÓN DASH
# ============================================================

app = Dash(__name__)
server = app.server


encabezado = html.Div([
    html.Div("MIAD - DSA", className="logo"),
    html.Div(
        "MÓDULO PREDICTIVO: DESEMPEÑO ACADÉMICO PRUEBAS SABER 11",
        className="header-title"
    )
], className="header")


barra_herramientas = html.Div([
    dcc.Upload(
        id="upload-data",
        children=html.Button(
            "Sube la base (.csv o .xlsx)",
            className="btn-upload"
        ),
        multiple=False
    ),
    html.Button(
        "Descargar reporte con predicción",
        id="btn-download",
        className="btn-download"
    ),
    dcc.Download(id="download-reporte"),
    html.Div(
        id="estado-api-mensaje",
        style={
            "marginLeft": "15px",
            "fontWeight": "bold",
            "color": "#304F70"
        }
    )
], className="toolbar")


graficas_estudiantes = html.Div([

    html.Div([
        dcc.Graph(
            id="grafica-distribucion-puntajes",
            figure=crear_histograma_puntajes(df_inicial),
            config={"displayModeBar": False}
        )
    ], className="chart-container chart-full"),

    html.Div([
        dcc.Graph(
            id="grafica-acompanamiento",
            figure=crear_grafica_acompanamiento(df_inicial),
            config={"displayModeBar": False}
        )
    ], className="chart-container"),

    html.Div([
        dcc.Graph(
            id="grafica-puntaje-estrato",
            figure=crear_grafica_estrato(df_inicial),
            config={"displayModeBar": False}
        )
    ], className="chart-container"),

    html.Div([
        dcc.Graph(
            id="grafica-internet",
            figure=crear_grafica_recursos(
                df_inicial,
                "Internet",
                "Acceso a Internet (Sí / No)"
            ),
            config={"displayModeBar": False}
        )
    ], className="chart-container"),

    html.Div([
        dcc.Graph(
            id="grafica-computador",
            figure=crear_grafica_recursos(
                df_inicial,
                "Computador",
                "Tiene computador (Sí / No)"
            ),
            config={"displayModeBar": False}
        )
    ], className="chart-container")

], className="charts-grid")


seccion_estudiantes = html.Div([

    html.H2(
        "1: Sobre las predicciones de tus estudiantes",
        className="section-title"
    ),

    html.Div([
        crear_tarjeta(
            "Cantidad de registros",
            f"{len(df_inicial)}",
            "card-registros"
        ),
        crear_tarjeta(
            "Puntaje promedio estimado",
            f"{df_inicial['Puntaje estimado'].mean():.1f}",
            "card-promedio"
        ),
        crear_tarjeta(
            "Porcentaje con puntaje ≥ 300",
            f"{(df_inicial['Puntaje estimado'] >= 300).mean()*100:.1f}%",
            "card-300"
        )
    ], className="indicators-container"),

    graficas_estudiantes

], className="dashboard-section")


seccion_individual = html.Div([

    html.H2(
        "2: Explicación de la predicción individual",
        className="section-title"
    ),

    html.P(
        "Selecciona un estudiante para consultar su resultado.",
        className="section-description"
    ),

    dcc.Dropdown(
        id="selector-estudiante",
        className="student-dropdown",
        options=[
            {"label": est, "value": est}
            for est in df_inicial["ID estudiante"]
        ],
        value=df_inicial["ID estudiante"].iloc[0],
        clearable=False,
        placeholder="Selecciona un estudiante"
    ),

    html.Div([

        html.Div([
            html.P(
                "Puntaje global estimado",
                className="card-title"
            ),
            html.H3(
                id="puntaje-individual",
                className="card-value"
            )
        ], className="individual-card score-card"),

        html.Div([
            html.P(
                "Estado de acompañamiento",
                className="card-title"
            ),
            html.H3(
                id="estado-individual",
                className="card-value"
            )
        ], className="individual-card status-card")

    ], className="individual-indicators"),

    html.Div([
        dcc.Graph(
            id="grafica-shap-individual",
            config={"displayModeBar": False}
        )
    ], className="chart-container"),

    html.Div([
        html.H4(
            "Detalle de contribuciones de las variables",
            className="table-title"
        ),
        html.Div(id="tabla-contribuciones")
    ], className="chart-container")

], className="dashboard-section")


seccion_modelo = html.Div([

    html.H2(
        "3: Métricas del modelo utilizado",
        className="section-title"
    ),

    html.Div([

        crear_tarjeta(
            "Error absoluto medio - MAE",
            MAE_MODELO,
            "card-mae"
        ),

        crear_tarjeta(
            "Raíz del error cuadrático medio - RMSE",
            RMSE_MODELO,
            "card-rmse"
        ),

        crear_tarjeta(
            "Coeficiente de determinación - R²",
            R2_MODELO,
            "card-r2"
        )

    ], className="indicators-container"),

    html.Div([
        dcc.Graph(
            id="grafica-importancia-global",
            figure=crear_grafica_importancia(IMPORTANCIA_INICIAL),
            config={"displayModeBar": False}
        )
    ], className="chart-container"),

    html.Div([

        html.H4(
            "Breve descripción de los datos empleados en el modelo",
            className="data-info-title"
        ),

        html.Ul([

            html.Li([
                "URL de obtención de datos Saber 11: ",
                html.A(
                    "Bogotá histórico Saber 11 | Datos Abiertos",
                    href="https://www.datos.gov.co/",
                    target="_blank"
                )
            ]),

            html.Li("Cobertura geográfica: Bogotá."),

            html.Li(
                "Unidad de observación utilizada: estudiante que presentó la prueba, no colegio ni municipio."
            ),

            html.Li(
                "Algoritmo en producción: Regresión Lineal Múltiple."
            ),

            html.Li(
                "Variable objetivo definida: Puntaje Global (PUNT_GLOBAL)."
            )

        ], className="data-info-list")

    ], className="data-info-container")

], className="dashboard-section")


app.layout = html.Div([

    dcc.Store(
        id="store-datos-actuales",
        data=df_inicial.to_dict("records")
    ),

    encabezado,

    html.Div([
        barra_herramientas,
        seccion_estudiantes,
        seccion_individual,
        seccion_modelo
    ], className="main-content")

])


# ============================================================
# 8. CALLBACK: CARGAR ARCHIVO Y CONSULTAR LA API
# ============================================================

@app.callback(

    Output("store-datos-actuales", "data"),
    Output("estado-api-mensaje", "children"),

    Output("card-registros-valor", "children"),
    Output("card-promedio-valor", "children"),
    Output("card-300-valor", "children"),

    Output("grafica-distribucion-puntajes", "figure"),
    Output("grafica-acompanamiento", "figure"),
    Output("grafica-puntaje-estrato", "figure"),
    Output("grafica-internet", "figure"),
    Output("grafica-computador", "figure"),
    Output("grafica-importancia-global", "figure"),

    Output("selector-estudiante", "options"),
    Output("selector-estudiante", "value"),

    Input("upload-data", "contents"),
    State("upload-data", "filename"),

    prevent_initial_call=True

)

def procesar_archivo_y_llamar_api(contents, filename):

    if contents is None:
        return (no_update,) * 13

    try:

        # ----------------------------------------------------
        # A. DECODIFICAR ARCHIVO
        # ----------------------------------------------------

        _, content_string = contents.split(",", 1)

        decoded = base64.b64decode(content_string)

        # ----------------------------------------------------
        # B. LEER ARCHIVO PARA EL DASHBOARD
        # ----------------------------------------------------

        df_subido = leer_archivo_dashboard(
            decoded,
            filename
        )

        if df_subido.empty:
            raise ValueError("El archivo no contiene registros.")

        # ----------------------------------------------------
        # C. ENVIAR ARCHIVO ORIGINAL A FASTAPI
        # ----------------------------------------------------

        nombre = filename.lower()

        if nombre.endswith(".csv"):
            mime_type = "text/csv"

        elif nombre.endswith(".xlsx"):
            mime_type = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )

        elif nombre.endswith(".xls"):
            mime_type = "application/vnd.ms-excel"

        else:
            raise ValueError("Formato de archivo no soportado.")

        files = {
            "file": (
                filename,
                decoded,
                mime_type
            )
        }

        response = requests.post(
            API_URL,
            files=files,
            timeout=60
        )

        response.raise_for_status()

        resultado_api = response.json()

        # ----------------------------------------------------
        # D. RECUPERAR PREDICCIONES
        # ----------------------------------------------------

        predicciones = resultado_api.get("predictions")

        if predicciones is None:
            raise ValueError(
                f"La API no devolvió predicciones: {resultado_api}"
            )

        if len(predicciones) != len(df_subido):
            raise ValueError(
                "La cantidad de predicciones devuelta por la API "
                "no coincide con los registros del archivo."
            )

        # Recuperar importancia global y contribuciones de la API
        importancia_api = resultado_api.get(
            "importancia_global",
            []
        )

        contribuciones_api = resultado_api.get(
            "contribuciones_individuales",
            []
        )

        # ----------------------------------------------------
        # E. PREPARAR DATAFRAME PARA EL DASHBOARD
        # ----------------------------------------------------

        df_actualizado = preparar_dataframe_tablero(
            df_subido,
            predicciones,
            contribuciones_api
        )

        # ----------------------------------------------------
        # F. ACTUALIZAR TARJETAS
        # ----------------------------------------------------

        n_reg = f"{len(df_actualizado):,}"

        prom = (
            f"{df_actualizado['Puntaje estimado'].mean():.1f}"
        )

        pct_300 = (
            f"{(df_actualizado['Puntaje estimado'] >= 300).mean()*100:.1f}%"
        )

        # ----------------------------------------------------
        # G. ACTUALIZAR TODAS LAS GRÁFICAS
        # ----------------------------------------------------

        fig_hist = crear_histograma_puntajes(
            df_actualizado
        )

        fig_acomp = crear_grafica_acompanamiento(
            df_actualizado
        )

        fig_est = crear_grafica_estrato(
            df_actualizado
        )

        fig_net = crear_grafica_recursos(
            df_actualizado,
            "Internet",
            "Acceso a Internet (Sí / No)"
        )

        fig_pc = crear_grafica_recursos(
            df_actualizado,
            "Computador",
            "Tiene computador (Sí / No)"
        )

        fig_imp = crear_grafica_importancia(
            importancia_api
        )

        # ----------------------------------------------------
        # H. ACTUALIZAR SELECTOR DE ESTUDIANTES
        # ----------------------------------------------------

        opciones_dropdown = [
            {"label": est, "value": est}
            for est in df_actualizado["ID estudiante"]
        ]

        primer_estudiante = (
            df_actualizado["ID estudiante"].iloc[0]
        )

        # Evitar problemas de serialización con valores nulos
        df_actualizado = (
            df_actualizado.astype(object)
            .where(pd.notna(df_actualizado), None)
        )

        mensaje = (
            f"Archivo '{filename}' procesado correctamente "
            f"vía API. Registros: {len(df_actualizado)}. "
            f"Versión del modelo: {resultado_api.get('version', 'No informada')}."
        )

        # ----------------------------------------------------
        # I. DEVOLVER RESULTADOS AL DASHBOARD
        # ----------------------------------------------------

        return (

            df_actualizado.to_dict("records"),
            mensaje,

            n_reg,
            prom,
            pct_300,

            fig_hist,
            fig_acomp,
            fig_est,
            fig_net,
            fig_pc,
            fig_imp,

            opciones_dropdown,
            primer_estudiante

        )

    except requests.exceptions.ConnectionError:

        mensaje_error = (
            f"No fue posible conectar con la API: {API_URL}. "
            "Verifica que FastAPI esté ejecutándose."
        )

        return (
            no_update,
            mensaje_error
        ) + (no_update,) * 11

    except requests.exceptions.Timeout:

        mensaje_error = (
            "La API tardó demasiado en responder. "
            "Intenta nuevamente."
        )

        return (
            no_update,
            mensaje_error
        ) + (no_update,) * 11

    except requests.exceptions.HTTPError as e:

        detalle = ""

        if e.response is not None:
            detalle = e.response.text

        mensaje_error = (
            f"Error HTTP de FastAPI: {e}. {detalle}"
        )

        return (
            no_update,
            mensaje_error
        ) + (no_update,) * 11

    except Exception as e:

        mensaje_error = (
            f"Error procesando el archivo: {str(e)}"
        )

        return (
            no_update,
            mensaje_error
        ) + (no_update,) * 11


# ============================================================
# 9. CALLBACK: INFORMACIÓN INDIVIDUAL
# ============================================================

@app.callback(

    Output("puntaje-individual", "children"),
    Output("estado-individual", "children"),
    Output("grafica-shap-individual", "figure"),
    Output("tabla-contribuciones", "children"),

    Input("selector-estudiante", "value"),
    Input("store-datos-actuales", "data")

)
def actualizar_estudiante(
    estudiante_seleccionado,
    datos_store
):

    df_actual = pd.DataFrame(datos_store)

    if df_actual.empty:
        return "N/A", "N/A", {}, "No hay datos disponibles."

    filtro = df_actual[
        df_actual["ID estudiante"] == estudiante_seleccionado
    ]

    estudiante = (
        filtro.iloc[0]
        if not filtro.empty
        else df_actual.iloc[0]
    )

    puntaje = float(estudiante["Puntaje estimado"])
    estado = estudiante["Estado"]

    # --------------------------------------------------------
    # RECUPERAR CONTRIBUCIONES DEL ESTUDIANTE
    # --------------------------------------------------------

    contribuciones = estudiante.get(
        "_contribuciones_api",
        None
    )

    if not isinstance(contribuciones, list) or not contribuciones:

        fig = px.bar(
            title="Contribuciones de variables a la predicción"
        )

        fig.add_annotation(
            text="No hay contribuciones individuales disponibles.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=15, color="#304F70")
        )

        fig.update_layout(template="plotly_white")

        tabla = html.P(
            "La API todavía no ha proporcionado "
            "contribuciones para este estudiante."
        )

        return round(puntaje, 2), estado, fig, tabla

    # --------------------------------------------------------
    # PREPARAR CONTRIBUCIONES
    # --------------------------------------------------------

    df_contribuciones = pd.DataFrame(contribuciones)

    df_contribuciones = df_contribuciones.sort_values(
        "contribucion",
        ascending=True
    )

    # --------------------------------------------------------
    # GRÁFICA DE CONTRIBUCIONES
    # --------------------------------------------------------

    fig = px.bar(
        df_contribuciones,
        x="contribucion",
        y="variable",
        orientation="h",
        title="Contribuciones de variables a la predicción",
        labels={
            "contribucion": "Contribución en puntos",
            "variable": "Variable"
        },
        color="contribucion",
        color_continuous_scale="RdYlGn",
        text="contribucion"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#555555"
    )

    fig.update_layout(
        template="plotly_white",
        coloraxis_showscale=False,
        xaxis_title="Contribución en puntos",
        yaxis_title="",
        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=15
        ),
        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=20,
            color="#304F70",
            weight="bold"
        ),
        title_x=0.5
    )

    # --------------------------------------------------------
    # TABLA DE CONTRIBUCIONES
    # --------------------------------------------------------

    filas = []

    for _, fila in df_contribuciones.iterrows():

        valor = float(fila["contribucion"])

        impacto = (
            "Positivo"
            if valor > 0
            else "Negativo"
            if valor < 0
            else "Neutro"
        )

        filas.append(
            html.Tr([
                html.Td(fila["variable"]),
                html.Td(f"{valor:.2f}"),
                html.Td(impacto)
            ])
        )

    tabla = html.Table([

        html.Thead(
            html.Tr([
                html.Th("Variable"),
                html.Th("Contribución"),
                html.Th("Impacto")
            ])
        ),

        html.Tbody(filas)

    ], className="contributions-table")

    return round(puntaje, 2), estado, fig, tabla


# ============================================================
# 10. CALLBACK: DESCARGAR REPORTE
# ============================================================

@app.callback(

    Output("download-reporte", "data"),

    Input("btn-download", "n_clicks"),

    State("store-datos-actuales", "data"),

    prevent_initial_call=True

)

def descargar_reporte(n_clicks, datos_store):

    df_export = pd.DataFrame(datos_store).drop(
        columns=["_contribuciones_api"],
        errors="ignore"
    )

    return dcc.send_data_frame(
        df_export.to_csv,
        "predicciones_saber11.csv",
        index=False
    )


# ============================================================
# 11. EJECUTAR APLICACIÓN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8050,
        debug=True
    )
