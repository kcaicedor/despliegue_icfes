# IMPORTACION LIBRERIAS
import os
import io
import base64
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, no_update

# 1. CONFIGURACIÓN DE CONEXIÓN A LA API
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/api/v1/predict")


# 2. DATOS INICIALES (ESTADO POR DEFECTO)
MAE_MODELO = 0.0
RMSE_MODELO = 0.0
R2_MODELO = 0.0

def generar_datos_iniciales():
    """Genera datos base mientras el usuario carga un archivo .csv o .xlsx."""
    np.random.seed(42)
    puntajes = np.clip(np.random.normal(loc=255, scale=45, size=200), 0, 500)
    df = pd.DataFrame({
        "ID estudiante": [f"Estudiante {i:03d}" for i in range(1, 201)],
        "Estrato": np.random.choice([1, 2, 3, 4, 5, 6], size=200, p=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04]),
        "Internet": np.random.choice(["Sí", "No"], size=200, p=[0.61, 0.39]),
        "Computador": np.random.choice(["Sí", "No"], size=200, p=[0.61, 0.39]),
        "Puntaje estimado": np.round(puntajes, 2)
    })
    df["Estado"] = np.where(df["Puntaje estimado"] < 300, "Requiere acompañamiento", "No requiere acompañamiento")
    return df

df_inicial = generar_datos_iniciales()

IMPORTANCIA_INICIAL = [
    {"variable": "Educación de la madre", "peso": 0.0},
    {"variable": "Estrato socioeconómico", "peso": 0.0},
    {"variable": "Educación del padre", "peso": 0.0},
    {"variable": "Acceso a Internet", "peso": 0.0},
    {"variable": "Disponibilidad de computador", "peso": 0.0},
    {"variable": "Jornada escolar", "peso": 0.0}
]


# 3. FUNCIONES DE PROCESAMIENTO Y GRÁFICAS

def preparar_dataframe_tablero(df_raw: pd.DataFrame, predicciones: list, contribuciones: list = None) -> pd.DataFrame:
    """Combina los datos subidos con las predicciones y contribuciones reales de la API."""
    df = df_raw.copy()
    df["Puntaje estimado"] = np.round(np.clip(predicciones, 0, 500), 2)
    df["Estado"] = np.where(df["Puntaje estimado"] < 300, "Requiere acompañamiento", "No requiere acompañamiento")

    if contribuciones and len(contribuciones) == len(df):
        df["_contribuciones_api"] = contribuciones
    else:
        df["_contribuciones_api"] = None

    if "ID estudiante" not in df.columns:
        df["ID estudiante"] = [f"Estudiante {i:04d}" for i in range(1, len(df) + 1)]

    if "Estrato" not in df.columns:
        if "fami_estratovivienda" in df.columns:
            df["Estrato"] = (
                df["fami_estratovivienda"]
                .astype(str)
                .str.extract(r"(\d+)")[0]
                .fillna("0")
                .astype(int)
            )
        else:
            df["Estrato"] = 0

    if "Internet" not in df.columns:
        if "fami_tieneinternet" in df.columns:
            df["Internet"] = df["fami_tieneinternet"].replace({"Si": "Sí", "SI": "Sí", "NO": "No"}).fillna("No")
        else:
            df["Internet"] = "No"

    if "Computador" not in df.columns:
        if "fami_tienecomputador" in df.columns:
            df["Computador"] = df["fami_tienecomputador"].replace({"Si": "Sí", "SI": "Sí", "NO": "No"}).fillna("No")
        else:
            df["Computador"] = "No"

    return df

def crear_tarjeta(titulo, valor, identificador):
    return html.Div([
        html.P(titulo, className="card-title"),
        html.H3(valor, className="card-value", id=f"{identificador}-valor")
    ], className="indicator-card", id=identificador)

def crear_histograma_puntajes(df):
    fig = px.histogram(
        df,
        x="Puntaje estimado",
        nbins=25,
        title="Distribución de puntajes globales estimados",
        labels={"Puntaje estimado": "Puntaje global estimado", "count": "Cantidad de estudiantes"},
        color_discrete_sequence=["#4E79A7"]
    )
    fig.add_vline(x=300, line_dash="dash", line_color="#D9534F", line_width=2, annotation_text="Umbral: 300", annotation_position="top right")
    promedio = df["Puntaje estimado"].mean()
    fig.add_vline(x=promedio, line_dash="dot", line_color="#2E8B57", line_width=2, annotation_text=f"Promedio: {promedio:.1f}", annotation_position="top left")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#304F70", size=16),
        title_font=dict(family="Segoe UI, Arial, sans-serif", size=22, color="#304F70", weight="bold"),
        title_x=0.5, margin=dict(l=45, r=25, t=65, b=45), showlegend=False
    )
    fig.update_xaxes(range=[0, 500], showgrid=False)
    fig.update_yaxes(title="Cantidad de estudiantes", gridcolor="#E8EAF0")
    return fig

def crear_grafica_acompanamiento(df):
    df_estado = df["Estado"].value_counts().reset_index()
    df_estado.columns = ["Estado", "Cantidad"]
    fig = px.pie(
        df_estado, names="Estado", values="Cantidad", hole=0,
        title="Estudiantes según necesidad de acompañamiento", color="Estado",
        color_discrete_map={"Requiere acompañamiento": "#4E79A7", "No requiere acompañamiento": "#8BCF7A"}
    )
    fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="<b>%{label}</b><br>Estudiantes: %{value}<br>Porcentaje: %{percent}<extra></extra>")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#304F70", size=16),
        title_font=dict(family="Segoe UI, Arial, sans-serif", size=20, color="#304F70", weight="bold"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="center", x=0.5)
    )
    return fig

def crear_grafica_estrato(df):
    df_estratos = df[df["Estrato"] > 0].groupby("Estrato")["Puntaje estimado"].mean().reset_index()
    fig = px.bar(
        df_estratos, x="Estrato", y="Puntaje estimado",
        title="Puntaje global estimado promedio por estrato",
        labels={"Estrato": "Estrato social", "Puntaje estimado": "Puntaje promedio estimado"},
        color_discrete_sequence=["#4E79A7"]
    )
    fig.add_hline(y=300, line_dash="dash", line_color="#D9534F", line_width=2, annotation_text="Umbral: 300", annotation_position="top right")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#304F70", size=16),
        title_font=dict(family="Segoe UI, Arial, sans-serif", size=20, color="#304F70", weight="bold"),
        margin=dict(l=45, r=25, t=65, b=45), showlegend=False
    )
    fig.update_xaxes(title="Estrato social", dtick=1)
    fig.update_yaxes(title="Puntaje promedio estimado", range=[0, 500], gridcolor="#E8EAF0")
    return fig

def crear_grafica_recursos(df, variable, titulo):
    df_recursos = df[variable].value_counts().reset_index()
    df_recursos.columns = ["Respuesta", "Cantidad"]
    fig = px.pie(
        df_recursos, names="Respuesta", values="Cantidad", hole=0.55, title=titulo,
        color="Respuesta", color_discrete_map={"Sí": "#4E79A7", "No": "#8BCF7A"}
    )
    fig.update_traces(textposition="outside", textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Estudiantes: %{value}<br>Porcentaje: %{percent}<extra></extra>")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#304F70", size=16),
        title_font=dict(family="Segoe UI, Arial, sans-serif", size=20, color="#304F70", weight="bold"),
        margin=dict(l=35, r=35, t=60, b=35), showlegend=False
    )
    return fig

def crear_grafica_importancia(lista_importancia):
    df_imp = pd.DataFrame(lista_importancia)
    fig = px.bar(
        df_imp.sort_values("peso", ascending=True),
        x="peso", y="variable", orientation="h",
        title="Importancia global de las variables (Coeficientes del Modelo)", text="peso"
    )
    fig.update_traces(marker_color="#527BAA", texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Magnitud de impacto (|β|)", yaxis_title="", template="plotly_white", showlegend=False,
        font=dict(family="Segoe UI, Arial, sans-serif", color="#304F70", size=16),
        title_font=dict(family="Segoe UI, Arial, sans-serif", size=22, color="#304F70", weight="bold"),
        title_x=0.5
    )
    return fig


# 4. LAYOUT DE LA APLICACIÓN DASH

app = Dash(__name__)
server = app.server

encabezado = html.Div([
    html.Div("MIAD - DSA", className="logo"),
    html.Div("MÓDULO PREDICTIVO: DESEMPEÑO ACADÉMICO PRUEBAS SABER 11", className="header-title")
], className="header")

barra_herramientas = html.Div([
    dcc.Upload(
        id="upload-data",
        children=html.Button("Sube la base (.csv o .xlsx)", className="btn-upload"),
        multiple=False
    ),
    html.Button("Descargar reporte con predicción", id="btn-download", className="btn-download"),
    dcc.Download(id="download-reporte"),
    html.Div(id="estado-api-mensaje", style={"marginLeft": "15px", "fontWeight": "bold", "color": "#304F70"})
], className="toolbar")

graficas_estudiantes = html.Div([
    html.Div([dcc.Graph(id="grafica-distribucion-puntajes", figure=crear_histograma_puntajes(df_inicial), config={"displayModeBar": False})], className="chart-container chart-full"),
    html.Div([dcc.Graph(id="grafica-acompanamiento", figure=crear_grafica_acompanamiento(df_inicial), config={"displayModeBar": False})], className="chart-container"),
    html.Div([dcc.Graph(id="grafica-puntaje-estrato", figure=crear_grafica_estrato(df_inicial), config={"displayModeBar": False})], className="chart-container"),
    html.Div([dcc.Graph(id="grafica-internet", figure=crear_grafica_recursos(df_inicial, "Internet", "Acceso a Internet (Sí / No)"), config={"displayModeBar": False})], className="chart-container"),
    html.Div([dcc.Graph(id="grafica-computador", figure=crear_grafica_recursos(df_inicial, "Computador", "Tiene computador (Sí / No)"), config={"displayModeBar": False})], className="chart-container")
], className="charts-grid")

seccion_estudiantes = html.Div([
    html.H2("1: Sobre las predicciones de tus estudiantes", className="section-title"),
    html.Div([
        crear_tarjeta("Cantidad de registros", f"{len(df_inicial)}", "card-registros"),
        crear_tarjeta("Puntaje promedio estimado", f"{df_inicial['Puntaje estimado'].mean():.1f}", "card-promedio"),
        crear_tarjeta("Porcentaje con puntaje ≥ 300", f"{(df_inicial['Puntaje estimado'] >= 300).mean()*100:.1f}%", "card-300")
    ], className="indicators-container"),
    graficas_estudiantes
], className="dashboard-section")

seccion_individual = html.Div([
    html.H2("2: Explicación de la predicción individual", className="section-title"),
    html.P("Selecciona un estudiante para consultar su resultado.", className="section-description"),
    dcc.Dropdown(
        id="selector-estudiante",
        className="student-dropdown",
        options=[{"label": est, "value": est} for est in df_inicial["ID estudiante"]],
        value=df_inicial["ID estudiante"].iloc[0],
        clearable=False,
        placeholder="Selecciona un estudiante"
    ),
    html.Div([
        html.Div([
            html.P("Puntaje global estimado", className="card-title"),
            html.H3(id="puntaje-individual", className="card-value")
        ], className="individual-card score-card"),
        html.Div([
            html.P("Estado de acompañamiento", className="card-title"),
            html.H3(id="estado-individual", className="card-value")
        ], className="individual-card status-card")
    ], className="individual-indicators"),
    html.Div([dcc.Graph(id="grafica-shap-individual", config={"displayModeBar": False})], className="chart-container"),
    html.Div([
        html.H4("Detalle de contribuciones de las variables", className="table-title"),
        html.Div(id="tabla-contribuciones")
    ], className="chart-container"),
], className="dashboard-section")

seccion_modelo = html.Div([
    html.H2("3: Métricas del modelo utilizado", className="section-title"),
    html.Div([
        crear_tarjeta("Error absoluto medio - MAE", MAE_MODELO, "card-mae"),
        crear_tarjeta("Raíz del error cuadrático medio - RMSE", RMSE_MODELO, "card-rmse"),
        crear_tarjeta("Coeficiente de determinación - R²", R2_MODELO, "card-r2")
    ], className="indicators-container"),
    html.Div([dcc.Graph(id="grafica-importancia-global", figure=crear_grafica_importancia(IMPORTANCIA_INICIAL), config={"displayModeBar": False})], className="chart-container"),
    html.Div([
        html.H4("Breve descripción de los datos empleados en el modelo", className="data-info-title"),
        html.Ul([
            html.Li(["URL de obtención de datos Saber 11: ", html.A("Bogotá histórico Saber 11 | Datos Abiertos", href="https://www.datos.gov.co/", target="_blank")]),
            html.Li("Cobertura geográfica: Bogotá."),
            html.Li("Unidad de observación utilizada: estudiante que presentó la prueba, no colegio ni municipio."),
            html.Li("Algoritmo en producción: Regresión Lineal Múltiple (Baseline ganador en MLflow)."),
            html.Li("Variable objetivo definida: Puntaje Global (PUNT_GLOBAL).")
        ], className="data-info-list")
    ], className="data-info-container")
], className="dashboard-section")

app.layout = html.Div([
    dcc.Store(id="store-datos-actuales", data=df_inicial.to_dict("records")),
    encabezado,
    html.Div([barra_herramientas, seccion_estudiantes, seccion_individual, seccion_modelo], className="main-content")
])


# 5. CALLBACKS (CONEXIÓN API E INTERACTIVIDAD CON DASH)

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
# Esta función procesa el archivo subido y llama a la API para obtener predicciones y contribuciones.
def procesar_archivo_y_llamar_api(contents, filename):
    if contents is None:
        return (no_update,) * 13

    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        if filename.endswith(".csv"):
            df_subido = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith((".xls", ".xlsx")):
            df_subido = pd.read_excel(io.BytesIO(decoded))
        else:
            return (no_update, "Formato no válido. Sube .csv o .xlsx") + (no_update,) * 11

        df_payload = df_subido.where(pd.notnull(df_subido), None)
        payload = {"inputs": df_payload.to_dict(orient="records")}

        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        resultado_api = response.json()

        predicciones = resultado_api["predictions"]
        importancia_api = resultado_api.get("importancia_global", IMPORTANCIA_INICIAL)
        contribuciones_api = resultado_api.get("contribuciones_individuales", None)

        df_actualizado = preparar_dataframe_tablero(df_subido, predicciones, contribuciones_api)

        n_reg = f"{len(df_actualizado):,}"
        prom = f"{df_actualizado['Puntaje estimado'].mean():.1f}"
        pct_300 = f"{(df_actualizado['Puntaje estimado'] >= 300).mean()*100:.1f}%"

        fig_hist = crear_histograma_puntajes(df_actualizado)
        fig_acomp = crear_grafica_acompanamiento(df_actualizado)
        fig_est = crear_grafica_estrato(df_actualizado)
        fig_net = crear_grafica_recursos(df_actualizado, "Internet", "Acceso a Internet (Sí / No)")
        fig_pc = crear_grafica_recursos(df_actualizado, "Computador", "Tiene computador (Sí / No)")
        fig_imp = crear_grafica_importancia(importancia_api)

        opciones_dropdown = [{"label": est, "value": est} for est in df_actualizado["ID estudiante"]]
        primer_estudiante = df_actualizado["ID estudiante"].iloc[0]

        mensaje = f"Archivo '{filename}' procesado vía API ({len(df_actualizado)} registros)."
        return (
            df_actualizado.to_dict("records"), mensaje,
            n_reg, prom, pct_300,
            fig_hist, fig_acomp, fig_est, fig_net, fig_pc, fig_imp,
            opciones_dropdown, primer_estudiante
        )

    except requests.exceptions.RequestException as e:
        mensaje_error = f"Error conectando con la API ({API_URL}): {e}"
        return (no_update, mensaje_error) + (no_update,) * 11
    except Exception as e:
        mensaje_error = f"Error procesando el archivo: {e}"
        return (no_update, mensaje_error) + (no_update,) * 11


@app.callback(
    Output("puntaje-individual", "children"),
    Output("estado-individual", "children"),
    Output("grafica-shap-individual", "figure"),
    Output("tabla-contribuciones", "children"),
    Input("selector-estudiante", "value"),
    Input("store-datos-actuales", "data")
)
# Esta función actualiza la información individual del estudiante seleccionado.
def actualizar_estudiante(estudiante_seleccionado, datos_store):
    df_actual = pd.DataFrame(datos_store)
    filtro = df_actual[df_actual["ID estudiante"] == estudiante_seleccionado]
    estudiante = filtro.iloc[0] if not filtro.empty else df_actual.iloc[0]

    puntaje = float(estudiante["Puntaje estimado"])
    estado = estudiante["Estado"]

    # Usa contribuciones reales si vinieron desde la API; de lo contrario, usa cálculo de respaldo
    contrib_api = estudiante.get("_contribuciones_api", None)
    if isinstance(contrib_api, list) and len(contrib_api) > 0:
        aportes = pd.DataFrame(contrib_api).rename(columns={"variable": "Variable", "contribucion": "Contribución"})
    else:
        dif_estrato = (float(estudiante.get("Estrato", 3)) - 3) * 6.5
        dif_internet = 8.2 if estudiante.get("Internet") == "Sí" else -8.2
        dif_pc = 7.4 if estudiante.get("Computador") == "Sí" else -7.4
        aportes = pd.DataFrame({
            "Variable": ["Estrato socioeconómico", "Acceso a Internet", "Disponibilidad de computador"],
            "Contribución": [dif_estrato, dif_internet, dif_pc]
        })

    aportes["Tipo"] = np.where(aportes["Contribución"] >= 0, "Aumenta", "Disminuye")

    fig = px.bar(
        aportes, x="Contribución", y="Variable", orientation="h", color="Tipo",
        color_discrete_map={"Aumenta": "#5CB85C", "Disminuye": "#D9534F"},
        title="Contribución de variables clave a la predicción"
    )
    fig.update_layout(
        showlegend=False, xaxis_title="Impacto estimado (puntos)", yaxis_title="", template="plotly_white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#304F70", size=16),
        title_font=dict(family="Segoe UI, Arial, sans-serif", size=20, color="#304F70", weight="bold"),
        title_x=0.5
    )

    tabla = html.Table([
        html.Thead(html.Tr([html.Th("Variable"), html.Th("Contribución"), html.Th("Interpretación")])),
        html.Tbody([
            html.Tr([html.Td(fila["Variable"]), html.Td(f"{fila['Contribución']:+.2f}"), html.Td(fila["Tipo"])])
            for _, fila in aportes.iterrows()
        ])
    ], className="contributions-table")

    return round(puntaje, 2), estado, fig, tabla


@app.callback(
    Output("download-reporte", "data"),
    Input("btn-download", "n_clicks"),
    State("store-datos-actuales", "data"),
    prevent_initial_call=True
)
# Esta función permite descargar un reporte CSV con las predicciones actuales.
def descargar_reporte(n_clicks, datos_store):
    df_export = pd.DataFrame(datos_store).drop(columns=["_contribuciones_api"], errors="ignore")
    return dcc.send_data_frame(df_export.to_csv, "predicciones_saber11.csv", index=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)