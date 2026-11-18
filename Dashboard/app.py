
# 1. IMPORTACIONES

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Input, Output

# 2. SIMULACIÓN DE VALORES

# 2.1 DATOS SIMULADOS PARA LAS GRÁFICAS

# Sección 1 Tarjetas: Predicciones de los estudiantes
cantidad_registros = "25.2K"
puntaje_promedio_estimado = 252.5
porcentaje_con_puntaje = 65.8

# Sección 3 Tajetas: Evaluación del modelo
MAE = 22.5
RMSE = 31.8
R2 = 0.78

# 2.2 DATOS SOBRE ESTUDIANTES:

# Simular puntajes globales estimados de 1.000 estudiantes

np.random.seed(42)

puntajes_estimados = np.random.normal(loc=310,scale=105,size=1000)

# Limitar los puntajes al rango de 0 a 500

puntajes_estimados = np.clip(puntajes_estimados,0,500)

# Crear un DataFrame con los puntajes

df_puntajes = pd.DataFrame({"Puntaje estimado": puntajes_estimados})

# Crear identificador único para cada estudiante
df_puntajes["ID estudiante"] = [
    f"Estudiante {i:03d}"
    for i in range(1, len(df_puntajes) + 1)]

print(df_puntajes.head())

# Generar estratos socioeconómicos simulados

df_puntajes["Estrato"] = np.random.choice([1, 2, 3, 4, 5, 6],
    size=len(df_puntajes),
    p=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04])

# Generar diferencias ilustrativas en los puntajes según el estrato socioeconómico

ajustes_estrato = {1: -25,2: -15,3: -5,4: 5,5: 15,6: 25}

df_puntajes["Puntaje estimado"] = (
    df_puntajes["Puntaje estimado"]
    + df_puntajes["Estrato"].map(ajustes_estrato)
).clip(0, 500)

# Acceso a internet

df_puntajes["Internet"] = np.random.choice(["Sí", "No"],size=len(df_puntajes),p=[0.61, 0.39])

# Disponibilidad de computador

df_puntajes["Computador"] = np.random.choice(["Sí", "No"],size=len(df_puntajes),p=[0.61, 0.39])

# Clasificación

df_puntajes["Estado"] = np.where(df_puntajes["Puntaje estimado"] < 300,"Requiere acompañamiento","No requiere acompañamiento")

    # Contar estudiantes por categoría
df_estado = (
        df_puntajes["Estado"]
        .value_counts()
        .reset_index()
    )

df_estado.columns = ["Estado", "Cantidad"]

# 2.3 IMPORTANCIA GLOBAL DE VARIABLES (SIMULADA)

df_importancia = pd.DataFrame({"Variable": [
        "Educación de la madre",
        "Estrato socioeconómico",
        "Educación del padre",
        "Acceso a Internet",
        "Disponibilidad de computador",
        "Jornada escolar"
    ],
    "Importancia": [12.5, 10.8, 9.4, 7.2, 6.5, 4.3]})

# 3. CREAR LA APLICACIÓN

app = Dash(__name__)

# 4. FUNCIONES



# 4.1 Crear tarjetas_

def crear_tarjeta(titulo, valor, identificador):
    return html.Div([

        html.P(titulo,className="card-title"),
        html.H3(valor,className="card-value")
    ], className="indicator-card", id=identificador)



# 4.2 Distribución de puntajes estimados: 

def crear_histograma_puntajes():

    # Crear histograma
    fig = px.histogram(
        df_puntajes,
        x="Puntaje estimado",
        nbins=25,
        title="Distribución de puntajes globales estimados",
        labels={
            "Puntaje estimado": "Puntaje global estimado",
            "count": "Cantidad de estudiantes"
        },
        color_discrete_sequence=["#4E79A7"]
    )

    # Agregar línea del umbral de 300 puntos
    fig.add_vline(
        x=300,
        line_dash="dash",
        line_color="#D9534F",
        line_width=2,
        annotation_text="Umbral: 300",
        annotation_position="top right"
    )

    # Agregar línea del promedio estimado
    promedio = df_puntajes["Puntaje estimado"].mean()

    fig.add_vline(
        x=promedio,
        line_dash="dot",
        line_color="#2E8B57",
        line_width=2,
        annotation_text=f"Promedio: {promedio:.1f}",
        annotation_position="top left"
    )

    # Configurar apariencia
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
            size=25,
            color="#304F70",
            weight="bold"
        ),
        title_x=0.5,
        title_xanchor="center",
        margin=dict(l=45, r=25, t=65, b=45),
        showlegend=False,

        ),

    fig.update_xaxes(
        range=[0, 500],
        showgrid=False
    )

    fig.update_yaxes(
        title="Cantidad de estudiantes",
        gridcolor="#E8EAF0"
    )

    return fig



# 4.3 Estudiantes que reqquieren acompañamiento:

def crear_grafica_acompanamiento():
    # Crear gráfico de torta
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

    # Mostrar porcentajes
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Estudiantes: %{value}<br>"
            "Porcentaje: %{percent}"
            "<extra></extra>"
        )
    )

    # Configurar apariencia
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

# 4.4 Puntaje promedio por estrato:

def crear_grafica_estrato():

    # Calcular el puntaje promedio por estrato

    df_estratos = (
        df_puntajes
        .groupby("Estrato")["Puntaje estimado"]
        .mean()
        .reset_index()
    )

    # Crear gráfico de barras

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

    # Agregar línea de referencia de 300 puntos

    fig.add_hline(
        y=300,
        line_dash="dash",
        line_color="#D9534F",
        line_width=2,
        annotation_text="Umbral: 300",
        annotation_position="top right"
    )

    # Configurar apariencia

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

    fig.update_xaxes(
        title="Estrato social",
        dtick=1
    )

    fig.update_yaxes(
        title="Puntaje promedio estimado",
        range=[0, 500],
        gridcolor="#E8EAF0"
    )

    return fig

# 4.5 Grafica de accesos a recursos

def crear_grafica_recursos(variable, titulo):

    # Contar estudiantes por categoría

    df_recursos = (
        df_puntajes[variable]
        .value_counts()
        .reset_index()
    )

    df_recursos.columns = ["Respuesta", "Cantidad"]

    # Crear gráfico de dona

    fig = px.pie(
        df_recursos,
        names="Respuesta",
        values="Cantidad",
        hole=0.55,
        title=titulo,
        color="Respuesta",
        color_discrete_map={
            "Sí": "#4E79A7",
            "No": "#8BCF7A"
        }
    )

    # Mostrar porcentajes

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Estudiantes: %{value}<br>"
            "Porcentaje: %{percent}"
            "<extra></extra>"
        )
    )

    # Configurar apariencia

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

# 4.6 Grafica importancia

def crear_grafica_importancia():

    fig = px.bar(
        df_importancia.sort_values("Importancia", ascending=True),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia global de las variables",
        text="Importancia"
    )

    fig.update_traces(
        marker_color="#527BAA",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Importancia media absoluta (SHAP)",
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
            size=25,
            color="#304F70",
            weight="bold"
        ),

        title_x=0.5,
        title_xanchor="center",
    )

    return fig

# 5. ENCABEZADO

encabezado = html.Div([
    html.Div("ICFES",className="logo"),
    html.Div("MÓDULO PREDICTIVO: DESEMPEÑO ACADÉMICO PRUEBAS SABER 11",className="header-title")
], className="header")

# 6. BOTONES

barra_herramientas = html.Div([
    html.Button("Sube la base (.csv o .xlsx)",className="btn-upload"),
    html.Button("Descargar reporte con predicción",className="btn-download")
], className="toolbar")

# 7 SECCIÓN  1: INFORMACIÓN GENERAL

# 7.1 Gráficas de la primera sección:

graficas_estudiantes = html.Div([

    # Gráfica 1: Distribución de puntajes
    html.Div([
        dcc.Graph(id="grafica-distribucion-puntajes",figure=crear_histograma_puntajes(),config={"displayModeBar": False})
    ], className="chart-container chart-full"),

    # Gráfica 2: Acompañamiento
    html.Div([
        dcc.Graph(id="grafica-acompanamiento",figure=crear_grafica_acompanamiento(),config={"displayModeBar": False})
    ], className="chart-container"),

    # Gráfica 3: Puntaje por estrato
    html.Div([
        dcc.Graph(id="grafica-puntaje-estrato",figure=crear_grafica_estrato(),config={"displayModeBar": False})
    ], className="chart-container"),

    # Gráfica 4: Internet
    html.Div([
        dcc.Graph(id="grafica-internet",figure=crear_grafica_recursos("Internet","Acceso a Internet (Sí / No)"),config={"displayModeBar": False})
    ], className="chart-container"),

    # Gráfica 5: Computador
    html.Div([
        dcc.Graph(id="grafica-computador",figure=crear_grafica_recursos("Computador","Tiene computador (Sí / No)"),config={"displayModeBar": False})
    ], className="chart-container")

], className="charts-grid")


# 7.2 Predicciones de los estudiantes

seccion_estudiantes = html.Div([

    html.H2("1.0 Sobre las predicciones de tus estudiantes",className="section-title"),

    html.Div([

        crear_tarjeta("Cantidad de registros",cantidad_registros,"card-registros"),
        crear_tarjeta("Puntaje promedio estimado",puntaje_promedio_estimado,"card-promedio"        ),
        crear_tarjeta("Porcentaje con puntaje ≥ 300",f"{porcentaje_con_puntaje}%","card-300")
    ], className="indicators-container"),

    graficas_estudiantes

], className="dashboard-section")

# 8. SECCIÓN 2: EXPLICACIÓN INDIVIDUAL

seccion_individual = html.Div([

    html.H2("2.0 Explicación de la predicción individual",className="section-title"),

    html.P("Selecciona un estudiante para consultar su resultado.",className="section-description"),

    dcc.Dropdown(
        id="selector-estudiante",
        className="student-dropdown",
        options=[{
                "label": estudiante,
                "value": estudiante
            }
            for estudiante in df_puntajes["ID estudiante"]
        ],
        value=df_puntajes["ID estudiante"].iloc[0],
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

        html.Div([
            dcc.Graph(id="grafica-shap-individual",config={"displayModeBar": False})
        ], className="chart-container"),
            
        html.Div([html.H4("Detalle de contribuciones de las variables",className="table-title"),

            html.Div(id="tabla-contribuciones")

        ], className="chart-container"),


], className="dashboard-section")

# 9. SECCIÓN 3: MODELO UTILIZADO

seccion_modelo = html.Div([

    html.H2("3.0 Sobre el modelo utilizado",className="section-title"),

    html.Div([
        crear_tarjeta("Error absoluto medio - MAE",MAE,"card-mae"),
        crear_tarjeta("Raíz del error cuadrático medio - RMSE",RMSE,"card-rmse"),
        crear_tarjeta("Coeficiente de determinación - R²",R2,"card-r2")
    ], className="indicators-container"),

        html.Div([
            dcc.Graph(id="grafica-importancia-global",figure=crear_grafica_importancia(),config={"displayModeBar": False})
        ], className="chart-container"),

    # Información del conjunto de datos

    html.Div([

        html.H4("Breve descripción de los datos empleados en el modelo",className="data-info-title"),

        html.Ul([

            html.Li(["URL de obtención de datos Saber 11: ",html.A(
                    "Bogotá histórico Saber 11 | Datos Abiertos",
                    href="https://www.datos.gov.co/",
                    target="_blank")]),

            html.Li("Periodo de referencia y aplicación del examen: 2022-4"),
            html.Li("Cobertura geográfica: Bogotá."),
            html.Li("Unidad de observación utilizada: estudiante que presentó la prueba, no colegio ni municipio."),
            html.Li("Número de registros y variables iniciales: 152.584 registros y 51 columnas."),
            html.Li("Variable objetivo definida: Puntaje Global (PUNT_GLOBAL).")

        ], className="data-info-list")

    ], className="data-info-container")

], className="dashboard-section")



# 10. DISEÑO GENERAL DEL DASHBOARD

app.layout = html.Div([encabezado,
    html.Div([barra_herramientas,seccion_estudiantes,seccion_individual,seccion_modelo], className="main-content")])

# 11. EJECUTAR LA APLICACIÓN

@app.callback(
    Output("puntaje-individual", "children"),
    Output("estado-individual", "children"),
    Output("grafica-shap-individual", "figure"),
    Output("tabla-contribuciones", "children"),
    Input("selector-estudiante", "value"))

def actualizar_estudiante(estudiante_seleccionado):

    # Buscar la información del estudiante
    estudiante = df_puntajes[df_puntajes["ID estudiante"] == estudiante_seleccionado].iloc[0]

    # Obtener puntaje y estado
    puntaje = estudiante["Puntaje estimado"]
    estado = estudiante["Estado"]

    # Crear contribuciones simuladas para visualizar la gráfica
    indice = df_puntajes.index[df_puntajes["ID estudiante"] == estudiante_seleccionado][0]

    generador = np.random.default_rng(seed=int(indice))

    aportes = pd.DataFrame({
        "Variable": [
            "Estrato socioeconómico",
            "Acceso a Internet",
            "Disponibilidad de computador"
        ],
        "Contribución": generador.normal(0, 8, 3)
    })

    aportes["Tipo"] = np.where(aportes["Contribución"] >= 0,"Aumenta","Disminuye")

    # Crear gráfica de contribuciones
    fig = px.bar(
        aportes,
        x="Contribución",
        y="Variable",
        orientation="h",
        color="Tipo",
        color_discrete_map={
            "Aumenta": "#5CB85C",
            "Disminuye": "#D9534F"
        },
        title="Contribución de las variables a la predicción (simulada)"
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Contribución simulada",
        yaxis_title="",
        template="plotly_white",

        font=dict(
            family="Segoe UI, Arial, sans-serif",
            color="#304F70",
            size=16
        ),

        title_font=dict(
            family="Segoe UI, Arial, sans-serif",
            size=25,
            color="#304F70",
            weight="bold"
        ),

        title_x=0.5,
        title_xanchor="center"
    )


        # Crear tabla de contribuciones
    tabla = html.Table([

        html.Thead(html.Tr([html.Th("Variable"),html.Th("Contribución"),html.Th("Interpretación")])
        ),

        html.Tbody([html.Tr([html.Td(fila["Variable"]),html.Td(round(fila["Contribución"], 2)),html.Td(fila["Tipo"])])
            for _, fila in aportes.iterrows()
        ])

    ], className="contributions-table")

    return round(puntaje, 2), estado, fig, tabla

if __name__ == "__main__":
    app.run(debug=True)
