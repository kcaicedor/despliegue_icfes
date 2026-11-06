
# INDICADORES SIMULADOS

cantidad_registros = "25.2K"
Puntaje_promedio_estimado = 252.5
Porcentaje_con_puntaje = 65.8
MAE= 22.5
RMSE= 31.8
R2 = 0.78

# IMPORTACIONES

from dash import Dash, html
import plotly.express as px
import pandas as pd

# CREAR LA APLICACIÓN

app = Dash(__name__)

# FUNCIÓN PARA CREAR TARJETAS

def crear_tarjeta(titulo, valor, identificador):

    return html.Div([

        html.P(
            titulo,
            className="card-title"
        ),

        html.H3(
            valor,
            className="card-value"
        )

    ], className="indicator-card", id=identificador)

# DEFINIR EL DISEÑO DEL TABLERO

app.layout = html.Div([

    # Encabezado principal
    html.Div([

        # Espacio para el logo
        html.Div(
            "ICFES",
            className="logo"
        ),

        # Título del tablero
        html.Div(
            "MÓDULO PREDICTIVO: DESEMPEÑO ACADÉMICO PRUEBAS SABER 11",
            className="header-title"
        )

    ], className="header"),


    # Contenido principal
    html.Div([

        # Barra de herramientas
        html.Div([

            # Botón para cargar archivos
            html.Button(
                "Sube la base (.csv o .xlsx)",
                className="btn-upload"
            ),

            # Botón para descargar el reporte
            html.Button(
                "Descargar reporte con predicción",
                className="btn-download"
            )

        ], className="toolbar"),


        # Tarjetas de indicadores
        html.Div([

            crear_tarjeta(
                "Cantidad de registros",
                cantidad_registros,
                "card-registros"
            ),

            crear_tarjeta(
                "Puntaje promedio estimado",
                Puntaje_promedio_estimado,
                "card-promedio"
            ),

            crear_tarjeta(
                "Porcentaje con puntaje ≥ 300 (%)",
                f"{Porcentaje_con_puntaje}%",
                "card-300"
            ),

            crear_tarjeta(
                "Error absoluto medio - MAE",
                MAE,
                "card-mae"
            ),

            crear_tarjeta(
                "Raíz del error cuadrático medio - RMSE",
                RMSE,
                "card-rmse"
            ),
            crear_tarjeta(
                "Coeficiente de determinación - R²",
                R2,
                "card-r2"
            ),

        ], className="indicators-container"),


        # Texto descriptivo
        html.P(
            "Tablero de análisis del desempeño académico",
            className="subtitle"
        )

    ], className="main-content")

])

# EJECUTAR LA APLICACIÓN

if __name__ == "__main__":
    app.run(debug=True)
