"""
charts.py — Graficos reutilizables con Plotly (Fase 9)

Todas las funciones devuelven un objeto plotly.graph_objects.Figure
listo para renderizar con st.plotly_chart() en el dashboard.

Funciones publicas:
  grafico_historico(df, banco)
      Serie historica mensual del tipo oficial (2000-2025).

  grafico_prediccion(historico, predicciones, banco, escenario)
      Historico (linea solida) + prediccion futura 2026-2030 (linea discontinua).

  grafico_escenarios(historico, pred_base, pred_opt, pred_pes, banco)
      Tres escenarios superpuestos sobre el historico reciente.

  grafico_probabilidades(tabla_scores, banco, escenario)
      Area apilada: P_Baja / P_Estable / P_Sube para el periodo predicho.

  grafico_scores(tabla_scores, banco, escenario)
      Confidence_Score y Net_Score a lo largo del horizonte de prediccion.

  grafico_importancia(importancias, banco)
      Barras horizontales con importancia de cada feature (Random Forest).

  grafico_confusion_matrix(cm_df, banco, conjunto)
      Heatmap de la matriz de confusion con anotaciones numericas.

  grafico_metricas_clasificacion(metricas_train, metricas_test, banco)
      Barras comparativas train vs test: Accuracy, F1_macro, F1_weighted.

  grafico_tres_bancos(df_bce, df_boe, df_fed)
      Tipo oficial historico de los tres bancos en un mismo grafico.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# ---------------------------------------------------------------------------
# Paleta de colores auxiliar
# ---------------------------------------------------------------------------
_COLOR_BASE      = "#1f77b4"   # azul Plotly por defecto
_COLOR_OPTIMISTA = "#2ca02c"   # verde
_COLOR_PESIMISTA = "#d62728"   # rojo
_COLOR_PRED      = "#ff7f0e"   # naranja (linea prediccion)
_COLOR_TRAIN     = "rgba(180,220,255,0.25)"  # fondo region train
_COLOR_TEST      = "rgba(255,220,180,0.25)"  # fondo region test
_COLOR_FUTURO    = "rgba(220,255,220,0.20)"  # fondo region futura

_LAYOUT_BASE = dict(
    template   = "plotly_white",
    font       = dict(family="Arial, sans-serif", size=12),
    legend     = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin     = dict(l=50, r=30, t=60, b=50),
    hovermode  = "x unified",
)


def _add_vline_labeled(fig, x: str, label: str, color: str = "gray",
                       dash: str = "dot", secondary_y: bool = False) -> None:
    """
    Workaround para Plotly >= 6.x: add_vline con annotation_text provoca
    TypeError cuando x es un string de fecha. Se añaden por separado la
    linea (add_shape) y la anotacion (add_annotation).
    """
    fig.add_shape(
        type      = "line",
        x0=x, x1=x, y0=0, y1=1,
        xref      = "x",
        yref      = "paper",
        line      = dict(color=color, dash=dash, width=1.2),
    )
    fig.add_annotation(
        x         = x,
        y         = 1.02,
        xref      = "x",
        yref      = "paper",
        text      = label,
        showarrow = False,
        font      = dict(size=10, color=color),
        xanchor   = "left",
    )


# ---------------------------------------------------------------------------
# 1. Serie historica del tipo oficial
# ---------------------------------------------------------------------------

def grafico_historico(df: pd.DataFrame, banco: str) -> go.Figure:
    """
    Grafico de linea del tipo de interes oficial mensual (2000-2025).

    Parametros:
        df:    DataFrame procesado con columna 'tipo_oficial' e indice DatetimeIndex.
               Habitualmente la salida de data_loader.cargar_datos_procesados(banco).
        banco: "BCE", "BoE" o "FED".

    Devuelve:
        go.Figure con una traza de linea coloreada segun config.COLORES[banco].
    """
    color = config.COLORES.get(banco, _COLOR_BASE)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x    = df.index,
        y    = df["tipo_oficial"],
        mode = "lines",
        name = f"Tipo oficial {banco}",
        line = dict(color=color, width=2),
        hovertemplate = "%{x|%Y-%m}<br>Tipo: %{y:.2f}%<extra></extra>",
    ))

    # Linea vertical en el inicio del test
    _add_vline_labeled(fig, config.TEST_START, "Inicio test")

    fig.update_layout(
        **_LAYOUT_BASE,
        title = f"Tipo de interes oficial — {banco} (historico mensual)",
        xaxis_title = "Fecha",
        yaxis_title = "Tipo oficial (%)",
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Historico + prediccion futura (un escenario)
# ---------------------------------------------------------------------------

def grafico_prediccion(
    historico: pd.DataFrame,
    predicciones: pd.DataFrame,
    banco: str,
    escenario: str = "Base",
    margen_ci=None,
) -> go.Figure:
    """
    Grafico combinado: serie historica (linea solida) + prediccion futura
    2026-2030 (linea discontinua con banda de confianza).

    Parametros:
        historico:    DataFrame con 'tipo_oficial' e indice DatetimeIndex.
        predicciones: DataFrame devuelto por predecir_escenario() con columnas
                      'tipo_predicho', 'Confidence_Score', 'Net_Score', etc.
        banco:        "BCE", "BoE" o "FED".
        escenario:    "Base", "Optimista" o "Pesimista".
        margen_ci:    float o None. Semiancho de la banda de confianza (pp).
                      Si se pasa, se usa el MAE medio walk-forward real.
                      Si es None, se usan los valores por defecto calibrados.

    Devuelve:
        go.Figure con historico + prediccion + banda de confianza ±MAE.
    """
    color   = config.COLORES.get(banco, _COLOR_BASE)
    colores_esc = {"Base": _COLOR_BASE, "Optimista": _COLOR_OPTIMISTA, "Pesimista": _COLOR_PESIMISTA}
    color_pred = colores_esc.get(escenario, _COLOR_PRED)

    # Banda de confianza: ±MAE medio walk-forward (si se pasa) o valores por defecto.
    # Valores por defecto calibrados con walk-forward 3 ventanas:
    #   BCE=0.155pp, BoE=0.184pp, FED=0.203pp (MAE medio de las 3 ventanas)
    mae_default = {"BCE": 0.155, "BoE": 0.184, "FED": 0.203}
    margen = float(margen_ci) if margen_ci is not None else mae_default.get(banco, 0.20)

    y_pred  = predicciones["tipo_predicho"]
    y_upper = y_pred + margen
    y_lower = y_pred - margen

    fig = go.Figure()

    # Region sombreada: periodo de test
    fig.add_vrect(
        x0        = config.TEST_START,
        x1        = config.TEST_END,
        fillcolor = _COLOR_TEST,
        layer     = "below",
        line_width= 0,
    )
    fig.add_annotation(x=config.TEST_START, y=1.04, xref="x", yref="paper",
                       text="Test", showarrow=False, font=dict(size=10, color="gray"), xanchor="left")

    # Region sombreada: prediccion futura
    fig.add_vrect(
        x0        = str(config.PREDICT_START) + "-01-01",
        x1        = str(config.PREDICT_END)   + "-12-31",
        fillcolor = _COLOR_FUTURO,
        layer     = "below",
        line_width= 0,
    )
    fig.add_annotation(x=str(config.PREDICT_START) + "-01-01", y=1.04, xref="x", yref="paper",
                       text="Prediccion", showarrow=False, font=dict(size=10, color="gray"), xanchor="left")

    # Linea historica
    fig.add_trace(go.Scatter(
        x    = historico.index,
        y    = historico["tipo_oficial"],
        mode = "lines",
        name = "Historico",
        line = dict(color=color, width=2),
        hovertemplate = "%{x|%Y-%m}<br>Real: %{y:.2f}%<extra></extra>",
    ))

    # Banda de confianza ±MAE walk-forward (relleno semitransparente)
    # Interpretacion: el modelo historicamente comete un error medio de 'margen' pp
    # → las predicciones futuras caben con alta probabilidad en esta banda.
    fig.add_trace(go.Scatter(
        x    = list(predicciones.index) + list(predicciones.index[::-1]),
        y    = list(y_upper) + list(y_lower[::-1]),
        fill = "toself",
        fillcolor = "rgba(255,140,0,0.15)",
        line = dict(color="rgba(0,0,0,0)"),
        showlegend = True,
        hoverinfo  = "skip",
        name = f"IC ±{margen:.3f}pp (MAE walk-forward)",
    ))

    # Linea de prediccion futura
    fig.add_trace(go.Scatter(
        x    = predicciones.index,
        y    = y_pred,
        mode = "lines+markers",
        name = f"Prediccion {escenario}",
        line = dict(color=color_pred, width=2.5, dash="dash"),
        marker = dict(size=5, symbol="circle"),
        hovertemplate = "%{x|%Y-%m}<br>Prediccion: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = (f"Tipo de interes — {banco} | Historico + Prediccion ({escenario})"),
        xaxis_title = "Fecha",
        yaxis_title = "Tipo oficial (%)",
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Tres escenarios superpuestos
# ---------------------------------------------------------------------------

def grafico_escenarios(
    historico: pd.DataFrame,
    pred_base: pd.DataFrame,
    pred_opt:  pd.DataFrame,
    pred_pes:  pd.DataFrame,
    banco: str,
) -> go.Figure:
    """
    Superpone los tres escenarios (Base, Optimista, Pesimista) sobre el
    historico reciente (ultimos 3 anos) para facilitar la comparacion.

    Parametros:
        historico: DataFrame con 'tipo_oficial' completo.
        pred_base/opt/pes: DataFrames de predicciones por escenario.
        banco: "BCE", "BoE" o "FED".

    Devuelve:
        go.Figure con 4 trazas: historico + 3 escenarios.
    """
    color = config.COLORES.get(banco, _COLOR_BASE)

    # Solo historico reciente (ultimos 4 anos) para no sobrecargar el grafico
    inicio_reciente = pd.Timestamp("2022-01-01")
    hist_reciente   = historico.loc[historico.index >= inicio_reciente]

    fig = go.Figure()

    # Region prediccion
    fig.add_vrect(
        x0         = str(config.PREDICT_START) + "-01-01",
        x1         = str(config.PREDICT_END)   + "-12-31",
        fillcolor  = _COLOR_FUTURO,
        layer      = "below",
        line_width = 0,
    )

    # C2 — Banda de incertidumbre macro: relleno entre Optimista y Pesimista.
    # Representa el rango de variacion del tipo predicho segun las hipotesis
    # macroeconomicas alternativas — analogia al fan-chart de los bancos centrales.
    _idx_fill = list(pred_opt.index) + list(pred_pes.index[::-1])
    _y_fill   = (
        list(pred_opt["tipo_predicho"])
        + list(pred_pes["tipo_predicho"][::-1])
    )
    fig.add_trace(go.Scatter(
        x          = _idx_fill,
        y          = _y_fill,
        fill       = "toself",
        fillcolor  = "rgba(150,150,150,0.18)",
        line       = dict(color="rgba(0,0,0,0)"),
        showlegend = True,
        name       = "Rango de incertidumbre macro (Opt-Pes)",
        hoverinfo  = "skip",
    ))

    # Historico reciente
    fig.add_trace(go.Scatter(
        x    = hist_reciente.index,
        y    = hist_reciente["tipo_oficial"],
        mode = "lines",
        name = "Historico",
        line = dict(color=color, width=2.5),
        hovertemplate = "%{x|%Y-%m}<br>Real: %{y:.2f}%<extra></extra>",
    ))

    # Escenario Base
    fig.add_trace(go.Scatter(
        x    = pred_base.index,
        y    = pred_base["tipo_predicho"],
        mode = "lines",
        name = "Base",
        line = dict(color=_COLOR_BASE, width=2, dash="dash"),
        hovertemplate = "%{x|%Y-%m}<br>Base: %{y:.2f}%<extra></extra>",
    ))

    # Escenario Optimista
    fig.add_trace(go.Scatter(
        x    = pred_opt.index,
        y    = pred_opt["tipo_predicho"],
        mode = "lines",
        name = "Optimista",
        line = dict(color=_COLOR_OPTIMISTA, width=2, dash="dot"),
        hovertemplate = "%{x|%Y-%m}<br>Optimista: %{y:.2f}%<extra></extra>",
    ))

    # Escenario Pesimista
    fig.add_trace(go.Scatter(
        x    = pred_pes.index,
        y    = pred_pes["tipo_predicho"],
        mode = "lines",
        name = "Pesimista",
        line = dict(color=_COLOR_PESIMISTA, width=2, dash="dashdot"),
        hovertemplate = "%{x|%Y-%m}<br>Pesimista: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title = f"Comparativa de escenarios — {banco} (2022-2030)",
        xaxis_title = "Fecha",
        yaxis_title = "Tipo oficial (%)",
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Probabilidades por clase (area apilada)
# ---------------------------------------------------------------------------

def grafico_probabilidades(
    tabla_scores: pd.DataFrame,
    banco: str,
    escenario: str = "Base",
) -> go.Figure:
    """
    Area apilada con P_Baja, P_Estable, P_Sube para el horizonte predicho.

    Parametros:
        tabla_scores: DataFrame devuelto por predecir_escenario() con columnas
                      'P_Baja', 'P_Estable', 'P_Sube' en porcentaje (0-100).
        banco:        "BCE", "BoE" o "FED".
        escenario:    nombre del escenario para el titulo.

    Devuelve:
        go.Figure con tres areas apiladas coloreadas semanticamente.
    """
    fig = go.Figure()

    # P_Sube (verde)
    fig.add_trace(go.Scatter(
        x         = tabla_scores.index,
        y         = tabla_scores["P_Sube"],
        mode      = "lines",
        name      = "P(Sube)",
        stackgroup = "one",
        fillcolor = "rgba(44,160,44,0.6)",
        line      = dict(color="rgba(44,160,44,0.8)", width=1),
        hovertemplate = "%{x|%Y-%m}<br>P(Sube): %{y:.1f}%<extra></extra>",
    ))

    # P_Estable (gris)
    fig.add_trace(go.Scatter(
        x         = tabla_scores.index,
        y         = tabla_scores["P_Estable"],
        mode      = "lines",
        name      = "P(Estable)",
        stackgroup = "one",
        fillcolor = "rgba(150,150,150,0.5)",
        line      = dict(color="rgba(120,120,120,0.8)", width=1),
        hovertemplate = "%{x|%Y-%m}<br>P(Estable): %{y:.1f}%<extra></extra>",
    ))

    # P_Baja (rojo)
    fig.add_trace(go.Scatter(
        x         = tabla_scores.index,
        y         = tabla_scores["P_Baja"],
        mode      = "lines",
        name      = "P(Baja)",
        stackgroup = "one",
        fillcolor = "rgba(214,39,40,0.6)",
        line      = dict(color="rgba(190,30,30,0.8)", width=1),
        hovertemplate = "%{x|%Y-%m}<br>P(Baja): %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title       = f"Distribucion de probabilidades — {banco} | {escenario}",
        xaxis_title = "Fecha",
        yaxis_title = "Probabilidad (%)",
        yaxis       = dict(range=[0, 100]),
    )

    return fig


# ---------------------------------------------------------------------------
# 5. Confidence Score y Net Score
# ---------------------------------------------------------------------------

def grafico_scores(
    tabla_scores: pd.DataFrame,
    banco: str,
    escenario: str = "Base",
) -> go.Figure:
    """
    Grafico de doble eje: Confidence_Score (barra) y Net_Score (linea).

    Parametros:
        tabla_scores: DataFrame con 'Confidence_Score' y 'Net_Score'.
        banco:        "BCE", "BoE" o "FED".
        escenario:    nombre del escenario.

    Devuelve:
        go.Figure con subplots de dos ejes Y.
    """
    color = config.COLORES.get(banco, _COLOR_BASE)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Confidence Score — barras
    fig.add_trace(
        go.Bar(
            x         = tabla_scores.index,
            y         = tabla_scores["Confidence_Score"],
            name      = "Confidence Score",
            marker_color = "rgba(100,149,237,0.6)",
            hovertemplate = "%{x|%Y-%m}<br>Confidence: %{y:.1f}<extra></extra>",
        ),
        secondary_y = False,
    )

    # Net Score — linea
    net = tabla_scores["Net_Score"]
    colores_net = ["rgba(44,160,44,0.8)" if v >= 0 else "rgba(214,39,40,0.8)" for v in net]

    fig.add_trace(
        go.Scatter(
            x         = tabla_scores.index,
            y         = net,
            mode      = "lines+markers",
            name      = "Net Score",
            line      = dict(color=color, width=2),
            marker    = dict(
                color = ["green" if v >= 0 else "red" for v in net],
                size  = 7,
            ),
            hovertemplate = "%{x|%Y-%m}<br>Net Score: %{y:.1f}<extra></extra>",
        ),
        secondary_y = True,
    )

    # Linea de cero en Net Score
    fig.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=True)

    fig.update_layout(
        **_LAYOUT_BASE,
        title = f"Scores del modelo — {banco} | {escenario}",
    )
    fig.update_yaxes(title_text="Confidence Score (0-100)", secondary_y=False, range=[0, 100])
    fig.update_yaxes(title_text="Net Score (-100 a +100)",  secondary_y=True)
    fig.update_xaxes(title_text="Fecha")

    return fig


# ---------------------------------------------------------------------------
# 6. Importancia de features (Random Forest)
# ---------------------------------------------------------------------------

def grafico_importancia(
    importancias: pd.Series,
    banco: str,
) -> go.Figure:
    """
    Barras horizontales con la importancia de cada feature del Random Forest.

    Parametros:
        importancias: pd.Series con indice = nombres de features y
                      valores = importancia (suma = 1.0).
                      Normalmente la salida de regression.importancia_features().
        banco:        "BCE", "BoE" o "FED".

    Devuelve:
        go.Figure ordenado de mayor a menor importancia.
    """
    color = config.COLORES.get(banco, _COLOR_BASE)

    imp_sorted = importancias.sort_values(ascending=True)
    pct = (imp_sorted * 100).round(2)

    # Escala de color: mas importante = mas oscuro
    max_imp = pct.max()
    colores = [
        f"rgba({int(31 + (1 - v/max_imp)*180)}, "
        f"{int(119 + (1 - v/max_imp)*100)}, "
        f"180, 0.85)"
        for v in pct.values
    ]

    fig = go.Figure(go.Bar(
        x           = pct.values,
        y           = pct.index,
        orientation = "h",
        marker_color = colores,
        text        = [f"{v:.2f}%" for v in pct.values],
        textposition = "outside",
        hovertemplate = "<b>%{y}</b><br>Importancia: %{x:.2f}%<extra></extra>",
    ))

    layout = {**_LAYOUT_BASE, **dict(
        title       = f"Importancia de variables — {banco} (Random Forest)",
        xaxis_title = "Importancia (%)",
        yaxis_title = "",
        xaxis       = dict(range=[0, max_imp * 1.15]),
        showlegend  = False,
        hovermode   = "y unified",
    )}
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# 7. Matriz de confusion (heatmap)
# ---------------------------------------------------------------------------

def grafico_confusion_matrix(
    cm_df: pd.DataFrame,
    banco: str,
    conjunto: str = "test",
) -> go.Figure:
    """
    Heatmap de la matriz de confusion con anotaciones numericas.

    Parametros:
        cm_df:    DataFrame (3x3) devuelto por classification.matriz_confusion().
                  Filas = clase real, columnas = clase predicha.
        banco:    "BCE", "BoE" o "FED".
        conjunto: "train" o "test" (para el titulo).

    Devuelve:
        go.Figure con heatmap coloreado y texto en cada celda.
    """
    z     = cm_df.values.astype(float)
    total = z.sum()
    z_pct = z / total * 100 if total > 0 else z

    clases_real = [c.replace("Real_", "") for c in cm_df.index]
    clases_pred = [c.replace("Pred_", "") for c in cm_df.columns]

    # Texto en cada celda: N (X%)
    texto = [
        [f"{int(z[i,j])}<br>({z_pct[i,j]:.1f}%)" for j in range(len(clases_pred))]
        for i in range(len(clases_real))
    ]

    fig = go.Figure(go.Heatmap(
        z          = z,
        x          = clases_pred,
        y          = clases_real,
        text       = texto,
        texttemplate = "%{text}",
        textfont   = dict(size=13),
        colorscale = "Blues",
        showscale  = True,
        hovertemplate = "Real: %{y}<br>Pred: %{x}<br>N: %{z}<extra></extra>",
    ))

    layout = {**_LAYOUT_BASE, **dict(
        title       = f"Matriz de confusion — {banco} ({conjunto})",
        xaxis_title = "Clase predicha",
        yaxis_title = "Clase real",
        hovermode   = "closest",
    )}
    fig.update_layout(**layout)

    return fig


# ---------------------------------------------------------------------------
# 8. Metricas de clasificacion: train vs test
# ---------------------------------------------------------------------------

def grafico_metricas_clasificacion(
    metricas_train: dict,
    metricas_test: dict,
    banco: str,
) -> go.Figure:
    """
    Barras agrupadas con Accuracy, F1_macro y F1_weighted (train y test).
    Incluye linea de referencia del clasificador trivial ("siempre Estable").

    Parametros:
        metricas_train: dict devuelto por evaluar_clasificacion(..., conjunto="train").
        metricas_test:  dict devuelto por evaluar_clasificacion(..., conjunto="test").
        banco:          "BCE", "BoE" o "FED".

    Devuelve:
        go.Figure con barras agrupadas y linea de baseline.
    """
    metricas_etiquetas = ["Accuracy", "F1_macro", "F1_weighted"]
    baseline_trivial = {"BCE": 0.625, "BoE": 0.792, "FED": 0.667}
    baseline = baseline_trivial.get(banco, 0.65)

    vals_train = [metricas_train.get(m, 0) for m in metricas_etiquetas]
    vals_test  = [metricas_test.get(m, 0)  for m in metricas_etiquetas]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name         = "Train",
        x            = metricas_etiquetas,
        y            = vals_train,
        marker_color = "rgba(100,149,237,0.75)",
        text         = [f"{v:.3f}" for v in vals_train],
        textposition = "outside",
    ))

    fig.add_trace(go.Bar(
        name         = "Test",
        x            = metricas_etiquetas,
        y            = vals_test,
        marker_color = config.COLORES.get(banco, _COLOR_BASE),
        opacity      = 0.85,
        text         = [f"{v:.3f}" for v in vals_test],
        textposition = "outside",
    ))

    # Linea de baseline trivial
    fig.add_hline(
        y               = baseline,
        line_dash       = "dot",
        line_color      = "gray",
        annotation_text = f"Baseline trivial ({baseline:.1%})",
        annotation_position = "bottom right",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        title       = f"Metricas de clasificacion — {banco}",
        xaxis_title = "Metrica",
        yaxis_title = "Valor",
        yaxis       = dict(range=[0, 1.15]),
        barmode     = "group",
    )

    return fig


# ---------------------------------------------------------------------------
# 9. Comparativa de los tres bancos (historico)
# ---------------------------------------------------------------------------

def grafico_tres_bancos(
    df_bce: pd.DataFrame,
    df_boe: pd.DataFrame,
    df_fed: pd.DataFrame,
) -> go.Figure:
    """
    Serie historica de los tres bancos centrales en un mismo grafico.
    Util para la pantalla de inicio del dashboard.

    Parametros:
        df_bce/boe/fed: DataFrames procesados con 'tipo_oficial' e indice fecha.

    Devuelve:
        go.Figure con tres trazas de linea, una por banco.
    """
    fig = go.Figure()

    for banco, df in [("BCE", df_bce), ("BoE", df_boe), ("FED", df_fed)]:
        color = config.COLORES[banco]
        fig.add_trace(go.Scatter(
            x    = df.index,
            y    = df["tipo_oficial"],
            mode = "lines",
            name = banco,
            line = dict(color=color, width=2),
            hovertemplate = f"{banco} %{{x|%Y-%m}}: %{{y:.2f}}%<extra></extra>",
        ))

    # Lineas de referencia de eventos clave
    eventos = [
        ("2008-09-01", "Crisis financiera"),
        ("2020-03-01", "COVID-19"),
        ("2022-06-01", "Ciclo subidas"),
        ("2024-01-01", "Inicio test"),
    ]
    for fecha, label in eventos:
        _add_vline_labeled(fig, fecha, label, color="rgba(120,120,120,0.7)")

    fig.update_layout(
        **_LAYOUT_BASE,
        title       = "Tipos de interes oficiales: BCE, BoE y FED (2000-2025)",
        xaxis_title = "Fecha",
        yaxis_title = "Tipo oficial (%)",
    )

    return fig
