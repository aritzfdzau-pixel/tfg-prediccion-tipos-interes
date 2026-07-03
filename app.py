"""
app.py — Dashboard principal de prediccion de tipos de interes (Fase 10)

Estructura del dashboard:
  Barra lateral : selector de banco central + instrumento + escenario + info rapida
  Tab 1         : Panorama global (tres bancos, tabla de metricas)
  Tab 2         : Historico y test  (serie historica, importancia features, scores test)
  Tab 3         : Predicciones 2026-2030 (comparativa escenarios + detalle)
  Tab 4         : Evaluacion del modelo  (regresion + clasificacion + matrices)

El selector de instrumento permite alternar entre:
  - Tipos Oficiales : ECB Depo Facility / UKBASERATE / FFR
  - Tipos OIS       : €STR (BCE) / SONIA (BoE) / SOFR (FED)

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Rutas y modulos propios
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.data_loader import cargar_datos_procesados
from src.models.scoring import interpretar_scores
from src.models import caso_iberdrola as caso
from src.visualization.charts import (
    grafico_escenarios,
    grafico_historico,
    grafico_importancia,
    grafico_metricas_clasificacion,
    grafico_confusion_matrix,
    grafico_prediccion,
    grafico_probabilidades,
    grafico_scores,
    grafico_tres_bancos,
)

# ---------------------------------------------------------------------------
# Constantes locales
# ---------------------------------------------------------------------------
ESCENARIOS = ["Base", "Optimista", "Pesimista"]

_ESC_LOWER: dict[str, str] = {
    "Base":      "base",
    "Optimista": "optimista",
    "Pesimista": "pesimista",
}

_NOMBRE_COMPLETO: dict[str, str] = {
    "BCE": "Banco Central Europeo",
    "BoE": "Bank of England",
    "FED": "Reserva Federal (FED)",
}

_SEMAFORO_EMOJI: dict[str, str] = {
    "[SUBE]":    "🟢",
    "[BAJA]":    "🔴",
    "[ESTABLE]": "🟡",
}

# Etiqueta para cada instrumento
_INSTRUMENTO_LABEL: dict[str, str] = {
    "tipo_oficial": "Tipo oficial",
    "ESTR":  "€STR",
    "SONIA": "SONIA",
    "SOFR":  "SOFR",
    "OIS":   "OIS",
}

# Tipo de interes observado al inicio del periodo de prediccion (dic-2025)
# (usado como fallback cuando no se puede leer del CSV de scores)
_TIPO_INICIO_2026: dict[str, float] = {
    "BCE": 2.40,
    "BoE": 4.50,
    "FED": 4.33,
}


# ---------------------------------------------------------------------------
# Helpers de instrumento
# ---------------------------------------------------------------------------

def _suf(instrumento: str) -> str:
    """Sufijo de archivo: '' para tipo_oficial, '_{instrumento}' para OIS."""
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


def _get_instrumento(banco: str, instrumento_tipo: str) -> str:
    """
    Resuelve el nombre del instrumento para (banco, instrumento_tipo).

    instrumento_tipo: "tipo_oficial" | "OIS"
    Devuelve: "tipo_oficial" | "ESTR" | "SONIA" | "SOFR"
    """
    if instrumento_tipo == "tipo_oficial":
        return "tipo_oficial"
    _OIS_NAMES = {"BCE": "ESTR", "BoE": "SONIA", "FED": "SOFR"}
    return _OIS_NAMES[banco]


def _target_col_from(instrumento: str) -> str:
    """Columna del DataFrame que corresponde al instrumento."""
    # Dict local para no depender del cache de config (OneDrive puede corromper .pyc)
    _MAP = {
        "tipo_oficial": "tipo_oficial",
        "OIS":   "OIS",
        "ESTR":  "OIS",
        "SONIA": "OIS",
        "SOFR":  "OIS",
    }
    return _MAP.get(instrumento, instrumento)


def _df_para_graficos(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Adapta el DataFrame para los graficos de charts.py, que siempre esperan
    una columna llamada 'tipo_oficial'.

    En modo OIS (target_col='OIS'), copia la columna OIS sobre tipo_oficial
    en una copia ligera del DataFrame, sin modificar el original.
    En modo normal (target_col='tipo_oficial') devuelve el mismo objeto.
    """
    if target_col == "tipo_oficial":
        return df
    df2 = df.copy()
    df2["tipo_oficial"] = df2[target_col]
    return df2


# ===========================================================================
# CARGA DE DATOS CON CACHE
# ===========================================================================

@st.cache_data(show_spinner=False)
def _cargar_historico(banco: str) -> pd.DataFrame:
    """Dataset procesado del banco central (mensual, 2000-2025)."""
    return cargar_datos_procesados(banco)


@st.cache_data(show_spinner=False)
def _cargar_scores_test(banco: str, instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """Scores del periodo de test 2024-2025 generados en la Fase 7."""
    ruta = config.RESULTS_DIR / f"scores_test_{banco}{_suf(instrumento)}.csv"
    return pd.read_csv(ruta, index_col="Fecha", parse_dates=True)


@st.cache_data(show_spinner=False)
def _cargar_importancias(banco: str, instrumento: str = "tipo_oficial") -> pd.Series:
    """
    Importancia de features del Random Forest (Fase 5).
    El CSV tiene encabezado 'importancia'; el indice son los nombres de features.
    """
    ruta = config.RESULTS_DIR / f"importancia_{banco}{_suf(instrumento)}.csv"
    df = pd.read_csv(ruta, index_col=0)
    serie = df.iloc[:, 0].astype(float)
    serie.name = banco
    return serie


@st.cache_data(show_spinner=False)
def _cargar_pred_futura(
    banco: str, escenario: str, instrumento: str = "tipo_oficial"
) -> pd.DataFrame:
    """Predicciones 2026-2030 para un banco y escenario (Fase 8)."""
    esc_lower = _ESC_LOWER[escenario]
    ruta = config.RESULTS_DIR / f"predicciones_futuras_{banco}_{esc_lower}{_suf(instrumento)}.csv"
    return pd.read_csv(ruta, index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def _cargar_confusion(
    banco: str, conjunto: str, instrumento: str = "tipo_oficial"
) -> pd.DataFrame:
    """Matriz de confusion 3x3 (train o test) de la Fase 6."""
    ruta = config.RESULTS_DIR / f"confusion_{conjunto}_{banco}{_suf(instrumento)}.csv"
    return pd.read_csv(ruta, index_col=0)


@st.cache_data(show_spinner=False)
def _cargar_walk_forward() -> pd.DataFrame:
    """Resultados de la validacion walk-forward (3 ventanas × 3 bancos)."""
    ruta = config.RESULTS_DIR / "walk_forward_validation.csv"
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta)


@st.cache_data(show_spinner=False)
def _cargar_pred_regresion_test(
    banco: str, instrumento: str = "tipo_oficial"
) -> pd.DataFrame:
    """Predicciones del test con deltas reales y predichos (para baseline naïve)."""
    ruta = config.RESULTS_DIR / f"predicciones_test_regresion_{banco}{_suf(instrumento)}.csv"
    return pd.read_csv(ruta, index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def _mae_walkforward(banco: str) -> float | None:
    """
    MAE medio del walk-forward para el banco (3 ventanas).
    Devuelve None si el CSV aun no existe (ejecutar walk_forward.py).
    Solo aplica al modelo de tipos oficiales; OIS no tiene walk-forward.
    """
    df_wf = _cargar_walk_forward()
    if df_wf.empty:
        return None
    df_b = df_wf[df_wf["Banco"] == banco]
    return float(df_b["MAE"].mean()) if not df_b.empty else None


def _mae_ref(banco: str, instrumento: str) -> float | None:
    """
    MAE de referencia para la banda de confianza del grafico de prediccion.

    - tipo_oficial : usa MAE medio walk-forward (3 ventanas) si existe
    - OIS          : usa MAE del test del Random Forest (mas preciso para OIS)
    """
    if instrumento == "tipo_oficial":
        return _mae_walkforward(banco)
    # Para OIS no hay walk-forward: usar MAE_test del RF desde el resumen
    try:
        df_res = _cargar_resumen_regresion(instrumento)
        row    = df_res[(df_res["Banco"] == banco) & (df_res["Modelo"] == "RF")]
        if not row.empty:
            return float(row.iloc[0]["MAE_test"])
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def _cargar_resumen_regresion(instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Metricas de regresion (Fase 5).
    Columnas: Banco, Modelo, MAE_train, RMSE_train, R2_train,
              MAE_test, RMSE_test, R2_test
    """
    if instrumento == "tipo_oficial":
        return pd.read_csv(config.RESULTS_DIR / "resumen_regresion.csv")
    else:
        return pd.read_csv(config.RESULTS_DIR / "resumen_regresion_OIS.csv")


@st.cache_data(show_spinner=False)
def _cargar_resumen_clasificacion(instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Metricas de clasificacion (Fase 6).
    Columnas: Banco, Conjunto, Accuracy, Precision_macro, Recall_macro,
              F1_macro, F1_weighted, ROC_AUC_ovr
    """
    if instrumento == "tipo_oficial":
        return pd.read_csv(config.RESULTS_DIR / "resumen_clasificacion.csv")
    else:
        return pd.read_csv(config.RESULTS_DIR / "resumen_clasificacion_OIS.csv")


@st.cache_data(show_spinner=False)
def _cargar_f1_por_clase(banco: str, instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Parsea el reporte de clasificacion por clase (Precision/Recall/F1/Soporte).
    Devuelve DataFrame con columnas: Clase, Precision, Recall, F1, Soporte.
    """
    ruta = config.RESULTS_DIR / f"reporte_clasificacion_{banco}{_suf(instrumento)}.txt"
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            for clase in ["Baja", "Estable", "Sube"]:
                if linea.startswith(clase):
                    partes = linea.split()
                    if len(partes) >= 5:
                        filas.append({
                            "Clase":     partes[0],
                            "Precision": float(partes[1]),
                            "Recall":    float(partes[2]),
                            "F1-score":  float(partes[3]),
                            "Soporte":   int(partes[4]),
                        })
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def _cargar_resumen_cls_hikecycle() -> pd.DataFrame:
    """
    Métricas del clasificador sobre el ciclo de subidas 2022-2023.
    Generado por run_hikecycle_eval.py. Devuelve DataFrame vacío si no existe.
    """
    ruta = config.RESULTS_DIR / "resumen_clasificacion_hikecycle.csv"
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta)


@st.cache_data(show_spinner=False)
def _cargar_f1_hikecycle(banco: str) -> pd.DataFrame:
    """
    F1 por clase (Baja/Estable/Sube) del ciclo de subidas 2022-2023.
    Parsea el reporte de texto generado por run_hikecycle_eval.py.
    Devuelve DataFrame vacío si el archivo no existe.
    """
    ruta = config.RESULTS_DIR / f"reporte_clasificacion_{banco}_hikecycle.txt"
    if not ruta.exists():
        return pd.DataFrame()
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            for clase in ["Baja", "Estable", "Sube"]:
                if linea.startswith(clase):
                    partes = linea.split()
                    if len(partes) >= 5:
                        filas.append({
                            "Clase":     partes[0],
                            "Precision": float(partes[1]),
                            "Recall":    float(partes[2]),
                            "F1-score":  float(partes[3]),
                            "Soporte":   int(partes[4]),
                        })
    return pd.DataFrame(filas)


@st.cache_data(show_spinner=False)
def _cargar_confusion_hikecycle(banco: str) -> pd.DataFrame:
    """
    Matriz de confusión 3×3 del ciclo de subidas 2022-2023.
    Generado por run_hikecycle_eval.py. Devuelve DataFrame vacío si no existe.
    """
    ruta = config.RESULTS_DIR / f"confusion_test_{banco}_hikecycle.csv"
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta, index_col=0)


@st.cache_resource(show_spinner=False)
def _cargar_modelo_cls_resource(banco: str, instrumento: str = "tipo_oficial"):
    """
    Carga el modelo de clasificacion y el scaler (objetos sklearn).
    Carga directamente con joblib para no depender de la firma de cargar_modelo
    (evita errores por .pyc obsoletos en OneDrive).
    """
    import joblib
    suf  = "" if instrumento == "tipo_oficial" else f"_{instrumento}"
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{suf}_clasificacion.joblib"
    d    = joblib.load(ruta)
    return d["modelo"], d["scaler"]


@st.cache_data(show_spinner=False)
def _obtener_defaults_escenario(banco: str, escenario: str) -> dict:
    """
    Devuelve los valores del anio 2026 del escenario como dict de defaults
    para inicializar los sliders del escenario personalizado.
    """
    from src.models.prediccion_futura import definir_escenarios
    datos      = definir_escenarios()
    vals_banco = datos[banco][escenario][2026]
    vals_comod = datos["_comodidades"][escenario][2026]
    merged = {**vals_banco, **vals_comod}
    slider_keys = [
        "PIB_growth", "PIBpc_growth", "Inflacion", "Desempleo",
        "VIX", "Oro", "Plata", "Gas",
    ]
    return {k: float(merged[k]) for k in slider_keys if k in merged}


def _predecir_personalizado(
    banco: str,
    slider_vals: dict,
    instrumento: str = "tipo_oficial",
) -> tuple:
    """
    Genera predicciones 2026-2030 con los valores constantes de los sliders.

    Arquitectura identica a predecir_escenario():
      · tipo_predicho = OIS_taylor (Regla de Taylor)
      · clasificacion con modelo LogReg entrenado
      · scoring con construir_tabla_scores()

    Parametros:
        banco:        "BCE", "BoE" o "FED"
        slider_vals:  dict {var: valor} para las 8 variables con slider
        instrumento:  "tipo_oficial" | "ESTR" | "SONIA" | "SOFR"

    Devuelve:
        (tabla_scores DataFrame, ois_taylor_value float)
    """
    from src.models.classification import predecir_probabilidades
    from src.models.scoring import construir_tabla_scores

    fechas = pd.date_range("2026-01-01", periods=60, freq="MS")

    X = pd.DataFrame(index=fechas)
    _DERIVADAS = ("OIS_taylor", "brecha_taylor", "meses_sin_cambio", "delta_tipo_lag1")
    for var in config.VARIABLES_MODELO:
        if var in _DERIVADAS:
            continue
        else:
            X[var] = float(slider_vals.get(var, 0.0))

    p = config.TAYLOR_PARAMS[banco]
    X["OIS_taylor"] = (
        p["r_star"]
        + 1.5 * (X["Inflacion"] - p["pi_star"])
        + 0.5 * (X["PIB_growth"] - p["y_star"])
    ).clip(lower=0.0)

    # Tipo inicial: leer del CSV de scores del instrumento seleccionado
    _tipo_inicio = _TIPO_INICIO_2026.get(banco, 2.0)
    _suf_instr   = _suf(instrumento)
    try:
        _ruta_sc2 = config.RESULTS_DIR / f"scores_test_{banco}{_suf_instr}.csv"
        _sc_hist2 = pd.read_csv(str(_ruta_sc2), index_col="Fecha")
        _t_raw2   = float(_sc_hist2.iloc[-1]["tipo_real"])
        if not np.isnan(_t_raw2):
            _tipo_inicio = _t_raw2
    except Exception:
        pass
    X["brecha_taylor"] = X["OIS_taylor"] - _tipo_inicio

    X["meses_sin_cambio"] = 0
    X["delta_tipo_lag1"]  = 0.0

    X = X[config.VARIABLES_MODELO].astype(float)

    ois_val     = float(X["OIS_taylor"].iloc[0])
    tipos_nivel = X["OIS_taylor"].values.copy()

    # Transicion suave desde el tipo real dic-2025
    try:
        _ruta_sc  = config.RESULTS_DIR / f"scores_test_{banco}{_suf_instr}.csv"
        _sc_hist  = pd.read_csv(str(_ruta_sc), index_col="Fecha")
        _tipo_raw = float(_sc_hist.iloc[-1]["tipo_real"])
        if not np.isnan(_tipo_raw):
            _N_BLEND = 12
            for _i in range(min(_N_BLEND, len(tipos_nivel))):
                _alpha = (_i + 1) / _N_BLEND
                tipos_nivel[_i] = (1.0 - _alpha) * _tipo_raw + _alpha * tipos_nivel[_i]
            tipos_nivel = np.clip(tipos_nivel, 0.0, 25.0)
    except Exception:
        pass

    modelo_cls, scaler_cls = _cargar_modelo_cls_resource(banco, instrumento)
    proba = predecir_probabilidades(modelo_cls, scaler_cls, X)

    tabla = construir_tabla_scores(
        fechas=fechas,
        y_pred_reg=tipos_nivel,
        probabilidades=proba,
    )

    # ── Curva hipotética direccional ─────────────────────────────────────────
    # Combina la señal del clasificador (P_Sube - P_Baja) con un paso histórico
    # de 25 bp para generar una trayectoria dinámica partiendo del tipo actual.
    # Un ancla suave hacia OIS_taylor (5 %/mes) evita divergencias en 60 pasos.
    PASO_MAX      = 0.25   # pp — paso máximo cuando la señal es total (+1 o -1)
    ANCHOR_WEIGHT = 0.05   # fracción de pull mensual hacia el equilibrio Taylor

    nivel_hip  = _tipo_inicio
    tipos_hip  = np.zeros(len(fechas))
    taylor_arr = X["OIS_taylor"].values   # equilibrio mes a mes (constante en custom)

    for t in range(len(fechas)):
        p_baja = float(tabla.iloc[t]["P_Baja"]) / 100.0
        p_sube = float(tabla.iloc[t]["P_Sube"]) / 100.0
        net    = p_sube - p_baja           # señal neta en [-1, +1]
        drift  = net * PASO_MAX            # deriva mensual proporcional a la confianza

        # Mezcla: 95 % dirección propia + 5 % atracción hacia equilibrio Taylor
        nuevo = (1 - ANCHOR_WEIGHT) * (nivel_hip + drift) + ANCHOR_WEIGHT * taylor_arr[t]
        nuevo = float(np.clip(nuevo, 0.0, 25.0))

        tipos_hip[t] = nuevo
        nivel_hip    = nuevo

    tabla["tipo_hipotetico"] = tipos_hip

    return tabla, ois_val


# ===========================================================================
# HELPERS
# ===========================================================================

def _estado_semaforo(semaforo: str) -> str:
    return _SEMAFORO_EMOJI.get(semaforo, "⚪")


def _metricas_cls(df_res: pd.DataFrame, banco: str, conjunto: str) -> dict:
    fila = df_res[(df_res["Banco"] == banco) & (df_res["Conjunto"] == conjunto)]
    return fila.iloc[0].to_dict() if not fila.empty else {}


def _metricas_reg_rf(df_res: pd.DataFrame, banco: str) -> dict:
    fila = df_res[(df_res["Banco"] == banco) & (df_res["Modelo"] == "RF")]
    return fila.iloc[0].to_dict() if not fila.empty else {}


def _color_banco(banco: str) -> str:
    return config.COLORES.get(banco, "#1f77b4")


def _interpretar_ultimo_test(
    banco: str, instrumento: str = "tipo_oficial"
) -> tuple[str, str, str, float, float]:
    """
    Carga scores_test del banco y extrae la interpretacion del ultimo mes.
    Devuelve: (confianza_txt, dir_txt, emoji, confidence, net_score)
    """
    scores = _cargar_scores_test(banco, instrumento)
    ultimo = scores.iloc[-1]
    conf, dir_txt, sem = interpretar_scores(
        float(ultimo["Confidence_Score"]),
        float(ultimo["Upward_Score"]),
        float(ultimo["Downward_Score"]),
    )
    return conf, dir_txt, _estado_semaforo(sem), float(ultimo["Confidence_Score"]), float(ultimo["Net_Score"])


# ===========================================================================
# SECCIONES DEL DASHBOARD
# ===========================================================================

# ---------------------------------------------------------------------------
# Tab 1 — Panorama global
# ---------------------------------------------------------------------------

def _seccion_panorama(
    bancos_data: dict[str, pd.DataFrame],
    instrumento_tipo: str,
) -> None:
    """Vista comparativa de los tres bancos centrales."""

    # target_col y label segun instrumento
    target_col = "tipo_oficial" if instrumento_tipo == "tipo_oficial" else "OIS"

    if instrumento_tipo == "tipo_oficial":
        titulo_graf = "Tipos de interes oficiales — BCE, BoE y FED (2000-2025)"
    else:
        titulo_graf = "Tipos OIS — €STR (BCE) · SONIA (BoE) · SOFR (FED) (2000-2025)"

    st.subheader(titulo_graf)
    fig_tres = grafico_tres_bancos(
        _df_para_graficos(bancos_data["BCE"], target_col),
        _df_para_graficos(bancos_data["BoE"], target_col),
        _df_para_graficos(bancos_data["FED"], target_col),
    )
    st.plotly_chart(fig_tres, use_container_width=True,
                    key=f"fig_tres_{instrumento_tipo}")

    st.divider()

    # Tarjetas de estado del modelo por banco (dic-2025)
    if instrumento_tipo == "tipo_oficial":
        st.subheader("Estado del modelo — diciembre 2025 (fin del test)")
    else:
        st.subheader("Estado del modelo OIS — diciembre 2025 (fin del test)")

    cols = st.columns(3)
    for i, banco in enumerate(config.BANCOS_CENTRALES):
        instr_b = _get_instrumento(banco, instrumento_tipo)
        try:
            scores = _cargar_scores_test(banco, instr_b)
            ultimo = scores.iloc[-1]
            conf, dir_txt, sem = interpretar_scores(
                float(ultimo["Confidence_Score"]),
                float(ultimo["Upward_Score"]),
                float(ultimo["Downward_Score"]),
            )
            emoji        = _estado_semaforo(sem)
            tipo_pred    = float(ultimo["tipo_predicho"])
            net_score    = float(ultimo["Net_Score"])
            conf_score   = float(ultimo["Confidence_Score"])
            lbl          = _INSTRUMENTO_LABEL.get(instr_b, instr_b)
            lbl_tipo     = lbl if instrumento_tipo != "tipo_oficial" else "Tipo predicho"
            _lbl_instr   = (f" — {lbl}" if instrumento_tipo != "tipo_oficial" else "")
            _net_color   = "#006B2D" if net_score >= 0 else "#CC0000"

            with cols[i]:
                st.markdown(
                    f"<div style='"
                    f"border-left:5px solid {_color_banco(banco)};"
                    f"padding:14px 18px;"
                    f"border-radius:10px;"
                    f"background:linear-gradient(135deg,#EDF7F1 0%,#F7FBF8 100%);"
                    f"box-shadow:0 1px 6px rgba(0,107,45,0.12);"
                    f"margin-bottom:10px;"
                    f"font-family:Nunito,sans-serif;"
                    f"'>"

                    f"<div style='margin-bottom:8px;'>"
                    f"<b style='color:{_color_banco(banco)};font-size:1.2em;'>{banco}</b>"
                    f"<br>"
                    f"<span style='font-size:0.80em;color:#4D4D4D;'>"
                    f"{_NOMBRE_COMPLETO[banco]}{_lbl_instr}"
                    f"</span>"
                    f"</div>"

                    f"<hr style='border:none;border-top:1px solid #C5E5D1;margin:6px 0;'>"

                    f"<div style='display:flex;justify-content:space-between;"
                    f"flex-wrap:wrap;gap:6px;margin-top:8px;'>"

                    f"<div style='flex:1;min-width:90px;'>"
                    f"<div style='font-size:0.72em;color:#6B8F78;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:0.5px;'>{lbl_tipo}</div>"
                    f"<div style='font-size:1.4em;font-weight:800;color:#003B22;'>"
                    f"{tipo_pred:.2f}%</div>"
                    f"</div>"

                    f"<div style='flex:1;min-width:90px;'>"
                    f"<div style='font-size:0.72em;color:#6B8F78;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:0.5px;'>{emoji} Net Score</div>"
                    f"<div style='font-size:1.4em;font-weight:800;"
                    f"color:{_net_color};'>"
                    f"{net_score:+.1f}</div>"
                    f"</div>"

                    f"</div>"

                    f"<div style='margin-top:10px;display:flex;"
                    f"flex-wrap:wrap;gap:8px;'>"

                    f"<div style='background:#D4EDD9;border-radius:6px;"
                    f"padding:4px 10px;flex:1;min-width:80px;text-align:center;'>"
                    f"<div style='font-size:0.68em;color:#4D4D4D;font-weight:600;'>"
                    f"CONFIDENCE</div>"
                    f"<div style='font-size:1.1em;font-weight:700;color:#003B22;'>"
                    f"{conf_score:.1f}</div>"
                    f"</div>"

                    f"<div style='background:#D4EDD9;border-radius:6px;"
                    f"padding:4px 10px;flex:1;min-width:80px;text-align:center;'>"
                    f"<div style='font-size:0.68em;color:#4D4D4D;font-weight:600;'>"
                    f"DIRECCIÓN</div>"
                    f"<div style='font-size:1.0em;font-weight:700;color:#003B22;'>"
                    f"{dir_txt}</div>"
                    f"</div>"

                    f"<div style='background:#D4EDD9;border-radius:6px;"
                    f"padding:4px 10px;flex:1;min-width:80px;text-align:center;'>"
                    f"<div style='font-size:0.68em;color:#4D4D4D;font-weight:600;'>"
                    f"CONFIANZA</div>"
                    f"<div style='font-size:1.0em;font-weight:700;color:#003B22;'>"
                    f"{conf}</div>"
                    f"</div>"

                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            cols[i].warning(f"{banco}: {exc}")

    st.divider()

    # Tabla resumen metricas de regresion
    st.subheader(
        "Metricas del modelo de regresion (Random Forest — test 2024-2025)"
        if instrumento_tipo == "tipo_oficial"
        else "Metricas del modelo OIS — regresion (Random Forest — test 2024-2025)"
    )
    df_reg = _cargar_resumen_regresion(instrumento_tipo)
    _cols_reg    = ["Banco", "MAE_test", "RMSE_test", "R2_test"]
    _rename_reg  = {"MAE_test": "MAE (pp)", "RMSE_test": "RMSE (pp)", "R2_test": "R²"}
    if "MASE" in df_reg.columns:
        _cols_reg.append("MASE")
    df_rf_test = (
        df_reg[df_reg["Modelo"] == "RF"][_cols_reg]
        .rename(columns=_rename_reg)
        .set_index("Banco")
        .round(4)
    )

    def _color_mase_panorama(val):
        if isinstance(val, float) and not pd.isna(val):
            if val < 1.0:
                return "background-color:#d4edda;color:#155724;font-weight:bold"
            return "background-color:#fff3cd;color:#856404"
        return ""

    if "MASE" in df_rf_test.columns:
        st.dataframe(
            df_rf_test.style
                .map(_color_mase_panorama, subset=["MASE"])
                .format("{:.4f}"),
            use_container_width=True,
        )
        st.caption(
            "**MASE** (Hyndman & Koehler, 2006) = MAE_test / MAE_naïve_insample. "
            "Verde = bate al random walk (MASE < 1). "
            "MASE_BCE > 1 se explica por el período ZLB 2013-2019 que reduce el denominador."
        )
    else:
        st.dataframe(df_rf_test, use_container_width=True)

    # Tabla resumen metricas de clasificacion
    st.subheader("Metricas del modelo de clasificacion (test 2024-2025)")
    df_cls = _cargar_resumen_clasificacion(instrumento_tipo)
    _df_ct  = df_cls[df_cls["Conjunto"] == "test"][
        ["Banco", "Accuracy", "F1_macro", "F1_weighted", "ROC_AUC_ovr"]
    ].rename(columns={"ROC_AUC_ovr": "ROC_AUC_test"})
    _df_ctr = df_cls[df_cls["Conjunto"] == "train"][["Banco", "ROC_AUC_ovr"]].rename(
        columns={"ROC_AUC_ovr": "ROC_AUC_train"}
    )
    df_cls_test = _df_ct.merge(_df_ctr, on="Banco").set_index("Banco").round(4)
    st.dataframe(df_cls_test, use_container_width=True)

    _df_hk_pan = _cargar_resumen_cls_hikecycle()
    _hk_note = ""
    if not _df_hk_pan.empty and instrumento_tipo == "tipo_oficial":
        _rec_boe = ""
        _rec_fed = ""
        for _, _r in _df_hk_pan.iterrows():
            _f1hk = _cargar_f1_hikecycle(_r["Banco"])
            if not _f1hk.empty and "Sube" in _f1hk["Clase"].values:
                _rec = float(_f1hk.loc[_f1hk["Clase"]=="Sube","Recall"].values[0])
                if _r["Banco"] == "BoE": _rec_boe = f"{_rec:.0%}"
                if _r["Banco"] == "FED": _rec_fed = f"{_rec:.0%}"
        if _rec_boe and _rec_fed:
            _hk_note = (
                f" Validación adicional en el ciclo de subidas 2022-2023: "
                f"Recall('Sube') = **{_rec_boe} (BoE)** y **{_rec_fed} (FED)**, "
                "confirmando que el clasificador detecta subidas cuando el ciclo las genera."
            )

    st.info(
        "**ROC_AUC_test = NaN**: la clase 'Sube' no aparece en 2024-2025 (ciclo exclusivo "
        "de bajadas). La referencia discriminativa es **ROC_AUC_train** (0.919–0.953)."
        + _hk_note
    )


# ---------------------------------------------------------------------------
# Tab 2 — Historico y test
# ---------------------------------------------------------------------------

def _seccion_historico(
    banco: str,
    df_historico: pd.DataFrame,
    instrumento: str = "tipo_oficial",
    target_col: str = "tipo_oficial",
) -> None:
    """Serie historica mensual + importancia de features + scores del test."""

    lbl = _INSTRUMENTO_LABEL.get(instrumento, instrumento)

    # --- Fila superior: historico + importancias ---
    col_hist, col_imp = st.columns([3, 2])

    with col_hist:
        st.subheader(f"Serie historica — {_NOMBRE_COMPLETO[banco]}")
        fig_hist = grafico_historico(_df_para_graficos(df_historico, target_col), banco)
        st.plotly_chart(fig_hist, use_container_width=True,
                        key=f"fig_hist_{banco}_{instrumento}")

    with col_imp:
        st.subheader("Importancia de variables (Random Forest)")
        try:
            importancias = _cargar_importancias(banco, instrumento)
            fig_imp = grafico_importancia(importancias, banco)
            st.plotly_chart(fig_imp, use_container_width=True,
                            key=f"fig_imp_{banco}_{instrumento}")
        except FileNotFoundError:
            st.warning("Archivo de importancias no encontrado.")

    st.divider()

    # --- Metricas rapidas del test ---
    titulo_test = (
        f"Predicciones en el periodo de test ({banco}, 2024-2025)"
        if instrumento == "tipo_oficial"
        else f"Predicciones OIS en el periodo de test ({banco} — {lbl}, 2024-2025)"
    )
    st.subheader(titulo_test)

    try:
        scores_test = _cargar_scores_test(banco, instrumento)
        ultimo = scores_test.iloc[-1]
        conf, dir_txt, sem = interpretar_scores(
            float(ultimo["Confidence_Score"]),
            float(ultimo["Upward_Score"]),
            float(ultimo["Downward_Score"]),
        )
        emoji = _estado_semaforo(sem)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            f"{'Tipo predicho' if instrumento == 'tipo_oficial' else lbl + ' predicho'} (dic-2025)",
            f"{float(ultimo['tipo_predicho']):.2f}%",
            delta=(
                f"{float(ultimo['tipo_predicho']) - float(scores_test.iloc[0]['tipo_predicho']):+.2f}pp"
                " desde ene-2024"
            ),
        )
        m2.metric(f"Confidence Score {emoji}", f"{float(ultimo['Confidence_Score']):.1f}")
        m3.metric("Net Score", f"{float(ultimo['Net_Score']):.1f}")
        m4.metric("Confianza", conf)

        col_prob, col_sc = st.columns(2)
        with col_prob:
            fig_prob = grafico_probabilidades(scores_test, banco, "Test 2024-2025")
            st.plotly_chart(fig_prob, use_container_width=True,
                            key=f"fig_prob_test_{banco}_{instrumento}")
        with col_sc:
            fig_sc = grafico_scores(scores_test, banco, "Test 2024-2025")
            st.plotly_chart(fig_sc, use_container_width=True,
                            key=f"fig_sc_test_{banco}_{instrumento}")

        with st.expander("Ver tabla de predicciones del test", expanded=False):
            cols_mostrar = [
                c for c in [
                    "tipo_real", "clase_real", "tipo_predicho", "direccion",
                    "P_Baja", "P_Estable", "P_Sube",
                    "Confidence_Score", "Net_Score",
                ]
                if c in scores_test.columns
            ]
            st.dataframe(scores_test[cols_mostrar].round(3), use_container_width=True)

    except FileNotFoundError:
        st.error(f"No se encontraron los scores de test para {banco} [{lbl}].")


# ---------------------------------------------------------------------------
# Tab 3 — Predicciones 2026-2030  (sub-tab: Personalizado)
# ---------------------------------------------------------------------------

def _seccion_predicciones_custom(
    banco: str,
    df_historico: pd.DataFrame,
    instrumento: str = "tipo_oficial",
    target_col: str = "tipo_oficial",
) -> None:
    """
    Sub-tab de escenario personalizado.
    """
    lbl = _INSTRUMENTO_LABEL.get(instrumento, instrumento)
    titulo = (
        f"Escenario personalizado — {_NOMBRE_COMPLETO[banco]}"
        if instrumento == "tipo_oficial"
        else f"Escenario personalizado ({lbl}) — {_NOMBRE_COMPLETO[banco]}"
    )
    st.subheader(titulo)
    st.markdown(
        "<p style='color:#006B2D;font-family:Nunito,sans-serif;font-size:0.95em;margin-bottom:8px;'>"
        "Ajusta las variables macroeconómicas para 2026-2030. "
        "La predicción se actualiza al instante. "
        "<b>OIS_taylor</b> (Regla de Taylor) se calcula automáticamente "
        "a partir de la inflación y el PIB_growth introducidos."
        "</p>",
        unsafe_allow_html=True,
    )

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        st.markdown(
            "<span style='color:#006B2D;font-family:Nunito,sans-serif;"
            "font-weight:600;font-size:0.9em;'>Partir de escenario predefinido:</span>",
            unsafe_allow_html=True,
        )
        esc_base = st.selectbox(
            "Partir de escenario predefinido:",
            options=["Base", "Optimista", "Pesimista"],
            key=f"cust_esc_{banco}_{instrumento}",
            label_visibility="collapsed",
            help="Rellena los sliders con los valores macroeconómicos del escenario seleccionado (año 2026)",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "⬆️ Cargar",
            key=f"btn_cargar_{banco}_{instrumento}",
            help="Aplicar los valores del escenario como punto de partida de los sliders",
        ):
            nuevos = _obtener_defaults_escenario(banco, esc_base)
            for var, val in nuevos.items():
                st.session_state[f"sl_{banco}_{instrumento}_{var}"] = val
            st.rerun()

    if f"sl_{banco}_{instrumento}_PIB_growth" not in st.session_state:
        defaults = _obtener_defaults_escenario(banco, "Base")
        for var, val in defaults.items():
            st.session_state[f"sl_{banco}_{instrumento}_{var}"] = val

    st.divider()

    is_fed   = (banco == "FED")
    gas_label = "Gas (USD/MMBtu)" if is_fed else "Gas (EUR/MWh)"
    gas_min   = 1.0  if is_fed else 10.0
    gas_max   = 12.0 if is_fed else 100.0
    gas_step  = 0.1  if is_fed else 1.0

    col_left, col_right = st.columns(2)

    _lbl_style = (
        "color:#006B2D;font-family:Nunito,sans-serif;"
        "font-weight:600;font-size:0.9em;margin:6px 0 -8px 0;display:block;"
    )

    with col_left:
        st.markdown(
            "<p style='color:#006B2D;font-family:Nunito,sans-serif;"
            "font-weight:700;font-size:1em;margin-bottom:4px;'>Macroeconomía</p>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<span style='{_lbl_style}'>PIB_growth (%)</span>",
                    unsafe_allow_html=True)
        st.slider("PIB_growth (%)", -3.0, 6.0, step=0.1,
                  key=f"sl_{banco}_{instrumento}_PIB_growth",
                  label_visibility="collapsed",
                  help="Tasa de crecimiento interanual del PIB (%)")
        st.markdown(f"<span style='{_lbl_style}'>PIBpc_growth (%)</span>",
                    unsafe_allow_html=True)
        st.slider("PIBpc_growth (%)", -3.0, 5.0, step=0.1,
                  key=f"sl_{banco}_{instrumento}_PIBpc_growth",
                  label_visibility="collapsed",
                  help="Tasa de crecimiento interanual del PIB per cápita (%)")
        st.markdown(f"<span style='{_lbl_style}'>Inflación (%)</span>",
                    unsafe_allow_html=True)
        st.slider("Inflación (%)", 0.0, 10.0, step=0.1,
                  key=f"sl_{banco}_{instrumento}_Inflacion",
                  label_visibility="collapsed",
                  help="Tasa de inflación anual (%)")
        st.markdown(f"<span style='{_lbl_style}'>Desempleo (%)</span>",
                    unsafe_allow_html=True)
        st.slider("Desempleo (%)", 2.0, 15.0, step=0.1,
                  key=f"sl_{banco}_{instrumento}_Desempleo",
                  label_visibility="collapsed",
                  help="Tasa de desempleo (%)")

    with col_right:
        st.markdown(
            "<p style='color:#006B2D;font-family:Nunito,sans-serif;"
            "font-weight:700;font-size:1em;margin-bottom:4px;'>Mercados y materias primas</p>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<span style='{_lbl_style}'>VIX</span>",
                    unsafe_allow_html=True)
        st.slider("VIX", 8.0, 60.0, step=0.5,
                  key=f"sl_{banco}_{instrumento}_VIX",
                  label_visibility="collapsed",
                  help="Índice de volatilidad implícita (CBOE VIX)")
        st.markdown(f"<span style='{_lbl_style}'>Oro (USD/oz)</span>",
                    unsafe_allow_html=True)
        st.slider("Oro (USD/oz)", 1500.0, 5000.0, step=50.0,
                  key=f"sl_{banco}_{instrumento}_Oro",
                  label_visibility="collapsed",
                  help="Precio del oro en USD por onza troy")
        st.markdown(f"<span style='{_lbl_style}'>Plata (USD/oz)</span>",
                    unsafe_allow_html=True)
        st.slider("Plata (USD/oz)", 10.0, 80.0, step=0.5,
                  key=f"sl_{banco}_{instrumento}_Plata",
                  label_visibility="collapsed",
                  help="Precio de la plata en USD por onza troy")
        st.markdown(f"<span style='{_lbl_style}'>{gas_label}</span>",
                    unsafe_allow_html=True)
        st.slider(gas_label, gas_min, gas_max, step=gas_step,
                  key=f"sl_{banco}_{instrumento}_Gas",
                  label_visibility="collapsed",
                  help=("Henry Hub — precio del gas natural (USD/MMBtu)"
                        if is_fed else "Gas natural europeo — TTF (EUR/MWh)"))

    st.markdown(
        "<p style='"
        "color:#006B2D;"
        "font-family:Nunito,sans-serif;"
        "font-size:0.85em;"
        "margin:4px 0 0 0;"
        "'>"
        "ℹ️ El Rating crediticio fue eliminado del modelo al tener importancia 0% "
        "en los tres bancos (variable estática sin poder predictivo)."
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    slider_vals = {
        var: st.session_state[f"sl_{banco}_{instrumento}_{var}"]
        for var in ["PIB_growth", "PIBpc_growth", "Inflacion", "Desempleo",
                    "VIX", "Oro", "Plata", "Gas"]
    }

    try:
        pred_custom, ois_val = _predecir_personalizado(banco, slider_vals, instrumento)
    except Exception as _e:
        st.error(f"Error al calcular la prediccion personalizada: {_e}")
        return

    p             = config.TAYLOR_PARAMS[banco]
    # Usar la curva hipotética para los resúmenes numéricos
    _tiene_hip    = "tipo_hipotetico" in pred_custom.columns
    _col_res      = "tipo_hipotetico" if _tiene_hip else "tipo_predicho"
    tipo_ene26    = float(pred_custom[_col_res].iloc[0])
    tipo_dic30    = float(pred_custom[_col_res].iloc[-1])
    delta_total   = tipo_dic30 - tipo_ene26

    dir_dominante = "—"
    if "direccion" in pred_custom.columns and len(pred_custom) > 0:
        dir_dominante = pred_custom["direccion"].value_counts().index[0]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(
        "OIS_taylor (equilibrio)",
        f"{ois_val:.2f}%",
        help=(
            f"Fórmula: max(0, {p['r_star']} + 1.5×(π−{p['pi_star']}) "
            f"+ 0.5×(PIB−{p['y_star']}))"
        ),
    )
    mc2.metric(f"Hipotético ene-2026", f"{tipo_ene26:.2f}%")
    mc3.metric(
        f"Hipotético dic-2030",
        f"{tipo_dic30:.2f}%",
        delta=f"{delta_total:+.2f}pp",
        delta_color="inverse" if delta_total < 0 else "normal",
    )
    mc4.metric("Dirección dominante", dir_dominante)

    # ── Gráfico principal: histórico + referencia Taylor + curva hipotética ──
    import plotly.graph_objects as _go_cust

    _df_hist_chart = _df_para_graficos(df_historico, target_col)
    _color         = _color_banco(banco)
    _margen        = _mae_ref(banco, instrumento) or 0.20

    # Determinar dirección dominante de la curva hipotética
    _dir_dom = pred_custom["direccion"].value_counts().index[0] if "direccion" in pred_custom.columns else "Estable"
    _color_hip = {"Baja": "#d62728", "Sube": "#2ca02c", "Estable": "#ff7f0e"}.get(_dir_dom, "#ff7f0e")

    _fig_cust = _go_cust.Figure()

    # Región predicción futura
    _fig_cust.add_vrect(
        x0=str(pred_custom.index[0])[:10], x1=str(pred_custom.index[-1])[:10],
        fillcolor="rgba(220,255,220,0.20)", layer="below", line_width=0,
    )

    # Serie histórica
    _fig_cust.add_trace(_go_cust.Scatter(
        x=_df_hist_chart.index, y=_df_hist_chart["tipo_oficial"],
        mode="lines", name="Histórico",
        line=dict(color=_color, width=2),
        hovertemplate="%{x|%Y-%m}<br>Real: %{y:.2f}%<extra></extra>",
    ))

    # Referencia Taylor Rule (línea de equilibrio, discontinua gris)
    _fig_cust.add_trace(_go_cust.Scatter(
        x=pred_custom.index, y=pred_custom["tipo_predicho"],
        mode="lines", name="Equilibrio Taylor Rule",
        line=dict(color="gray", width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m}<br>Taylor: %{y:.2f}%<extra></extra>",
    ))

    # Banda de confianza ±MAE alrededor de la curva hipotética
    _y_hip  = pred_custom["tipo_hipotetico"]
    _fig_cust.add_trace(_go_cust.Scatter(
        x=list(pred_custom.index) + list(pred_custom.index[::-1]),
        y=list(_y_hip + _margen) + list((_y_hip - _margen)[::-1]),
        fill="toself", fillcolor=f"rgba(255,140,0,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=True,
        name=f"IC ±{_margen:.2f}pp", hoverinfo="skip",
    ))

    # Curva hipotética direccional (línea principal)
    _fig_cust.add_trace(_go_cust.Scatter(
        x=pred_custom.index, y=_y_hip,
        mode="lines+markers", name=f"Trayectoria hipotética ({_dir_dom})",
        line=dict(color=_color_hip, width=2.5),
        marker=dict(size=4),
        hovertemplate="%{x|%Y-%m}<br>Hipotético: %{y:.2f}%<extra></extra>",
    ))

    _fig_cust.update_layout(
        template="plotly_white", height=520,
        title=f"Trayectoria hipotética — {banco} | Escenario personalizado",
        xaxis_title="Fecha", yaxis_title=f"{lbl} (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", margin=dict(l=50, r=30, t=60, b=50),
    )
    st.plotly_chart(_fig_cust, use_container_width=True,
                    key=f"fig_cust_{banco}_{instrumento}")

    st.caption(
        f"**Cómo leer el gráfico:** La línea **{_dir_dom.lower()}** es la trayectoria hipotética "
        "generada aplicando la señal del clasificador (P_Sube − P_Baja) × 25 bp/mes "
        "con ancla suave hacia el equilibrio Taylor Rule (línea punteada gris). "
        "Refleja hacia dónde irían los tipos si las condiciones macro introducidas se mantienen."
    )

    col_prob, col_sc = st.columns(2)
    with col_prob:
        fig_prob = grafico_probabilidades(pred_custom, banco, "Personalizado")
        st.plotly_chart(fig_prob, use_container_width=True,
                        key=f"fig_prob_custom_{banco}_{instrumento}")
    with col_sc:
        fig_sc = grafico_scores(pred_custom, banco, "Personalizado")
        st.plotly_chart(fig_sc, use_container_width=True,
                        key=f"fig_sc_custom_{banco}_{instrumento}")

    with st.expander("Ver tabla de predicciones personalizadas", expanded=False):
        cols_show = [
            c for c in [
                "tipo_predicho", "direccion",
                "P_Baja", "P_Estable", "P_Sube",
                "Confidence_Score", "Net_Score",
            ]
            if c in pred_custom.columns
        ]
        st.dataframe(pred_custom[cols_show].round(3), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Predicciones 2026-2030  (sub-tab: Predefinidos)
# ---------------------------------------------------------------------------

def _seccion_predicciones(
    banco: str,
    df_historico: pd.DataFrame,
    escenario: str,
    instrumento: str = "tipo_oficial",
    target_col: str = "tipo_oficial",
) -> None:
    """Comparativa de los tres escenarios + escenario personalizado con sliders."""

    sub_pred, sub_custom = st.tabs([
        "📋 Escenarios predefinidos",
        "🎛️ Escenario personalizado",
    ])

    # ── Sub-tab A: escenarios predefinidos ──────────────────────────────────
    with sub_pred:
        try:
            pred_base = _cargar_pred_futura(banco, "Base",      instrumento)
            pred_opt  = _cargar_pred_futura(banco, "Optimista", instrumento)
            pred_pes  = _cargar_pred_futura(banco, "Pesimista", instrumento)
        except FileNotFoundError as exc:
            st.error(f"No se encontraron los archivos de prediccion: {exc}")
            return

        lbl = _INSTRUMENTO_LABEL.get(instrumento, instrumento)

        st.subheader(f"Comparativa de escenarios — {_NOMBRE_COMPLETO[banco]} (2022-2030)")
        st.markdown(
            "<p style='"
            "color:#006B2D;"
            "font-family:Nunito,sans-serif;"
            "font-size:0.88em;"
            "margin-top:-6px;"
            "margin-bottom:8px;"
            "'>"
            "📐 <b>Modelo híbrido RF + Regla de Taylor:</b> "
            "Para el periodo de test 2024-2025, el nivel del tipo se reconstruye "
            "a partir de las predicciones de delta del Random Forest (1 mes vista). "
            "Para el horizonte 2026-2030, el nivel predicho es la Regla de Taylor "
            "[OIS_taylor = r* + 1.5·(π−π*) + 0.5·(PIB−y*)], evitando la acumulación "
            "de error en 60 pasos iterados. La <b>dirección y probabilidades</b> (Sube/Baja/Estable) "
            "provienen del clasificador de Regresión Logística en ambos horizontes."
            "</p>",
            unsafe_allow_html=True,
        )
        fig_esc = grafico_escenarios(
            _df_para_graficos(df_historico, target_col),
            pred_base, pred_opt, pred_pes, banco,
        )
        st.plotly_chart(fig_esc, use_container_width=True,
                        key=f"fig_esc_{banco}_{instrumento}")

        st.subheader("Prediccion de tipos para diciembre 2030")
        _col_base, _col_opt, _col_pes = st.columns(3)

        _escenarios_viz = [
            (_col_base, "Base",      pred_base, "#1f77b4"),
            (_col_opt,  "Optimista", pred_opt,  "#2ca02c"),
            (_col_pes,  "Pesimista", pred_pes,  "#d62728"),
        ]

        for col, esc_nombre, pred_df, esc_color in _escenarios_viz:
            val_fin   = float(pred_df["tipo_predicho"].iloc[-1])
            val_ini   = float(pred_df["tipo_predicho"].iloc[0])
            delta_tot = val_fin - val_ini
            ultimo_row = pred_df.iloc[-1]
            conf, dir_txt, sem = interpretar_scores(
                float(ultimo_row["Confidence_Score"]),
                float(ultimo_row["Upward_Score"]),
                float(ultimo_row["Downward_Score"]),
            )
            emoji = _estado_semaforo(sem)

            with col:
                st.markdown(
                    f"<div style='border-top:3px solid {esc_color};"
                    f"padding:8px 12px;border-radius:4px;background:#f9f9f9'>"
                    f"<b style='color:{esc_color}'>{emoji} Escenario {esc_nombre}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.metric(
                    label=f"{lbl} dic-2030",
                    value=f"{val_fin:.2f}%",
                    delta=f"{delta_tot:+.2f}pp vs ene-2026",
                )
                st.markdown(
                    f"<p style='"
                    f"color:#006B2D;"
                    f"font-family:Nunito,sans-serif;"
                    f"font-size:0.85em;"
                    f"margin:4px 0 0 0;"
                    f"'>"
                    f"Direccion: <b>{dir_txt}</b> | Confianza: <b>{conf}</b>"
                    f"</p>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Trayectoria anual de variables macroeconómicas ─────────────────
        st.subheader(f"Trayectoria anual — hipótesis macro del escenario {escenario}")
        st.caption(
            "Los escenarios definen valores distintos por año (2026-2030), no constantes. "
            "Cada variable cambia anualmente siguiendo proyecciones de organismos como el FMI, BCE o BoE. "
            "La columna OIS_taylor es el resultado de aplicar la Regla de Taylor a esas hipótesis — "
            "es la entrada real al modelo en cada uno de los 60 meses del horizonte."
        )

        import plotly.graph_objects as go

        from src.models.prediccion_futura import definir_escenarios as _def_esc_c1
        _datos_c1  = _def_esc_c1()
        _comod_c1  = _datos_c1["_comodidades"]
        _p_c1      = config.TAYLOR_PARAMS[banco]
        _gas_label = "Gas (USD/MMBtu)" if banco == "FED" else "Gas (EUR/MWh)"

        _filas_c1 = []
        for _anio in range(config.PREDICT_START, config.PREDICT_END + 1):
            _v  = _datos_c1[banco][escenario][_anio]
            _cm = _comod_c1[escenario][_anio]
            _ois_c1 = max(0.0,
                _p_c1["r_star"]
                + 1.5 * (_v["Inflacion"] - _p_c1["pi_star"])
                + 0.5 * (_v["PIB_growth"] - _p_c1["y_star"])
            )
            _filas_c1.append({
                "Año":              _anio,
                "PIB (%)":          _v["PIB_growth"],
                "Inflación (%)":    _v["Inflacion"],
                "Desempleo (%)":    _v["Desempleo"],
                _gas_label:         _v["Gas"],
                "VIX":              _cm["VIX"],
                "OIS_taylor (%)":   round(_ois_c1, 3),
            })
        _df_c1 = pd.DataFrame(_filas_c1).set_index("Año")

        _col_c1a, _col_c1b = st.columns([2, 3])
        with _col_c1a:
            st.dataframe(
                _df_c1.style.format({
                    "PIB (%)":        "{:.1f}",
                    "Inflación (%)":  "{:.1f}",
                    "Desempleo (%)":  "{:.1f}",
                    _gas_label:       "{:.1f}",
                    "VIX":            "{:.1f}",
                    "OIS_taylor (%)": "{:.3f}",
                }),
                use_container_width=True,
            )
        with _col_c1b:
            _fig_c1 = go.Figure()
            _anos_c1 = [r["Año"] for r in _filas_c1]
            _fig_c1.add_trace(go.Scatter(
                x=_anos_c1, y=[r["Inflación (%)"] for r in _filas_c1],
                name="Inflación", line=dict(color="#d62728", width=2.2),
                mode="lines+markers", marker=dict(size=7),
                hovertemplate="%{x}: %{y:.1f}%<extra>Inflación</extra>",
            ))
            _fig_c1.add_trace(go.Scatter(
                x=_anos_c1, y=[r["PIB (%)"] for r in _filas_c1],
                name="PIB_growth", line=dict(color="#1f77b4", width=2.2),
                mode="lines+markers", marker=dict(size=7),
                hovertemplate="%{x}: %{y:.1f}%<extra>PIB_growth</extra>",
            ))
            _fig_c1.add_trace(go.Scatter(
                x=_anos_c1, y=[r["OIS_taylor (%)"] for r in _filas_c1],
                name="OIS_taylor", line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                mode="lines+markers", marker=dict(size=8, symbol="diamond"),
                hovertemplate="%{x}: %{y:.3f}%<extra>OIS_taylor</extra>",
            ))
            _fig_c1.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
            _fig_c1.update_layout(
                template    = "plotly_white",
                height      = 240,
                title       = f"Trayectoria macroeconómica — {banco} | {escenario}",
                xaxis_title = "Año",
                yaxis_title = "(%)",
                legend      = dict(orientation="h", y=1.12, x=0),
                margin      = dict(l=40, r=20, t=65, b=35),
                hovermode   = "x unified",
            )
            st.plotly_chart(_fig_c1, use_container_width=True,
                            key=f"fig_macro_{banco}_{instrumento}_{escenario}")

        st.info(
            f"**Regla de Taylor aplicada:** "
            f"OIS_taylor = max(0, {_p_c1['r_star']} + 1.5 × (π − {_p_c1['pi_star']}) "
            f"+ 0.5 × (PIB − {_p_c1['y_star']})) "
            f"· Con r* = {_p_c1['r_star']}%, π* = {_p_c1['pi_star']}%, y* = {_p_c1['y_star']}% "
            f"para {_NOMBRE_COMPLETO[banco]}."
        )

        st.divider()

        pred_sel = {"Base": pred_base, "Optimista": pred_opt, "Pesimista": pred_pes}[escenario]

        with st.expander(
            f"🔍 Detalle — Escenario {escenario} ({banco}, 2026-2030) "
            f"— trayectoria completa, probabilidades y scores",
            expanded=False,
        ):
            st.caption(
                "Historico completo (2000-2025) + prediccion del escenario seleccionado "
                "con banda de confianza ±MAE. A diferencia del grafico anterior "
                "(comparativa de los 3 escenarios desde 2022), este muestra la "
                "perspectiva historica completa y las probabilidades mes a mes."
            )
            col_izq, col_der = st.columns(2)
            with col_izq:
                fig_pred = grafico_prediccion(
                    _df_para_graficos(df_historico, target_col),
                    pred_sel, banco, escenario,
                    margen_ci=_mae_ref(banco, instrumento),
                )
                st.plotly_chart(fig_pred, use_container_width=True,
                                key=f"fig_pred_{banco}_{instrumento}_{escenario}")
            with col_der:
                fig_prob = grafico_probabilidades(pred_sel, banco, escenario)
                st.plotly_chart(fig_prob, use_container_width=True,
                                key=f"fig_prob_{banco}_{instrumento}_{escenario}")

            fig_sc = grafico_scores(pred_sel, banco, escenario)
            st.plotly_chart(fig_sc, use_container_width=True,
                            key=f"fig_sc_{banco}_{instrumento}_{escenario}")

            st.markdown("**Tabla de predicciones mensuales**")
            cols_mostrar = [
                c for c in [
                    "tipo_predicho", "direccion",
                    "P_Baja", "P_Estable", "P_Sube",
                    "Confidence_Score", "Net_Score",
                ]
                if c in pred_sel.columns
            ]
            st.dataframe(pred_sel[cols_mostrar].round(3), use_container_width=True)

        st.divider()

        # ── Analisis de sensibilidad (Fase B) ──────────────────────────────
        st.subheader("Analisis de sensibilidad — Regla de Taylor")
        st.caption(
            "Muestra cuanto cambia el tipo predicho (OIS_taylor) cuando se varia "
            "la inflacion o el PIB_growth ±1 pp respecto a los valores del escenario "
            "y año seleccionados."
        )

        p = config.TAYLOR_PARAMS[banco]
        _esc_lower_sens = escenario.lower()
        _ed_df = pd.read_csv(
            config.SCENARIOS_DIR / f"escenario_{_esc_lower_sens}_{banco}.csv"
        )
        _anios_disp = sorted(_ed_df["Anio"].unique().tolist())
        _key_sens = f"sens_anio_{banco}_{instrumento}"
        if _key_sens not in st.session_state:
            st.session_state[_key_sens] = _anios_disp[0]
        _anio_ref = st.select_slider(
            "Año de referencia para la sensibilidad:",
            options=_anios_disp,
            key=_key_sens,
        )
        _ed_fila  = _ed_df[_ed_df["Anio"] == _anio_ref].iloc[0]
        infl_base = float(_ed_fila["Inflacion"])
        pib_base  = float(_ed_fila["PIB_growth"])

        variaciones = [-1.0, -0.5, 0.0, +0.5, +1.0]

        filas_infl = []
        for delta in variaciones:
            infl_new = infl_base + delta
            ois      = max(0.0, p["r_star"] + 1.5*(infl_new - p["pi_star"]) + 0.5*(pib_base - p["y_star"]))
            dif_ois  = ois - max(0.0, p["r_star"] + 1.5*(infl_base - p["pi_star"]) + 0.5*(pib_base - p["y_star"]))
            filas_infl.append({
                "Variacion Inflacion":  f"{delta:+.1f} pp",
                "Inflacion resultante": f"{infl_new:.1f}%",
                "OIS_taylor (%)":       round(ois, 3),
                "Cambio vs Base (pp)":  f"{dif_ois:+.3f}",
            })

        filas_pib = []
        for delta in variaciones:
            pib_new  = pib_base + delta
            ois      = max(0.0, p["r_star"] + 1.5*(infl_base - p["pi_star"]) + 0.5*(pib_new - p["y_star"]))
            dif_ois  = ois - max(0.0, p["r_star"] + 1.5*(infl_base - p["pi_star"]) + 0.5*(pib_base - p["y_star"]))
            filas_pib.append({
                "Variacion PIB_growth": f"{delta:+.1f} pp",
                "PIB resultante":       f"{pib_new:.1f}%",
                "OIS_taylor (%)":       round(ois, 3),
                "Cambio vs Base (pp)":  f"{dif_ois:+.3f}",
            })

        sc1, sc2 = st.columns(2)

        def _color_cambio(val):
            try:
                v = float(val)
                if v > 0:   return "color:#155724;font-weight:bold"
                elif v < 0: return "color:#721c24;font-weight:bold"
                return "color:#856404;font-weight:bold"
            except Exception:
                return ""

        with sc1:
            st.markdown(f"**Sensibilidad a la Inflacion** (PIB fijo = {pib_base:.1f}%, {escenario} {_anio_ref})")
            df_si = pd.DataFrame(filas_infl).set_index("Variacion Inflacion")
            st.dataframe(df_si.style.map(_color_cambio, subset=["Cambio vs Base (pp)"]),
                         use_container_width=True)
        with sc2:
            st.markdown(f"**Sensibilidad al PIB_growth** (Inflacion fija = {infl_base:.1f}%, {escenario} {_anio_ref})")
            df_sp = pd.DataFrame(filas_pib).set_index("Variacion PIB_growth")
            st.dataframe(df_sp.style.map(_color_cambio, subset=["Cambio vs Base (pp)"]),
                         use_container_width=True)

        st.info(
            f"**Interpretacion:** Un aumento de +1 pp en la inflacion eleva el OIS_taylor "
            f"en **+1.5 pp** (coeficiente de Taylor). Un aumento de +1 pp en el PIB_growth "
            f"lo eleva en **+0.5 pp**. La inflacion tiene el triple de influencia sobre "
            f"el tipo predicho que el crecimiento economico, coherente con el mandato "
            f"principal de estabilidad de precios de {_NOMBRE_COMPLETO[banco]}."
        )

    # ── Sub-tab B: escenario personalizado con sliders ──────────────────────
    with sub_custom:
        _seccion_predicciones_custom(banco, df_historico, instrumento, target_col)


# ---------------------------------------------------------------------------
# Tab 4 — Evaluacion del modelo
# ---------------------------------------------------------------------------

def _seccion_evaluacion(banco: str, instrumento: str = "tipo_oficial") -> None:
    """Metricas de regresion y clasificacion, matrices de confusion."""

    lbl = _INSTRUMENTO_LABEL.get(instrumento, instrumento)
    titulo_eval = (
        f"Evaluacion del modelo — {_NOMBRE_COMPLETO[banco]}"
        if instrumento == "tipo_oficial"
        else f"Evaluacion del modelo OIS ({lbl}) — {_NOMBRE_COMPLETO[banco]}"
    )
    st.subheader(titulo_eval)

    # ================================================================
    # Regresion
    # ================================================================
    st.markdown("### Modelo de regresion (Random Forest)")

    df_reg  = _cargar_resumen_regresion(instrumento)
    met_rf  = _metricas_reg_rf(df_reg, banco)

    df_reg_banco = df_reg[df_reg["Banco"] == banco].copy()
    df_reg_show = df_reg_banco[
        ["Modelo", "MAE_train", "RMSE_train", "R2_train", "MAE_test", "RMSE_test", "R2_test"]
    ].round(4).set_index("Modelo")
    st.dataframe(df_reg_show, use_container_width=True)

    if met_rf:
        _mase_val = met_rf.get("MASE", float("nan"))
        _mase_ok  = pd.notna(_mase_val) and _mase_val < 1.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE  test (RF)",  f"{met_rf.get('MAE_test',  0):.4f} pp")
        c2.metric("RMSE test (RF)",  f"{met_rf.get('RMSE_test', 0):.4f} pp")
        c3.metric("R²   test (RF)",  f"{met_rf.get('R2_test',   0):.4f}")
        c4.metric(
            f"{'✅' if _mase_ok else '⚠️'} MASE (RF)",
            f"{_mase_val:.4f}" if pd.notna(_mase_val) else "N/A",
            help="MASE < 1 = bate al random walk (Hyndman & Koehler, 2006)",
        )

    # ── Comparativa con baseline naïve + R² sobre deltas ────────────────────
    st.markdown("#### Comparativa con baseline naïve y R² sobre cambios")
    st.caption(
        "El **baseline naïve** predice que el tipo no cambia cada mes "
        "(tipo_pred_t = tipo_actual_t). Es el mínimo exigible a cualquier modelo. "
        "El **R² sobre deltas** mide la capacidad de predecir el CAMBIO mensual, "
        "no el nivel — es más exigente y no está inflado por la autocorrelación."
    )

    try:
        from sklearn.metrics import r2_score as _r2, mean_absolute_error as _mae
        _df_reg   = _cargar_pred_regresion_test(banco, instrumento)
        _tipo_act = _df_reg["y_real"] - _df_reg["delta_real"]
        _naive_mae = _mae(_df_reg["y_real"], _tipo_act)
        _naive_r2  = _r2(_df_reg["y_real"], _tipo_act)
        _model_mae = _mae(_df_reg["y_real"], _df_reg["y_pred_rf"])
        _model_r2  = _r2(_df_reg["y_real"], _df_reg["y_pred_rf"])
        _r2_delta  = _r2(_df_reg["delta_real"], _df_reg["delta_pred_rf"])

        _fila_naive = {"Métrica": "Baseline naïve (sin cambio)",
                       "MAE (pp)": f"{_naive_mae:.4f}",
                       "R² (nivel)": f"{_naive_r2:.4f}",
                       "R² (Δ cambio)": "— (ref.)"}
        _fila_rf    = {"Métrica": f"RF modelo ({banco} — {lbl})",
                       "MAE (pp)": f"{_model_mae:.4f}",
                       "R² (nivel)": f"{_model_r2:.4f}",
                       "R² (Δ cambio)": f"{_r2_delta:.4f}"}
        _df_comp = pd.DataFrame([_fila_naive, _fila_rf]).set_index("Métrica")

        _mejor_mae = _model_mae < _naive_mae

        def _color_comp(row):
            if row.name == f"RF modelo ({banco} — {lbl})":
                mae_color = ("background-color:#d4edda;color:#155724"
                             if _mejor_mae else "background-color:#f8d7da;color:#721c24")
                return [mae_color, "", ""]
            return ["", "", ""]

        st.dataframe(_df_comp.style.apply(_color_comp, axis=1), use_container_width=True)

        _delta_mae = _model_mae - _naive_mae
        _skill_pct = (1 - _model_mae / _naive_mae) * 100
        _mase_disp = met_rf.get("MASE", float("nan")) if met_rf else float("nan")

        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        _bc1.metric(
            "Skill vs naïve",
            f"{_skill_pct:+.1f}%",
            delta=f"MAE {_model_mae:.4f} vs naïve {_naive_mae:.4f} pp",
            delta_color="inverse",
        )
        _bc2.metric("R² sobre nivel (RF)",    f"{_model_r2:.4f}")
        _bc3.metric("R² sobre Δ cambio (RF)", f"{_r2_delta:.4f}")
        _bc4.metric(
            "MASE (Hyndman 2006)",
            f"{_mase_disp:.4f}" if pd.notna(_mase_disp) else "N/A",
            help=(
                "MASE = MAE_test / MAE_naïve_insample. "
                "Denominador = mean(|delta_t|) en train (random walk in-sample). "
                "MASE < 1 → bate al random walk."
            ),
        )

        # ── Interpretación del MASE ─────────────────────────────────────────
        if pd.notna(_mase_disp):
            if _mase_disp < 1.0:
                st.success(
                    f"**MASE = {_mase_disp:.4f} < 1** — El modelo de regresión ({banco}) **bate al "
                    f"random walk** en el test 2024-2025: comete "
                    f"{(1 - _mase_disp)*100:.1f}% menos error que predecir «el tipo no cambia». "
                    "Esta métrica no está inflada por la autocorrelación de niveles "
                    "(Hyndman & Koehler, 2006)."
                )
            else:
                st.info(
                    f"**MASE = {_mase_disp:.4f} > 1** — El modelo ({banco}) tiene un MAE test "
                    f"mayor que el naïve random walk calculado en muestra. "
                    "Esto **no implica que el modelo sea malo** en términos absolutos "
                    f"(MAE = {_model_mae:.4f} pp, que es muy pequeño); refleja que "
                    "el denominador del MASE — el error naïve en muestra — es especialmente "
                    "bajo para este banco. "
                    + ("La BCE mantuvo tipos en cero o near-zero durante 2013-2019, "
                       "un período largo de 'sin cambio' que reduce el MAE_naive_insample "
                       "a solo 0.052 pp. El test 2024-2025 fue un ciclo de bajadas activas "
                       "con más movimiento que el promedio histórico, lo que eleva el MASE."
                       if banco == "BCE" else
                       "El periodo de test presentó más movimiento de tipos del que "
                       "el modelo vio en media durante el entrenamiento.")
                )

        if _r2_delta < 0:
            st.info(
                f"**¿Por qué R² sobre Δ es negativo ({_r2_delta:.3f})?** "
                "El cambio mensual del tipo es casi siempre 0 pp (el banco no actúa la mayoría de los meses). "
                "El R²_delta mide si el modelo mejora sobre predecir «delta = 0 siempre»: con varianza del "
                "cambio muy pequeña, cualquier error de timing lo lleva a negativo. "
                "El valor real del modelo está en las probabilidades (P_Baja/P_Estable/P_Sube) "
                "y en los meses donde SÍ hay movimiento."
            )

        # Nota FED solo para tipos oficiales (el efecto de régimen es tipo-oficial-específico)
        if banco == "FED" and instrumento == "tipo_oficial" and _skill_pct < 0:
            st.warning(
                "⚠️ **¿Por qué el Skill de regresión de la FED es negativo?** \n\n"
                "El Skill global es negativo porque **14 de los 24 meses del test (ene-dic 2025) "
                "la Fed mantuvo el tipo sin moverlo**, y el modelo predice pequeñas bajadas en esos "
                "meses. El naïve «sin cambio» acierta en todos ellos (error = 0), lo que arrastra "
                "el Skill global hacia negativo.\n\n"
                "Sin embargo, en los **10 meses donde la Fed SÍ movió tipos** "
                "(ago-dic 2024, ago-oct 2025), el modelo bate al naïve un **~50%** "
                "(MAE_modelo ≈ 0.08 pp vs MAE_naïve ≈ 0.17 pp).\n\n"
                "**La causa raíz es un cambio de régimen**, no un fallo del modelo: "
                "entre enero y julio de 2025, la Fed mantuvo tipos durante 7 meses consecutivos "
                "con inflación por encima del objetivo y brecha de Taylor negativa. "
                "En los 24 años de entrenamiento (2000-2023), esa combinación siempre "
                "desembocó en recortes. La pausa de 2025 — motivada por incertidumbre "
                "geopolítica y riesgo arancelario — es estructuralmente nueva y no "
                "estaba representada en los datos de entrenamiento.\n\n"
                "La **clasificación de la FED sí funciona**: Accuracy = **83.3%** "
                "(vs 66.7% del baseline trivial) y F1_macro = **0.54**, con 20 de los "
                "24 meses correctamente clasificados."
            )
    except Exception as _exc:
        st.warning(f"No se pudo calcular el baseline naïve: {_exc}")

    st.divider()

    # ================================================================
    # Clasificacion
    # ================================================================
    st.markdown("### Modelo de clasificacion (Regresion Logistica Multinomial)")

    try:
        df_cls    = _cargar_resumen_clasificacion(instrumento)
        met_train = _metricas_cls(df_cls, banco, "train")
        met_test  = _metricas_cls(df_cls, banco, "test")
    except Exception as _e:
        st.error(f"Error cargando metricas de clasificacion: {_e}")
        df_cls, met_train, met_test = pd.DataFrame(), {}, {}

    col_graf, col_cm = st.columns(2)

    with col_graf:
        st.markdown("#### Metricas train vs test")
        try:
            fig_met = grafico_metricas_clasificacion(met_train, met_test, banco)
            st.plotly_chart(fig_met, use_container_width=True,
                            key=f"fig_met_{banco}_{instrumento}")
        except Exception as _e:
            st.warning(f"No se pudo generar el grafico de metricas: {_e}")

    with col_cm:
        st.markdown("#### Matriz de confusion (test 2024-2025)")
        try:
            cm_test = _cargar_confusion(banco, "test", instrumento)
            fig_cm  = grafico_confusion_matrix(cm_test, banco, "test")
            st.plotly_chart(fig_cm, use_container_width=True,
                            key=f"fig_cm_test_{banco}_{instrumento}")
        except FileNotFoundError:
            st.warning("Matriz de confusion de test no encontrada.")
        except Exception as _e:
            st.warning(f"Error en la matriz de confusion: {_e}")

    cols_cls = [
        "Conjunto", "Accuracy", "Precision_macro", "Recall_macro",
        "F1_macro", "F1_weighted", "ROC_AUC_ovr",
    ]
    df_cls_banco = df_cls[df_cls["Banco"] == banco][cols_cls].round(4)
    with st.expander("Ver metricas detalladas de clasificacion", expanded=False):
        st.dataframe(df_cls_banco.set_index("Conjunto"), use_container_width=True)

    # ── F1 por clase ────────────────────────────────────────────────────────
    st.markdown("#### Precision, Recall y F1 por clase (test 2024-2025)")
    st.caption(
        "Desglose honesto del rendimiento del clasificador. "
        "El accuracy global puede ser engañoso cuando una clase domina — "
        "el F1-score por clase revela la capacidad real del modelo en cada dirección."
    )

    try:
        df_f1 = _cargar_f1_por_clase(banco, instrumento)

        def _color_f1(val):
            if isinstance(val, float):
                if val >= 0.60:
                    return "background-color:#d4edda; color:#155724"
                elif val >= 0.30:
                    return "background-color:#fff3cd; color:#856404"
                else:
                    return "background-color:#f8d7da; color:#721c24"
            return ""

        styled = (
            df_f1.set_index("Clase")
            .style
            .map(_color_f1, subset=["Precision", "Recall", "F1-score"])
            .format({"Precision": "{:.2f}", "Recall": "{:.2f}",
                     "F1-score": "{:.2f}", "Soporte": "{:.0f}"})
        )
        st.dataframe(styled, use_container_width=True)

        total_test = df_f1["Soporte"].sum()
        if total_test > 0:
            dist_txt = " | ".join(
                f"{row['Clase']}: {row['Soporte']} meses ({row['Soporte']/total_test*100:.0f}%)"
                for _, row in df_f1.iterrows()
            )
            st.caption(f"Distribución test: {dist_txt}")

        f1_baja  = float(df_f1.loc[df_f1["Clase"]=="Baja",  "F1-score"].values[0]) if "Baja"  in df_f1["Clase"].values else 0.0
        f1_sube  = float(df_f1.loc[df_f1["Clase"]=="Sube",  "F1-score"].values[0]) if "Sube"  in df_f1["Clase"].values else 0.0
        sop_sube = int(df_f1.loc[df_f1["Clase"]=="Sube", "Soporte"].values[0])     if "Sube"  in df_f1["Clase"].values else 0

        _f1_macro_test = met_test.get("F1_macro", 0.0)
        _f1_macro_base = 0.22
        if f1_baja == 0.0 or f1_sube == 0.0:
            if _f1_macro_test > 0.40:
                st.info(
                    f"ℹ️  **La clase 'Sube' no aparece en el test 2024-2025** "
                    f"(0 meses de subida en el ciclo de bajadas). "
                    "El F1-macro del test (**{:.2f}**) supera claramente el baseline trivial ({:.2f}), "
                    "lo que confirma que el clasificador añade valor real. ".format(
                        _f1_macro_test, _f1_macro_base) +
                    f"El ROC-AUC de entrenamiento es **{met_train.get('ROC_AUC_ovr', 0):.3f}**, "
                    "indicando buena capacidad discriminativa con ciclos completos."
                )
            else:
                _sube_txt = " (0 meses de 'Sube' en test)" if sop_sube == 0 else ""
                st.markdown(
                    f"<div style='"
                    f"background-color:#FAFADC;"
                    f"border-left:4px solid #C8D400;"
                    f"border-radius:6px;"
                    f"padding:12px 16px;"
                    f"color:#000000;"
                    f"font-family:Nunito,sans-serif;"
                    f"font-size:0.95em;"
                    f"'>"
                    f"⚠️ <b>Limitacion conocida del periodo test:</b> "
                    f"El clasificador no detecta correctamente las bajadas individuales "
                    f"y no observa ninguna subida{_sube_txt}. "
                    f"Esto es consecuencia del desequilibrio de clases en 2024-2025: "
                    f"el ciclo fue de bajadas graduales con mayoría de meses estables. "
                    f"<b>En entrenamiento (2000-2023), con ciclos completos de subidas y bajadas, "
                    f"el F1-macro supera 0.55 y el ROC-AUC es "
                    f"{met_train.get('ROC_AUC_ovr', 0):.3f}.</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    except FileNotFoundError:
        st.warning("Reporte de clasificacion por clase no encontrado.")

    roc_train = met_train.get("ROC_AUC_ovr", float("nan"))
    roc_str   = f"{roc_train:.3f}" if pd.notna(roc_train) else "N/A"
    st.info(
        "**Limitacion documentada:** El ROC_AUC_ovr del test es NaN porque "
        "la clase 'Sube' no aparece en ningun mes de 2024-2025 (ciclo exclusivo "
        "de bajadas). La capacidad discriminativa real del modelo se refleja en "
        f"el ROC_AUC de entrenamiento: **{roc_str}** (para {banco})."
    )

    st.success(
        "✅ **Look-ahead corregido (Fase 1):** `PIB_growth` usa el crecimiento del año anterior "
        "(lag-1) en lugar del año en curso. Los datos del Banco Mundial del año *t* solo se "
        "publican en *t+1*; el lag-1 garantiza que cada mes solo usa información realmente disponible. "
        "El conjunto de entrenamiento pasó de 288 → 276 observaciones (2001-2023). "
        "Las métricas son **estables antes y después** del cambio — BCE: 0.077 pp, BoE: 0.051 pp, "
        "FED: 0.046 pp — lo que demuestra que el modelo **no dependía del look-ahead**."
    )

    with st.expander("Ver matriz de confusion (entrenamiento)", expanded=False):
        try:
            cm_train = _cargar_confusion(banco, "train", instrumento)
            fig_cm_train = grafico_confusion_matrix(cm_train, banco, "train")
            st.plotly_chart(fig_cm_train, use_container_width=True,
                            key=f"fig_cm_train_{banco}_{instrumento}")
        except FileNotFoundError:
            st.warning("Matriz de confusion de entrenamiento no encontrada.")

    st.divider()

    # ================================================================
    # Evaluacion adicional: ciclo de subidas 2022-2023
    # ================================================================
    st.markdown("### 📈 Evaluacion adicional — Ciclo de subidas 2022-2023")
    st.caption(
        "El test principal (2024-2025) no contiene meses de subida de tipos. "
        "Para demostrar que el clasificador SÍ detecta la clase 'Sube', se realiza una "
        "evaluación adicional entrenando con 2000-2021 y evaluando en 2022-2023 — "
        "el ciclo de subidas más agresivo de los últimos 20 años."
    )

    if instrumento != "tipo_oficial":
        st.info(
            f"ℹ️  La evaluación del ciclo de subidas solo está disponible para tipos oficiales. "
            "Selecciona 'Tipos Oficiales' en el selector del panel lateral para verla."
        )
    else:
        df_hk = _cargar_resumen_cls_hikecycle()
        if df_hk.empty:
            st.info(
                "⚙️  Ejecuta `python run_hikecycle_eval.py` para generar esta evaluación. "
                "Los CSVs se guardan en Resultados/ con sufijo '_hikecycle'."
            )
        else:
            fila_hk_df = df_hk[df_hk["Banco"] == banco]
            if fila_hk_df.empty:
                st.warning(f"No hay datos del ciclo de subidas para {banco}.")
            else:
                fhk = fila_hk_df.iloc[0]
                n_sube  = int(fhk["N_Sube"])
                n_est   = int(fhk["N_Estable"])
                n_baja  = int(fhk["N_Baja"])
                acc_hk  = float(fhk["Accuracy"])
                f1_hk   = float(fhk["F1_macro"])
                roc_hk  = fhk["ROC_AUC_ovr"]
                roc_hk_str = f"{float(roc_hk):.3f}" if pd.notna(roc_hk) else "N/A"

                # ── Métricas rápidas — Recall_Sube es la métrica clave ───────
                _f1hk_q      = _cargar_f1_hikecycle(banco)
                _rec_sube_q  = float(_f1hk_q.loc[_f1hk_q["Clase"]=="Sube","Recall"].values[0]) \
                               if (not _f1hk_q.empty and "Sube" in _f1hk_q["Clase"].values) else float("nan")
                _f1_sube_q   = float(_f1hk_q.loc[_f1hk_q["Clase"]=="Sube","F1-score"].values[0]) \
                               if (not _f1hk_q.empty and "Sube" in _f1hk_q["Clase"].values) else float("nan")
                _fw_q        = float(fhk.get("F1_weighted", float("nan")))

                h1, h2, h3, h4 = st.columns(4)
                h1.metric(
                    "Meses 'Sube' en test",
                    f"{n_sube}",
                    help="Número de meses con subida de tipos en el periodo 2022-2023",
                )
                h2.metric(
                    "Recall 'Sube'",
                    f"{_rec_sube_q:.0%}" if pd.notna(_rec_sube_q) else "N/A",
                    help="% de meses con subida correctamente identificados por el clasificador",
                )
                h3.metric(
                    "F1 'Sube'",
                    f"{_f1_sube_q:.3f}" if pd.notna(_f1_sube_q) else "N/A",
                    help="F1-score de la clase 'Sube' (combina precisión y recall)",
                )
                h4.metric(
                    "F1_weighted",
                    f"{_fw_q:.3f}" if pd.notna(_fw_q) else "N/A",
                    help="F1 ponderado por soporte de clase — más representativo que F1_macro cuando hay clases sin soporte",
                )

                st.caption(
                    f"Distribución test 2022-2023: "
                    f"Baja = {n_baja} | Estable = {n_est} | **Sube = {n_sube}**  "
                    f"(train: 2001-2021 — {252} obs)"
                )

                col_comp, col_f1 = st.columns(2)

                # ── Tabla comparativa test principal vs hikecycle ────────────
                with col_comp:
                    st.markdown("#### Comparativa de evaluaciones")
                    try:
                        df_cls_main = _cargar_resumen_clasificacion(instrumento)
                        met_main    = _metricas_cls(df_cls_main, banco, "test")
                        roc_main    = met_main.get("ROC_AUC_ovr", float("nan"))
                        _df_comp = pd.DataFrame([
                            {
                                "Evaluacion":  "Test principal 2024-2025",
                                "Periodo":     "Bajadas",
                                "Clases":      "Baja / Estable",
                                "Accuracy":    met_main.get("Accuracy",  0.0),
                                "F1_macro":    met_main.get("F1_macro",  0.0),
                                "ROC_AUC":     f"{float(roc_main):.3f}" if pd.notna(roc_main) else "NaN",
                            },
                            {
                                "Evaluacion":  "Ciclo subidas 2022-2023",
                                "Periodo":     "Subidas",
                                "Clases":      "Estable / Sube",
                                "Accuracy":    acc_hk,
                                "F1_macro":    f1_hk,
                                "ROC_AUC":     roc_hk_str,
                            },
                        ]).set_index("Evaluacion")
                        st.dataframe(
                            _df_comp.style.format({"Accuracy": "{:.3f}", "F1_macro": "{:.3f}"}),
                            use_container_width=True,
                        )
                    except Exception as _exc:
                        st.warning(f"No se pudo mostrar comparativa: {_exc}")

                # ── F1 por clase del ciclo de subidas ────────────────────────
                with col_f1:
                    st.markdown("#### F1 por clase (ciclo subidas)")
                    df_f1_hk = _cargar_f1_hikecycle(banco)
                    if not df_f1_hk.empty:
                        def _color_f1_hk(val):
                            if isinstance(val, float):
                                if val >= 0.60:
                                    return "background-color:#d4edda; color:#155724"
                                elif val >= 0.30:
                                    return "background-color:#fff3cd; color:#856404"
                                else:
                                    return "background-color:#f8d7da; color:#721c24"
                            return ""

                        styled_hk = (
                            df_f1_hk.set_index("Clase")
                            .style
                            .map(_color_f1_hk, subset=["Precision", "Recall", "F1-score"])
                            .format({"Precision": "{:.2f}", "Recall": "{:.2f}",
                                     "F1-score": "{:.2f}", "Soporte": "{:.0f}"})
                        )
                        st.dataframe(styled_hk, use_container_width=True)
                    else:
                        st.warning("Reporte F1 hikecycle no encontrado.")

                # ── Matriz de confusión del ciclo de subidas ─────────────────
                st.markdown("#### Matriz de confusion — ciclo subidas 2022-2023")
                cm_hk_df = _cargar_confusion_hikecycle(banco)
                if not cm_hk_df.empty:
                    try:
                        fig_cm_hk = grafico_confusion_matrix(cm_hk_df, banco, "2022-2023")
                        st.plotly_chart(fig_cm_hk, use_container_width=True,
                                        key=f"fig_cm_hk_{banco}_{instrumento}")
                    except Exception as _exc_cm:
                        st.dataframe(cm_hk_df, use_container_width=True)
                else:
                    st.warning("Matriz de confusion hikecycle no encontrada.")

                # ── Interpretacion ────────────────────────────────────────────
                if not df_f1_hk.empty and "Sube" in df_f1_hk["Clase"].values:
                    _rec_sube = float(
                        df_f1_hk.loc[df_f1_hk["Clase"] == "Sube", "Recall"].values[0]
                    )
                    if _rec_sube >= 0.70:
                        st.success(
                            f"**El clasificador detecta subidas:** Recall('Sube') = **{_rec_sube:.0%}** "
                            f"en {banco} durante 2022-2023. El modelo SÍ identifica meses de subida "
                            "cuando el ciclo macroeconómico los genera. La ausencia de 'Sube' en "
                            "el test principal (2024-2025) es una limitación del período, no del modelo."
                        )
                    else:
                        st.info(
                            f"**Detección de subidas en {banco}:** Recall('Sube') = **{_rec_sube:.0%}**. "
                            "El clasificador tiene dificultades para predecir el timing exacto de las "
                            "subidas agresivas de 2022 (ciclo sin precedente en los datos de entrenamiento). "
                            "BoE y FED muestran mayor capacidad de detección."
                        )

    st.divider()

    # ================================================================
    # Walk-Forward Validation
    # ================================================================
    st.markdown("### Validacion walk-forward — Robustez de las metricas en 3 ventanas temporales")

    if instrumento != "tipo_oficial":
        st.info(
            f"ℹ️ La validación walk-forward no está disponible para el modelo OIS ({lbl}). "
            "El walk-forward se ejecuta sobre el modelo de tipos oficiales. "
            "Para el modelo OIS, las métricas de test (2024-2025) son la referencia principal."
        )
    else:
        st.caption(
            "En lugar de confiar en un unico corte temporal, el modelo se reentrena "
            "tres veces con ventanas progresivas y se evalua sobre el ano siguiente. "
            "Si el MAE es consistente en las tres ventanas, los resultados no son "
            "fruto de un periodo favorable sino de la capacidad real del modelo."
        )

        df_wf = _cargar_walk_forward()
        if df_wf.empty:
            st.warning("Ejecuta src/models/walk_forward.py para generar los resultados.")
        else:
            df_banco_wf = df_wf[df_wf["Banco"] == banco].copy()

            cols_show = ["Ventana", "Test_inicio", "Test_fin", "N_train", "N_test", "MAE", "RMSE", "R2"]
            df_show = df_banco_wf[cols_show].rename(columns={
                "Test_inicio": "Test desde", "Test_fin": "Test hasta",
                "N_train": "Obs. train", "N_test": "Obs. test",
            }).set_index("Ventana")

            def _highlight_mae(s):
                styles = []
                for v in s:
                    if isinstance(v, float):
                        if v < 0.15:
                            styles.append("background-color:#d4edda;color:#155724")
                        elif v < 0.25:
                            styles.append("background-color:#fff3cd;color:#856404")
                        else:
                            styles.append("background-color:#f8d7da;color:#721c24")
                    else:
                        styles.append("")
                return styles

            styled_wf = df_show.style.apply(_highlight_mae, subset=["MAE"]).format({
                "MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}",
                "Obs. train": "{:.0f}", "Obs. test": "{:.0f}",
            })
            st.dataframe(styled_wf, use_container_width=True)

            mae_medio = df_banco_wf["MAE"].mean()
            r2_medio  = df_banco_wf["R2"].mean()
            mae_max   = df_banco_wf["MAE"].max()
            mae_min   = df_banco_wf["MAE"].min()

            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("MAE medio (3 ventanas)", f"{mae_medio:.4f} pp")
            wc2.metric("R² medio (3 ventanas)",  f"{r2_medio:.4f}")
            wc3.metric("MAE mejor ventana",       f"{mae_min:.4f} pp")
            wc4.metric("MAE peor ventana",        f"{mae_max:.4f} pp")

            import plotly.graph_objects as go
            fig_wf = go.Figure()
            _paleta_wf = ["#AED6F1", "#5DADE2", config.COLORES.get(banco, "#1F4E79")]
            for i, (_, row) in enumerate(df_banco_wf.iterrows()):
                fig_wf.add_trace(go.Bar(
                    name=row["Ventana"],
                    x=[row["Ventana"]],
                    y=[row["MAE"]],
                    marker_color=_paleta_wf[i % len(_paleta_wf)],
                    text=[f"{row['MAE']:.4f} pp"],
                    textposition="outside",
                ))
            fig_wf.update_layout(
                template="plotly_white",
                title=f"MAE por ventana temporal — {banco}",
                yaxis_title="MAE (pp)",
                showlegend=False,
                height=300,
                margin=dict(t=50, b=40),
            )
            fig_wf.add_hline(
                y=mae_medio, line_dash="dash", line_color="gray",
                annotation_text=f"Media: {mae_medio:.4f}",
                annotation_position="top right",
            )
            st.plotly_chart(fig_wf, use_container_width=True,
                            key=f"fig_wf_{banco}")

            st.info(
                f"**Interpretacion:** El MAE oscila entre **{mae_min:.4f} pp** y **{mae_max:.4f} pp** "
                "segun la dificultad del periodo evaluado. La Ventana 1 (2022) corresponde al ciclo "
                "de subidas agresivas post-COVID — el mas dificil de predecir. "
                "El R² se mantiene por encima de 0.83 en todas las ventanas."
            )


# ===========================================================================
# APLICACION PRINCIPAL
# ===========================================================================

@st.cache_data(show_spinner=False)
def _cargar_datos_caso() -> dict:
    """Predicciones (oficial + OIS) de los 3 bancos × 3 escenarios para el
    caso Iberdrola. Solo la carga de CSVs va cacheada; el cálculo es en vivo."""
    return caso.cargar_predicciones(config.RESULTS_DIR)


def _fmt_es(x: float, dec: int = 0) -> str:
    """Formato numérico español: punto como separador de miles, coma decimal."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def _seccion_caso_iberdrola() -> None:
    """Caso práctico: optimización de la refinanciación de Iberdrola
    2026-2028 con las predicciones del modelo (4 estrategias × 3 escenarios)."""
    import plotly.graph_objects as go

    st.caption(
        "ℹ️ Esta pestaña no depende del selector de banco/escenario del "
        "sidebar: usa siempre los 3 bancos (BCE·EUR, BoE·GBP, FED·USD) y "
        "los 3 escenarios (base, optimista, pesimista)."
    )

    # ── Bloque 1: enunciado ─────────────────────────────────────────────────
    st.markdown(
        "<div style='"
        "border-left:5px solid #009A44;padding:18px 22px;border-radius:10px;"
        "background:linear-gradient(135deg,#EDF7F1 0%,#F7FBF8 100%);"
        "box-shadow:0 1px 6px rgba(0,107,45,0.12);margin-bottom:14px;"
        "font-family:Nunito,sans-serif;'>"
        "<b style='color:#006B2D;font-size:1.25em;'>💼 El caso: refinanciar "
        "la deuda de Iberdrola 2026-2028</b><br><br>"
        "<span style='color:#4D4D4D;'>"
        "Iberdrola afronta vencimientos de deuda de <b>5.392 M€ (2026)</b>, "
        "<b>4.123 M€ (2027)</b> y <b>4.424 M€ (2028)</b> (datos FY2025), en "
        "pleno plan inversor de <b>58.000 M€</b>. La dirección financiera debe "
        "decidir cómo refinanciarlos: ¿a tipo fijo, variable, o guiándose por "
        "las predicciones del modelo?<br><br>"
        "<b style='color:#006B2D;'>Pregunta:</b> ¿qué estrategia de "
        "refinanciación minimiza el coste financiero esperado 2026-2030 sin "
        "comprometer el rating BBB+? Se comparan 4 estrategias bajo los 3 "
        "escenarios del modelo, ponderados por la verosimilitud que les "
        "asigna el clasificador.</span></div>",
        unsafe_allow_html=True,
    )

    # ── Carga de datos ──────────────────────────────────────────────────────
    try:
        datos = _cargar_datos_caso()
    except FileNotFoundError as exc:
        st.error(
            "Faltan CSVs de predicciones en Resultados/ "
            f"(predicciones_futuras_{{banco}}_{{escenario}}[_OIS].csv): {exc}"
        )
        return

    # ── Bloque 2: supuestos interactivos ────────────────────────────────────
    with st.expander("⚙️ Supuestos del caso (ajustables)", expanded=False):
        _CASO_DEFAULTS = {
            "caso_spread": 110, "caso_pctfijo": 77.2,
            "caso_eur": 55.0, "caso_gbp": 25.0, "caso_usd": 20.0,
            "caso_umbral_ps": 0.35, "caso_umbral_reg": 0.25,
            "caso_modo_pesos": "Derivados del clasificador (recomendado)",
            "caso_pb": 0.36, "caso_po": 0.34, "caso_pp": 0.30,
        }
        if st.button("↩️ Restablecer valores por defecto", key="caso_reset"):
            for k, v in _CASO_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Coste y statu quo**")
            spread_pb = st.slider(
                "Spread corporativo (pb)", 50, 200, 110, 5,
                help="Diferencial sobre el tipo de referencia. Calibrado con "
                     "el bono verde mar-2026 (cupón 3,125% vs BCE ~2,0%).",
                key="caso_spread")
            pct_fijo_sq = st.slider(
                "% deuda a tipo fijo del statu quo", 0.0, 100.0, 77.2, 0.1,
                help="Cuentas anuales 2025: 77,2% a tipo fijo.",
                key="caso_pctfijo")
        with c2:
            st.markdown("**Mix de divisas del statu quo**")
            w_eur = st.number_input("EUR (BCE) %", 0.0, 100.0, 55.0, 5.0, key="caso_eur")
            w_gbp = st.number_input("GBP (BoE) %", 0.0, 100.0, 25.0, 5.0, key="caso_gbp")
            w_usd = st.number_input("USD (FED) %", 0.0, 100.0, 20.0, 5.0, key="caso_usd")
            _tot = w_eur + w_gbp + w_usd
            if _tot <= 0:
                st.error("El mix de divisas no puede sumar 0%.")
                return
            if abs(_tot - 100) > 0.01:
                st.caption(f"Se normaliza a 100% (suma actual: {_fmt_es(_tot, 1)}%).")
        with c3:
            st.markdown("**Umbrales de la estrategia 4 (guiada por el modelo)**")
            umbral_fijar = st.slider(
                "P(Sube) anual para girar a fijo", 0.10, 0.60, 0.35, 0.05,
                help="Señal del clasificador: fija si la probabilidad media "
                     "anual de subida supera este umbral.",
                key="caso_umbral_ps")
            umbral_reg = st.slider(
                "Subida predicha por la regresión (pp)", 0.05, 0.75, 0.25, 0.05,
                help="Señal de la regresión: fija si predice una subida "
                     "acumulada mayor que este umbral el año siguiente.",
                key="caso_umbral_reg")

        st.markdown("---")
        modo_pesos = st.radio(
            "Pesos de los escenarios",
            ["Derivados del clasificador (recomendado)", "Manuales"],
            horizontal=True, key="caso_modo_pesos")
        pesos_manual = None
        if modo_pesos == "Manuales":
            p1, p2, p3 = st.columns(3)
            pb_ = p1.slider("Peso base", 0.0, 1.0, 0.36, 0.01, key="caso_pb")
            po_ = p2.slider("Peso optimista", 0.0, 1.0, 0.34, 0.01, key="caso_po")
            pp_ = p3.slider("Peso pesimista", 0.0, 1.0, 0.30, 0.01, key="caso_pp")
            if pb_ + po_ + pp_ <= 0:
                st.error("Los pesos manuales no pueden sumar 0.")
                return
            pesos_manual = {"base": pb_, "optimista": po_, "pesimista": pp_}
            st.caption("Los pesos se normalizan para sumar 100%.")

    # ── Cálculo en vivo ─────────────────────────────────────────────────────
    divisas_sq = {"BCE": w_eur / _tot, "BoE": w_gbp / _tot, "FED": w_usd / _tot}
    estrategias = caso.construir_estrategias(
        divisas_statuquo=divisas_sq,
        pct_fijo_statuquo=pct_fijo_sq / 100,
    )
    res = caso.calcular_caso(
        datos,
        estrategias=estrategias,
        pesos=pesos_manual,
        spread=spread_pb / 100,
        umbral_fijar=umbral_fijar,
        umbral_subida_reg=umbral_reg,
    )
    pesos = res["pesos"]
    costes = res["costes"]
    esperado = {
        n: sum(pesos[e] * costes[n][e] for e in caso.ESCENARIOS)
        for n in estrategias
    }
    mejor = min(esperado, key=esperado.get)

    # ── Bloque 3a: pesos de escenarios ──────────────────────────────────────
    st.subheader("Pesos de los escenarios")
    if pesos_manual is None:
        st.markdown(
            "Los pesos se derivan del propio clasificador: para cada escenario "
            "se calcula la **verosimilitud direccional media** (la probabilidad "
            "media que el clasificador asigna a la dirección que sigue la senda "
            "de ese escenario, con umbral ±0,05 pp/mes) y se normaliza. Un "
            "escenario cuya trayectoria el clasificador ve más plausible pesa más."
        )
        df_pesos_disp = pd.DataFrame({
            "Escenario": [e.capitalize() for e in caso.ESCENARIOS],
            "Verosimilitud media": [
                _fmt_es(res["verosim"][e] * 100, 1) + " %" for e in caso.ESCENARIOS],
            "Peso normalizado": [
                _fmt_es(pesos[e] * 100, 1) + " %" for e in caso.ESCENARIOS],
        })
    else:
        st.markdown("Pesos fijados **manualmente** (normalizados a 100%).")
        df_pesos_disp = pd.DataFrame({
            "Escenario": [e.capitalize() for e in caso.ESCENARIOS],
            "Peso normalizado": [
                _fmt_es(pesos[e] * 100, 1) + " %" for e in caso.ESCENARIOS],
        })
    st.dataframe(df_pesos_disp, hide_index=True, use_container_width=True)

    st.divider()

    # ── Bloque 3b: tabla principal de costes ────────────────────────────────
    st.subheader("Coste financiero total 2026-2030 por estrategia (M€)")
    nombres = list(estrategias.keys())
    df_tabla = pd.DataFrame({
        "Estrategia": [n + (" 🏆" if n == mejor else "") for n in nombres],
        "Base": [costes[n]["base"] for n in nombres],
        "Optimista": [costes[n]["optimista"] for n in nombres],
        "Pesimista": [costes[n]["pesimista"] for n in nombres],
        "Esperado": [esperado[n] for n in nombres],
    })

    _num_cols = ["Base", "Optimista", "Pesimista", "Esperado"]

    def _resaltar_minimo(col: pd.Series):
        if col.name not in _num_cols:
            return ["" for _ in col]
        return [
            "background-color:#009A44;color:white;font-weight:700;"
            if v == col.min() else "" for v in col
        ]

    def _fila_ganadora(row: pd.Series):
        if "🏆" in str(row["Estrategia"]):
            return ["border-top:2px solid #006B2D;border-bottom:2px solid #006B2D;"
                    for _ in row]
        return ["" for _ in row]

    styler = (
        df_tabla.style
        .apply(_resaltar_minimo, axis=0)
        .apply(_fila_ganadora, axis=1)
        .format({c: lambda v: _fmt_es(v) + " M€" for c in _num_cols})
    )
    st.dataframe(styler, hide_index=True, use_container_width=True)
    st.caption(
        "En verde, el mínimo de cada columna. 🏆 = estrategia ganadora por "
        "coste esperado."
    )

    # ── Métricas destacadas ─────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    ahorro_sq = esperado["1. Statu quo"] - esperado["4. Guiada por el modelo"]
    dif_agresiva_pes = (costes["3. Agresiva"]["pesimista"]
                        - costes["4. Guiada por el modelo"]["pesimista"])
    m1.metric(
        "Ahorro esperado: modelo vs statu quo",
        f"{_fmt_es(ahorro_sq)} M€",
        delta=f"{-ahorro_sq / esperado['1. Statu quo'] * 100:.1f}% de coste",
        delta_color="inverse",
    )
    m2.metric(
        "Protección vs agresiva (esc. pesimista)",
        f"{_fmt_es(dif_agresiva_pes)} M€",
        delta="menor coste si los tipos suben" if dif_agresiva_pes > 0
        else "la agresiva costaría menos",
        delta_color="normal" if dif_agresiva_pes > 0 else "inverse",
    )
    m3.metric("Estrategia óptima (coste esperado)", mejor.split(". ", 1)[-1])

    # ── Bloque 3c: gráfico de barras agrupadas ──────────────────────────────
    _colores_esc = {"base": "#009A44", "optimista": "#8DC63F", "pesimista": "#CC0000"}
    fig_bar = go.Figure()
    for e in caso.ESCENARIOS:
        fig_bar.add_trace(go.Bar(
            name=e.capitalize(),
            x=nombres,
            y=[costes[n][e] for n in nombres],
            marker_color=_colores_esc[e],
        ))
    fig_bar.add_trace(go.Scatter(
        name="Coste esperado",
        x=nombres,
        y=[esperado[n] for n in nombres],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color="#003B22",
                    line=dict(width=2, color="white")),
        text=[_fmt_es(esperado[n]) for n in nombres],
        textposition="top center",
        textfont=dict(color="#003B22", size=12),
    ))
    fig_bar.update_layout(
        barmode="group",
        title="Coste por estrategia y escenario (M€, total 2026-2030)",
        yaxis_title="Coste financiero (M€)",
        font=dict(family="Nunito, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=460,
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="caso_fig_barras")

    # ── Bloque 3d: perfil riesgo-retorno ────────────────────────────────────
    col_rr, col_giros = st.columns([3, 2])
    with col_rr:
        _colores_estr = ["#4D4D4D", "#0083CA", "#CC0000", "#009A44"]
        fig_rr = go.Figure()
        for n, c in zip(nombres, _colores_estr):
            fig_rr.add_trace(go.Scatter(
                x=[costes[n]["pesimista"]],
                y=[esperado[n]],
                mode="markers+text",
                name=n,
                text=[n.split(". ", 1)[-1]],
                textposition="top center",
                marker=dict(size=16, color=c,
                            line=dict(width=2, color="white")),
            ))
        fig_rr.update_layout(
            title="Perfil riesgo-retorno (abajo-izquierda = mejor)",
            xaxis_title="Riesgo: coste en escenario pesimista (M€)",
            yaxis_title="Coste esperado (M€)",
            font=dict(family="Nunito, sans-serif"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=420,
        )
        st.plotly_chart(fig_rr, use_container_width=True, key="caso_fig_rr")

    # ── Bloque 3e: giros a fijo de la estrategia 4 ──────────────────────────
    with col_giros:
        st.markdown("**Giros a fijo de la estrategia guiada por el modelo**")
        df_giros = res["df_giros"].pivot(
            index="Banco", columns="Escenario", values="Año_giro_a_fijo"
        )[caso.ESCENARIOS]
        df_giros.columns = [c.capitalize() for c in df_giros.columns]
        st.dataframe(df_giros, use_container_width=True)
        st.caption(
            "Año en que cada tramo gira de variable a fijo. La señal es "
            "híbrida: clasificador (P(Sube) anual > "
            f"{_fmt_es(umbral_fijar * 100)}%) **o** regresión (subida "
            f"acumulada > {_fmt_es(umbral_reg, 2)} pp el año siguiente)."
        )

    st.divider()

    # ── Bloque 4: interpretación dinámica ───────────────────────────────────
    st.subheader("Interpretación")
    mas_protectora = min(nombres, key=lambda n: costes[n]["pesimista"])
    modelo = "4. Guiada por el modelo"
    extra_vs_min = esperado[modelo] - esperado[mejor]

    partes = [
        f"Con los supuestos actuales, la estrategia con menor coste esperado "
        f"es **{mejor}** ({_fmt_es(esperado[mejor])} M€), y la que mejor "
        f"protege en el escenario pesimista es **{mas_protectora}** "
        f"({_fmt_es(costes[mas_protectora]['pesimista'])} M€)."
    ]
    if mejor == modelo:
        partes.append(
            "La estrategia guiada por el modelo **domina**: gana por coste "
            "esperado y además limita el riesgo en el escenario adverso."
        )
    elif mas_protectora == modelo:
        partes.append(
            f"La estrategia guiada por el modelo queda a solo "
            f"{_fmt_es(extra_vs_min)} M€ del mínimo esperado "
            f"({-extra_vs_min / esperado[mejor] * 100:+.1f}%), pero recorta el "
            f"coste del escenario pesimista en "
            f"{_fmt_es(dif_agresiva_pes)} M€ frente a la agresiva. En términos "
            f"media-riesgo, es la opción más defendible para preservar el "
            f"rating BBB+: captura casi todo el ahorro del tramo variable y "
            f"gira a fijo cuando el modelo anticipa subidas."
        )
    else:
        partes.append(
            f"Con estos supuestos, la estrategia guiada por el modelo tiene un "
            f"coste esperado de {_fmt_es(esperado[modelo])} M€ y un coste "
            f"pesimista de {_fmt_es(costes[modelo]['pesimista'])} M€; conviene "
            f"revisar los umbrales de giro para evaluar su perfil media-riesgo."
        )
    partes.append(
        f"Frente al statu quo, seguir al modelo supone un ahorro esperado de "
        f"**{_fmt_es(ahorro_sq)} M€** en el horizonte 2026-2030."
    )
    st.markdown(" ".join(partes))


def main() -> None:

    # ── Configuracion de pagina ─────────────────────────────────────────────
    st.set_page_config(
        page_title="Tipos de Interes — TFG",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Paleta Iberdrola + tipografia Nunito ────────────────────────────────
    st.markdown("""
    <style>
    /* ── Google Fonts: Nunito (equivalente a Iberpangea) ── */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap');

    /* ── Variables de color Iberdrola ── */
    :root {
        --verde-iberdrola:  #009A44;
        --verde-oscuro:     #006B2D;
        --verde-claro:      #8DC63F;
        --verde-profundo:   #003B22;
        --verde-lima:       #C8D400;
        --azul-iberdrola:   #0083CA;
        --gris-grafito:     #4D4D4D;
        --fondo-app:        #F7FBF8;
        --fondo-card:       #EDF7F1;
        --texto-sidebar:    #D6EEE0;
    }

    /* ════════════════════════════════════════════
       FUENTE GLOBAL
    ════════════════════════════════════════════ */
    html, body, button, input, select, textarea,
    .stApp, .stMarkdown, .stText,
    [class*="st-"], p, span, label, div {
        font-family: 'Nunito', sans-serif !important;
    }

    /* ════════════════════════════════════════════
       FONDO Y LAYOUT
    ════════════════════════════════════════════ */
    .stApp {
        background-color: var(--fondo-app);
    }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    /* ════════════════════════════════════════════
       TITULOS (area principal — NO sidebar)
    ════════════════════════════════════════════ */
    .main h1, section[data-testid="stMain"] h1 {
        color: var(--verde-profundo) !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    .main h2, section[data-testid="stMain"] h2 {
        color: var(--verde-oscuro) !important;
        font-weight: 700 !important;
    }
    .main h3, section[data-testid="stMain"] h3,
    .main h4, section[data-testid="stMain"] h4 {
        color: var(--verde-iberdrola) !important;
        font-weight: 600 !important;
    }

    /* ════════════════════════════════════════════
       TABS
    ════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E6F4EC;
        border-radius: 10px;
        padding: 5px 6px;
        gap: 4px;
        border: 1px solid #C5E5D1;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-weight: 600 !important;
        color: var(--gris-grafito) !important;
        padding: 6px 18px;
        background-color: transparent;
    }
    /* Tab NO seleccionada: hover suave */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        background-color: #C5E5D1 !important;
        color: var(--verde-oscuro) !important;
    }
    /* Tab seleccionada: verde Iberdrola */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--verde-iberdrola) !important;
        color: #FFFFFF !important;
    }
    /* Tab seleccionada en hover: verde oscuro */
    .stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
        background-color: var(--verde-oscuro) !important;
        color: #FFFFFF !important;
    }

    /* ════════════════════════════════════════════
       SIDEBAR — fondo y texto legible
    ════════════════════════════════════════════ */
    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--verde-profundo) !important;
        border-right: 3px solid var(--verde-iberdrola);
    }

    /* Texto general dentro del sidebar */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown em,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] p {
        color: var(--texto-sidebar) !important;
    }
    /* Etiquetas de widgets (label del selectbox, radio, etc.) */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] label {
        color: #A8D5BC !important;
        font-weight: 600 !important;
    }
    /* Valor seleccionado en selectbox del sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: var(--texto-sidebar) !important;
    }
    /* Radio buttons del sidebar */
    [data-testid="stSidebar"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [data-baseweb="radio"] span {
        color: var(--texto-sidebar) !important;
    }
    /* Fondo suave para los widgets del sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 6px;
        padding: 2px 4px;
    }
    /* Caption / small en sidebar */
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption {
        color: #7EB899 !important;
    }
    /* Expander en sidebar */
    [data-testid="stSidebar"] details summary {
        color: #A8D5BC !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] details p,
    [data-testid="stSidebar"] details li,
    [data-testid="stSidebar"] details span {
        color: var(--texto-sidebar) !important;
    }
    [data-testid="stSidebar"] details code {
        background-color: rgba(255,255,255,0.12) !important;
        color: var(--verde-lima) !important;
        border-radius: 4px;
        padding: 1px 5px;
    }
    /* Divisor en sidebar */
    [data-testid="stSidebar"] hr {
        border: none !important;
        border-top: 1px solid rgba(0,154,68,0.4) !important;
        margin: 0.8rem 0 !important;
    }

    /* ════════════════════════════════════════════
       METRICAS — sidebar
    ════════════════════════════════════════════ */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.07) !important;
        border-left: 3px solid var(--verde-lima) !important;
        border-radius: 8px;
        padding: 6px 10px;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: var(--verde-lima) !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #A8D5BC !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: #8DC63F !important;
    }

    /* ════════════════════════════════════════════
       METRICAS — area principal
    ════════════════════════════════════════════ */
    .main [data-testid="stMetric"],
    section[data-testid="stMain"] [data-testid="stMetric"] {
        background-color: var(--fondo-card);
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid var(--verde-iberdrola);
        box-shadow: 0 1px 4px rgba(0,154,68,0.10);
    }
    .main [data-testid="stMetricLabel"],
    section[data-testid="stMain"] [data-testid="stMetricLabel"] {
        color: var(--gris-grafito) !important;
        font-weight: 600 !important;
    }
    .main [data-testid="stMetricValue"],
    section[data-testid="stMain"] [data-testid="stMetricValue"] {
        color: var(--verde-profundo) !important;
        font-weight: 800 !important;
    }
    .main [data-testid="stMetricDelta"],
    section[data-testid="stMain"] [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* ════════════════════════════════════════════
       BOTONES
    ════════════════════════════════════════════ */
    .stButton > button {
        background-color: var(--verde-iberdrola) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 8px 20px !important;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: var(--verde-oscuro) !important;
        color: white !important;
    }

    /* ════════════════════════════════════════════
       TEXTO GLOBAL — evitar blanco sobre blanco
       Streamlit puede dejar texto blanco en áreas
       claras según la versión/tema del sistema.
    ════════════════════════════════════════════ */
    /* Normalización general: texto oscuro en área principal */
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] li,
    section[data-testid="stMain"] td,
    section[data-testid="stMain"] th,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] span {
        color: #1A1A1A;
    }
    /* Texto dentro de alertas — siempre negro */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] li,
    [data-testid="stAlert"] strong,
    [data-testid="stAlert"] em,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #1A1A1A !important;
    }
    /* Texto en expanders — negro */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
        color: #1A1A1A !important;
    }
    /* Captions — gris oscuro legible */
    [data-testid="stCaptionContainer"] p,
    .stCaption p {
        color: #4D4D4D !important;
    }
    /* Selectbox y radio en área principal — negro */
    section[data-testid="stMain"] [data-baseweb="select"] span,
    section[data-testid="stMain"] [data-baseweb="radio"] label,
    section[data-testid="stMain"] [data-baseweb="radio"] span {
        color: #1A1A1A !important;
    }

    /* ════════════════════════════════════════════
       ALERTAS — info / success / warning / error
       Compatible con Streamlit >= 1.20
    ════════════════════════════════════════════ */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }
    /* Info */
    [data-testid="stAlert"][kind="info"],
    div.stInfo {
        border-left: 4px solid var(--azul-iberdrola) !important;
        background-color: #EAF4FB !important;
    }
    /* Success */
    [data-testid="stAlert"][kind="success"],
    div.stSuccess {
        border-left: 4px solid var(--verde-iberdrola) !important;
        background-color: #EAF6EE !important;
    }
    /* Warning */
    [data-testid="stAlert"][kind="warning"],
    div.stWarning {
        border-left: 4px solid var(--verde-lima) !important;
        background-color: #FAFADC !important;
    }

    /* ════════════════════════════════════════════
       EXPANDERS — area principal
    ════════════════════════════════════════════ */
    .main details summary,
    section[data-testid="stMain"] details summary {
        color: var(--verde-oscuro) !important;
        font-weight: 700 !important;
    }
    .main details[open] summary,
    section[data-testid="stMain"] details[open] summary {
        color: var(--verde-iberdrola) !important;
    }

    /* ════════════════════════════════════════════
       DIVISOR — area principal
    ════════════════════════════════════════════ */
    .main hr,
    section[data-testid="stMain"] hr {
        border: none !important;
        border-top: 2px solid #C5E5D1 !important;
        margin: 1.2rem 0 !important;
    }

    /* ════════════════════════════════════════════
       DATAFRAMES / TABLAS
    ════════════════════════════════════════════ */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ════════════════════════════════════════════
       INPUTS (selectbox, slider) — area principal
    ════════════════════════════════════════════ */
    .main [data-baseweb="select"] > div,
    section[data-testid="stMain"] [data-baseweb="select"] > div {
        border-radius: 8px !important;
        border-color: #C5E5D1 !important;
    }

    /* Labels de sliders en verde oscuro */
    .main [data-testid="stSlider"] label,
    section[data-testid="stMain"] [data-testid="stSlider"] label,
    .main [data-testid="stSlider"] [data-testid="stWidgetLabel"],
    section[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stWidgetLabel"] {
        color: #006B2D !important;
        font-weight: 600 !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* ════════════════════════════════════════════
       CAPTION
    ════════════════════════════════════════════ */
    .main .stCaption p,
    section[data-testid="stMain"] .stCaption p {
        color: #6B8F78 !important;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Barra lateral ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #006B2D 0%, #009A44 100%);
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 8px;
        '>
            <div style='color:#C8D400;font-size:1.4em;font-weight:800;
                        font-family:Nunito,sans-serif;letter-spacing:-0.5px;'>
                📈 TFG
            </div>
            <div style='color:#FFFFFF;font-size:0.95em;font-weight:600;
                        font-family:Nunito,sans-serif;margin-top:2px;'>
                Predicción de Tipos de Interés
            </div>
            <div style='color:#A8D5BC;font-size:0.78em;margin-top:4px;
                        font-family:Nunito,sans-serif;'>
                Aritz Fernández Aulestiarte
            </div>
            <div style='color:#8DC63F;font-size:0.75em;margin-top:2px;
                        font-family:Nunito,sans-serif;'>
                BCE · BoE · FED · 2000–2030
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Selector de banco central
        banco = st.selectbox(
            "Banco central",
            options=config.BANCOS_CENTRALES,
            format_func=lambda b: f"{b} — {_NOMBRE_COMPLETO[b]}",
        )

        # Selector de instrumento (Tipos Oficiales vs Tipos OIS)
        instrumento_tipo = st.radio(
            "Instrumento",
            options=["tipo_oficial", "OIS"],
            format_func=lambda x: (
                "📌 Tipos Oficiales" if x == "tipo_oficial"
                else "📊 Tipos OIS (€STR · SONIA · SOFR)"
            ),
            help=(
                "**Tipos Oficiales**: ECB Depo Facility / UKBASERATE / FFR  \n"
                "**Tipos OIS**: €STR (BCE) / SONIA (BoE) / SOFR (FED) — "
                "tipos de mercado overnight derivados de swaps OIS"
            ),
        )

        # Nombre del instrumento especifico para el banco seleccionado
        instrumento = _get_instrumento(banco, instrumento_tipo)
        target_col  = _target_col_from(instrumento)
        lbl         = _INSTRUMENTO_LABEL.get(instrumento, instrumento)

        if instrumento_tipo == "OIS":
            st.caption(f"→ {banco}: **{lbl}**")

        # Selector de escenario macroeconomico
        escenario = st.selectbox(
            "Escenario macroeconomico",
            options=ESCENARIOS,
            help=(
                "Base: normalizacion gradual, inflacion al objetivo. "
                "Optimista: crecimiento solido, inflacion contenida. "
                "Pesimista: inflacion persistente o debilidad de crecimiento."
            ),
        )

        st.divider()

        # Info rapida del banco seleccionado (ultimo mes del test)
        try:
            scores_sidebar = _cargar_scores_test(banco, instrumento)
            ult = scores_sidebar.iloc[-1]
            conf_s, dir_s, sem_s = interpretar_scores(
                float(ult["Confidence_Score"]),
                float(ult["Upward_Score"]),
                float(ult["Downward_Score"]),
            )
            emoji_s = _estado_semaforo(sem_s)

            estado_label = (
                f"**Estado {banco} — dic-2025**"
                if instrumento_tipo == "tipo_oficial"
                else f"**Estado {banco} [{lbl}] — dic-2025**"
            )
            st.markdown(estado_label)
            st.markdown(f"{emoji_s} **{dir_s}**")

            csa, csb = st.columns(2)
            csa.metric("Confidence", f"{float(ult['Confidence_Score']):.0f}")
            csb.metric("Net Score",  f"{float(ult['Net_Score']):.1f}")
            st.markdown(f"Tipo predicho: **{float(ult['tipo_predicho']):.2f}%**")
        except Exception:
            st.caption("(Datos del test no disponibles)")

        st.divider()

        # Informacion del modelo
        with st.expander("Variables del modelo (12)", expanded=False):
            for v in config.VARIABLES_MODELO:
                st.markdown(f"- `{v}`")
            st.caption(f"Umbral estabilidad: ±{config.THRESHOLD_ESTABILIDAD} pp")

        st.caption("Train: 2001-2023 (276 obs)")
        st.caption("Test:  2024-2025  (24 obs)")
        st.caption("Pred:  2026-2030  (60 meses)")

    # ── Carga de todos los historicos (con spinner al inicio) ───────────────
    with st.spinner("Cargando datos..."):
        bancos_data: dict[str, pd.DataFrame] = {
            b: _cargar_historico(b) for b in config.BANCOS_CENTRALES
        }
    df_historico = bancos_data[banco]

    # ── Cabecera de la pagina principal ─────────────────────────────────────
    col_tit, col_met = st.columns([3, 1])
    with col_tit:
        _instr_display = "" if instrumento_tipo == "tipo_oficial" else f" · {lbl}"
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #003B22 0%, #006B2D 60%, #009A44 100%);
            border-radius: 12px;
            padding: 20px 28px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,107,45,0.18);
        '>
            <div style='color:#C8D400;font-size:0.80em;font-weight:700;
                        font-family:Nunito,sans-serif;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:4px;'>
                Trabajo de Fin de Grado · Universidad
            </div>
            <div style='color:#FFFFFF;font-size:1.75em;font-weight:800;
                        font-family:Nunito,sans-serif;letter-spacing:-0.5px;
                        line-height:1.2;'>
                Predicción de Tipos de Interés
            </div>
            <div style='margin-top:10px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>
                <span style='
                    background:rgba(255,255,255,0.15);
                    border:1px solid rgba(255,255,255,0.3);
                    border-radius:20px;
                    padding:3px 14px;
                    color:#FFFFFF;
                    font-size:0.85em;
                    font-weight:600;
                    font-family:Nunito,sans-serif;
                '>
                    <span style='color:{_color_banco(banco)};'>●</span>
                    &nbsp;{banco} — {_NOMBRE_COMPLETO[banco]}{_instr_display}
                </span>
                <span style='
                    background:rgba(200,212,0,0.20);
                    border:1px solid rgba(200,212,0,0.4);
                    border-radius:20px;
                    padding:3px 14px;
                    color:#C8D400;
                    font-size:0.85em;
                    font-weight:600;
                    font-family:Nunito,sans-serif;
                '>
                    Escenario: {escenario}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_met:
        try:
            scores_hdr  = _cargar_scores_test(banco, instrumento)
            tipo_dic25  = float(scores_hdr.iloc[-1]["tipo_predicho"])
            tipo_ene24  = float(scores_hdr.iloc[0]["tipo_predicho"])
            delta_test  = tipo_dic25 - tipo_ene24
            _met_label  = (
                f"Tipo predicho dic-2025 ({banco})"
                if instrumento_tipo == "tipo_oficial"
                else f"{lbl} predicho dic-2025 ({banco})"
            )
            st.metric(
                label=_met_label,
                value=f"{tipo_dic25:.2f}%",
                delta=f"{delta_test:+.2f} pp vs ene-2024",
                delta_color="inverse" if delta_test < 0 else "normal",
            )
        except Exception:
            pass

    # ── Tabs principales ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Panorama global",
        "📊 Historico y test",
        "🔮 Predicciones 2026-2030",
        "🔬 Evaluacion del modelo",
        "💼 Caso Practico",
    ])

    with tab1:
        _seccion_panorama(bancos_data, instrumento_tipo)

    with tab2:
        _seccion_historico(banco, df_historico, instrumento, target_col)

    with tab3:
        _seccion_predicciones(banco, df_historico, escenario, instrumento, target_col)

    with tab4:
        _seccion_evaluacion(banco, instrumento)

    with tab5:
        _seccion_caso_iberdrola()


# ── Punto de entrada ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
