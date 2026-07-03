# -*- coding: utf-8 -*-
"""
Caso práctico Iberdrola: motor de cálculo compartido.

Compara 4 estrategias de refinanciación de los vencimientos 2026-2028 de
Iberdrola usando las predicciones del modelo (tipos oficiales + OIS).
Este módulo contiene únicamente funciones puras parametrizadas; lo importan
tanto `run_caso_iberdrola.py` (script CLI) como la tab "Caso Iberdrola" del
dashboard (`app.py`), de modo que exista una sola fuente de verdad.

Metodología:
- El coste de la deuda VARIABLE cada año = tipo oficial predicho + spread.
- El coste de FIJAR en el año t = media de la senda OIS predicha desde t
  hasta 2030 + spread (el swap se aproxima por el promedio de OIS esperados).
  Los swaps contratados en 2026 se valoran con la curva OIS del escenario
  base (es lo que cotiza el mercado hoy); desde 2027, con la del escenario.
- Los pesos de los escenarios se derivan del clasificador: verosimilitud
  direccional media de cada escenario, normalizada.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONSTANTES Y SUPUESTOS POR DEFECTO
# ============================================================

# Vencimientos reales de deuda de Iberdrola (M€, presentación FY2025)
REFINANCIACION_DEFAULT = {2026: 5392, 2027: 4123, 2028: 4424}

# Spread corporativo sobre el tipo de referencia (%).
# Calibrado con emisiones reales: bono verde mar-2026 a 4 años, cupón 3,125%
# vs tipo oficial BCE ~2,0% en esa fecha -> ~110 pb.
SPREAD_DEFAULT = 1.10

ANIO_INI, ANIO_FIN = 2026, 2030
ANIOS = list(range(ANIO_INI, ANIO_FIN + 1))

BANCOS = ["BCE", "BoE", "FED"]          # EUR, GBP, USD
ESCENARIOS = ["base", "optimista", "pesimista"]

# Umbral de dirección para clasificar la senda de un escenario (pp/mes)
UMBRAL_DIR_DEFAULT = 0.05
# Regla de giro de la estrategia guiada por el modelo (usa las DOS mitades
# del híbrido): fijar si P(Sube) anual > umbral_fijar (clasificador) o si
# la regresión predice subida acumulada > umbral_subida_reg el año siguiente.
UMBRAL_FIJAR_DEFAULT = 0.35
UMBRAL_SUBIDA_REG_DEFAULT = 0.25  # pp

# Mix por defecto del statu quo (cuentas anuales 2025)
PCT_FIJO_STATUQUO_DEFAULT = 0.772
DIVISAS_STATUQUO_DEFAULT = {"BCE": 0.55, "BoE": 0.25, "FED": 0.20}
DIVISAS_AGRESIVA_DEFAULT = {"BCE": 0.80, "BoE": 0.10, "FED": 0.10}
DIVISAS_MODELO_DEFAULT = {"BCE": 0.60, "BoE": 0.25, "FED": 0.15}


def construir_estrategias(
    divisas_statuquo: dict[str, float] | None = None,
    pct_fijo_statuquo: float = PCT_FIJO_STATUQUO_DEFAULT,
    divisas_agresiva: dict[str, float] | None = None,
    divisas_modelo: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Definición de las 4 estrategias: reparto por divisa y política
    fijo/variable. pct_fijo = proporción fijada en el año de emisión."""
    dv_sq = divisas_statuquo or dict(DIVISAS_STATUQUO_DEFAULT)
    return {
        "1. Statu quo": {
            "divisas": dv_sq,
            "politica": "mixta", "pct_fijo": pct_fijo_statuquo,
        },
        "2. Conservadora": {
            "divisas": dict(dv_sq),
            "politica": "fija", "pct_fijo": 1.00,
        },
        "3. Agresiva": {
            "divisas": divisas_agresiva or dict(DIVISAS_AGRESIVA_DEFAULT),
            "politica": "variable", "pct_fijo": 0.00,
        },
        "4. Guiada por el modelo": {
            "divisas": divisas_modelo or dict(DIVISAS_MODELO_DEFAULT),
            "politica": "dinamica", "pct_fijo": None,
        },
    }


# ============================================================
# CARGA DE PREDICCIONES
# ============================================================

def cargar_predicciones(res_dir: Path | str) -> dict:
    """Lee los 18 CSVs de predicciones (3 bancos × 3 escenarios × 2
    instrumentos) y devuelve un dict con:
      oficial[banco][esc] : Serie de medias anuales del tipo oficial predicho
      ois[banco][esc]     : Serie de medias anuales del OIS predicho
      probs[banco][esc]   : DataFrame mensual con tipo_predicho y P_* (%)
    Lanza FileNotFoundError si falta algún CSV."""
    res = Path(res_dir)
    oficial: dict = {}
    ois: dict = {}
    probs: dict = {}
    for b in BANCOS:
        oficial[b], ois[b], probs[b] = {}, {}, {}
        for e in ESCENARIOS:
            f_of = res / f"predicciones_futuras_{b}_{e}.csv"
            f_oi = res / f"predicciones_futuras_{b}_{e}_OIS.csv"
            df_of = pd.read_csv(f_of, index_col=0, parse_dates=True)
            df_oi = pd.read_csv(f_oi, index_col=0, parse_dates=True)
            oficial[b][e] = df_of.groupby(df_of.index.year)["tipo_predicho"].mean()
            ois[b][e] = df_oi.groupby(df_oi.index.year)["tipo_predicho"].mean()
            probs[b][e] = df_of[["tipo_predicho", "P_Baja", "P_Estable", "P_Sube"]]
    return {"oficial": oficial, "ois": ois, "probs": probs}


# ============================================================
# 1) PESOS DE ESCENARIOS DERIVADOS DEL CLASIFICADOR
# ============================================================

def verosimilitud_escenario(
    datos: dict, esc: str, umbral_dir: float = UMBRAL_DIR_DEFAULT
) -> float:
    """Prob. media que el clasificador asigna a la dirección que sigue
    la senda del escenario, promediada sobre meses y bancos."""
    vals = []
    for b in BANCOS:
        df = datos["probs"][b][esc]
        delta = df["tipo_predicho"].diff()
        for d, pb, pe, ps in zip(delta, df["P_Baja"], df["P_Estable"], df["P_Sube"]):
            if pd.isna(d):
                continue
            if d < -umbral_dir:
                vals.append(pb / 100)
            elif d > umbral_dir:
                vals.append(ps / 100)
            else:
                vals.append(pe / 100)
    return float(np.mean(vals))


def calcular_pesos(
    datos: dict, umbral_dir: float = UMBRAL_DIR_DEFAULT
) -> tuple[dict[str, float], dict[str, float]]:
    """Devuelve (verosimilitudes, pesos normalizados) por escenario."""
    verosim = {e: verosimilitud_escenario(datos, e, umbral_dir) for e in ESCENARIOS}
    total = sum(verosim.values())
    pesos = {e: v / total for e, v in verosim.items()}
    return verosim, pesos


def tabla_pesos(verosim: dict[str, float], pesos: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "Escenario": ESCENARIOS,
        "Verosimilitud_media": [round(verosim[e], 4) for e in ESCENARIOS],
        "Peso_normalizado": [round(pesos[e], 4) for e in ESCENARIOS],
    })


# ============================================================
# 2) MOTOR DE COSTE DE LAS ESTRATEGIAS
# ============================================================

def tipo_swap(datos: dict, banco: str, esc: str, anio: int) -> float:
    """Tipo fijo contratable en 'anio': media de la senda OIS predicha
    desde ese año hasta el fin del horizonte.
    CLAVE: un swap contratado en 2026 se paga a la curva que cotiza el
    mercado HOY (escenario base); los escenarios solo divergen después.
    A partir de 2027 el mercado ya ha repreciado -> curva del escenario."""
    esc_precio = "base" if anio == ANIO_INI else esc
    senda = datos["ois"][banco][esc_precio]
    return float(senda.loc[anio:ANIO_FIN].mean())


def anio_giro_modelo(
    datos: dict, banco: str, esc: str,
    umbral_fijar: float = UMBRAL_FIJAR_DEFAULT,
    umbral_subida_reg: float = UMBRAL_SUBIDA_REG_DEFAULT,
) -> int | None:
    """Primer año con señal de subida -> fijar. Señal híbrida:
    clasificador (P_Sube anual > umbral) O regresión (subida acumulada
    predicha > umbral el año siguiente). None = nunca gira."""
    df = datos["probs"][banco][esc]
    p_sube_anual = df.groupby(df.index.year)["P_Sube"].mean() / 100
    senda = datos["oficial"][banco][esc]
    for a in ANIOS:
        senal_clf = a in p_sube_anual.index and p_sube_anual.loc[a] > umbral_fijar
        senal_reg = (a + 1 in senda.index and
                     senda.loc[a + 1] - senda.loc[a] > umbral_subida_reg)
        if senal_clf or senal_reg:
            return a
    return None


def coste_estrategia(
    datos: dict, cfg: dict, esc: str,
    spread: float = SPREAD_DEFAULT,
    refinanciacion: dict[int, float] | None = None,
    umbral_fijar: float = UMBRAL_FIJAR_DEFAULT,
    umbral_subida_reg: float = UMBRAL_SUBIDA_REG_DEFAULT,
) -> float:
    """Coste financiero total 2026-2030 (M€) de refinanciar los vencimientos
    según la estrategia, bajo un escenario."""
    refi = refinanciacion or REFINANCIACION_DEFAULT
    oficial = datos["oficial"]
    total_M = 0.0
    for anio_emision, nominal in refi.items():
        for b, w in cfg["divisas"].items():
            tramo = nominal * w
            if cfg["politica"] == "dinamica":
                giro = anio_giro_modelo(datos, b, esc, umbral_fijar, umbral_subida_reg)
            for anio in range(anio_emision, ANIO_FIN + 1):
                if cfg["politica"] == "fija":
                    tasa = tipo_swap(datos, b, esc, anio_emision) + spread
                elif cfg["politica"] == "variable":
                    tasa = float(oficial[b][esc].loc[anio]) + spread
                elif cfg["politica"] == "mixta":
                    t_fijo = tipo_swap(datos, b, esc, anio_emision) + spread
                    t_var = float(oficial[b][esc].loc[anio]) + spread
                    tasa = cfg["pct_fijo"] * t_fijo + (1 - cfg["pct_fijo"]) * t_var
                else:  # dinamica: variable hasta que el modelo señala Sube
                    if giro is not None and anio >= max(giro, anio_emision):
                        tasa = tipo_swap(datos, b, esc, max(giro, anio_emision)) + spread
                    else:
                        tasa = float(oficial[b][esc].loc[anio]) + spread
                total_M += tramo * tasa / 100
    return total_M


def calcular_caso(
    datos: dict,
    estrategias: dict[str, dict] | None = None,
    pesos: dict[str, float] | None = None,
    spread: float = SPREAD_DEFAULT,
    refinanciacion: dict[int, float] | None = None,
    umbral_fijar: float = UMBRAL_FIJAR_DEFAULT,
    umbral_subida_reg: float = UMBRAL_SUBIDA_REG_DEFAULT,
    umbral_dir: float = UMBRAL_DIR_DEFAULT,
) -> dict:
    """Ejecuta el caso completo y devuelve:
      verosim, pesos      : dicts por escenario
      df_pesos            : tabla de pesos
      df_costes           : 4 estrategias × (coste por escenario + esperado)
      mejor               : nombre de la estrategia con menor coste esperado
      df_giros            : año de giro a fijo de la estrategia dinámica
      costes              : dict {estrategia: {escenario: coste_float}} sin redondear
    Si `pesos` es None se derivan del clasificador."""
    estrategias = estrategias or construir_estrategias()
    if pesos is None:
        verosim, pesos = calcular_pesos(datos, umbral_dir)
    else:
        verosim = {e: float("nan") for e in ESCENARIOS}
        total = sum(pesos.values())
        pesos = {e: pesos[e] / total for e in ESCENARIOS}

    costes: dict[str, dict[str, float]] = {}
    filas = []
    for nombre, cfg in estrategias.items():
        costes[nombre] = {
            e: coste_estrategia(datos, cfg, e, spread, refinanciacion,
                                umbral_fijar, umbral_subida_reg)
            for e in ESCENARIOS
        }
        fila = {"Estrategia": nombre}
        for e in ESCENARIOS:
            fila[f"Coste_{e} (M€)"] = round(costes[nombre][e], 0)
        fila["Coste_esperado (M€)"] = round(
            sum(pesos[e] * costes[nombre][e] for e in ESCENARIOS), 0)
        filas.append(fila)

    df_costes = pd.DataFrame(filas)
    mejor = df_costes.loc[df_costes["Coste_esperado (M€)"].idxmin(), "Estrategia"]

    giros = []
    for b in BANCOS:
        for e in ESCENARIOS:
            g = anio_giro_modelo(datos, b, e, umbral_fijar, umbral_subida_reg)
            giros.append({"Banco": b, "Escenario": e,
                          "Año_giro_a_fijo": g or "nunca (variable)"})
    df_giros = pd.DataFrame(giros)

    return {
        "verosim": verosim,
        "pesos": pesos,
        "df_pesos": tabla_pesos(verosim, pesos),
        "df_costes": df_costes,
        "mejor": mejor,
        "df_giros": df_giros,
        "costes": costes,
    }
