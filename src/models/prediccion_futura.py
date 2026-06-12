"""
prediccion_futura.py — Prediccion futura 2026-2030 con escenarios (Fase 8)

Genera predicciones de tipos de interes para BCE, BoE y FED en tres
escenarios macroeconomicos (Base, Optimista, Pesimista) para 2026-2030.

Pipeline por escenario:
  1. Definir valores de las 9 variables explicativas por ano y banco.
  2. Poblar y guardar los CSV de escenarios (escenarios/{escenario}_{banco}.csv).
  3. Ejecutar regresion + clasificacion + scoring sobre cada escenario.
  4. Guardar resultados en Resultados/predicciones_futuras_{banco}_{escenario}.csv

Valores de escenario basados en:
  - Punto de partida: ultimos valores observados (dic-2025)
  - BCE tipo ~2.0%, BoE tipo ~3.75%, FED tipo ~3.64%
  - Proyecciones macroeconomicas 2026-2030 (IMF WEO, consenso mercado)
  - Los valores anuales se propagan mensualmente (mismo metodo que el historico)

Funciones publicas:
  definir_escenarios()             -> dict anidado con valores por banco/escenario/anio
  poblar_y_guardar_escenarios()    -> guarda los 9 CSVs corregidos y poblados
  predecir_escenario(banco, esc)   -> pd.DataFrame con scores completos
  ejecutar_predicciones_futuras()  -> dict con todos los resultados
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def _suf(instrumento: str) -> str:
    """Sufijo de archivo: '' para tipo_oficial, '_OIS' para OIS."""
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


# ---------------------------------------------------------------------------
# Definicion de valores de escenario
# ---------------------------------------------------------------------------
# Estructura: ESCENARIOS[banco][escenario][anio] = {variable: valor}
#
# Notas metodologicas:
#   · Rating: eliminado del modelo (importancia RF = 0%, variable estática).
#   · OIS: proxy del tipo de interes; en escenarios futuros se asume que
#     OIS sigue de cerca el tipo oficial (relacion historica validada).
#   · Oro y Plata: variables globales, mismas para los 3 bancos.
#   · Gas: NGAS_EUR para BCE/BoE (EUR/MWh), NGAS_US para FED (USD/MMBtu).
#   · Las variables de PIB y PIBpc son tasas de crecimiento interanual (%).
#   · Inflacion y Desempleo en porcentaje (%).
#   · VIX es el indice de volatilidad implicita (puntos).
# ---------------------------------------------------------------------------

def definir_escenarios() -> dict:
    """
    Devuelve el diccionario con los valores macroeconomicos de cada escenario.

    Tres escenarios por banco:
      Base      : normalizacion gradual. Inflacion converge al objetivo.
                  Crecimiento moderado. Tipos estables o ligeramente a la baja.
      Optimista : crecimiento solido, inflacion bien anclada por debajo del
                  objetivo, desempleo reducido. Favorece tipos mas bajos.
      Pesimista : debilidad economica con inflacion persistente (stagflation),
                  desempleo al alza, volatilidad elevada. Tipos mas altos o
                  inciertos.

    Devuelve:
        dict con estructura ESCENARIOS[banco][escenario][anio] = {var: valor}
    """
    anos = list(range(config.PREDICT_START, config.PREDICT_END + 1))  # [2026..2030]

    # ------------------------------------------------------------------
    # Commodities globales (mismo valor BCE, BoE, FED por escenario)
    # Unidades: Oro USD/oz, Plata USD/oz, VIX puntos
    # ------------------------------------------------------------------
    comodidades = {
        "Base": {
            2026: {"VIX": 16.0, "Oro": 2850.0, "Plata": 32.0},
            2027: {"VIX": 15.5, "Oro": 2950.0, "Plata": 33.0},
            2028: {"VIX": 15.0, "Oro": 3050.0, "Plata": 34.0},
            2029: {"VIX": 15.0, "Oro": 3100.0, "Plata": 35.0},
            2030: {"VIX": 15.0, "Oro": 3150.0, "Plata": 36.0},
        },
        "Optimista": {
            2026: {"VIX": 13.0, "Oro": 3000.0, "Plata": 35.0},
            2027: {"VIX": 12.0, "Oro": 3100.0, "Plata": 37.0},
            2028: {"VIX": 11.5, "Oro": 3200.0, "Plata": 38.0},
            2029: {"VIX": 11.0, "Oro": 3300.0, "Plata": 39.0},
            2030: {"VIX": 11.0, "Oro": 3400.0, "Plata": 40.0},
        },
        "Pesimista": {
            2026: {"VIX": 24.0, "Oro": 2600.0, "Plata": 28.0},
            2027: {"VIX": 28.0, "Oro": 2500.0, "Plata": 26.0},
            2028: {"VIX": 26.0, "Oro": 2550.0, "Plata": 27.0},
            2029: {"VIX": 23.0, "Oro": 2650.0, "Plata": 28.0},
            2030: {"VIX": 21.0, "Oro": 2750.0, "Plata": 29.0},
        },
    }

    # ------------------------------------------------------------------
    # BCE — Zona Euro
    # Punto partida (dic-2025): tipo~2.0%, OIS~1.9%, Inf~2.3%, Desmp~6.1%
    # Gas: NGAS_EUR en EUR/MWh (rango historico tipico: 10-80)
    # ------------------------------------------------------------------
    bce = {
        "Base": {
            2026: {"PIB_growth": 1.4, "PIBpc_growth": 1.2, "Inflacion": 2.3, "Desempleo": 6.2, "Gas": 28.0, "OIS": 1.90},
            2027: {"PIB_growth": 1.6, "PIBpc_growth": 1.4, "Inflacion": 2.1, "Desempleo": 6.0, "Gas": 26.0, "OIS": 1.75},
            2028: {"PIB_growth": 1.7, "PIBpc_growth": 1.5, "Inflacion": 2.0, "Desempleo": 5.9, "Gas": 25.0, "OIS": 1.70},
            2029: {"PIB_growth": 1.8, "PIBpc_growth": 1.6, "Inflacion": 2.0, "Desempleo": 5.8, "Gas": 24.0, "OIS": 1.70},
            2030: {"PIB_growth": 1.8, "PIBpc_growth": 1.6, "Inflacion": 2.0, "Desempleo": 5.7, "Gas": 24.0, "OIS": 1.70},
        },
        "Optimista": {
            2026: {"PIB_growth": 2.5, "PIBpc_growth": 2.2, "Inflacion": 1.9, "Desempleo": 5.8, "Gas": 22.0, "OIS": 1.50},
            2027: {"PIB_growth": 2.8, "PIBpc_growth": 2.5, "Inflacion": 1.7, "Desempleo": 5.5, "Gas": 20.0, "OIS": 1.40},
            2028: {"PIB_growth": 2.9, "PIBpc_growth": 2.6, "Inflacion": 1.6, "Desempleo": 5.3, "Gas": 19.0, "OIS": 1.35},
            2029: {"PIB_growth": 3.0, "PIBpc_growth": 2.7, "Inflacion": 1.6, "Desempleo": 5.1, "Gas": 18.0, "OIS": 1.35},
            2030: {"PIB_growth": 3.0, "PIBpc_growth": 2.7, "Inflacion": 1.5, "Desempleo": 5.0, "Gas": 18.0, "OIS": 1.30},
        },
        "Pesimista": {
            2026: {"PIB_growth": 0.4, "PIBpc_growth": 0.1, "Inflacion": 3.3, "Desempleo": 7.1, "Gas": 45.0, "OIS": 2.70},
            2027: {"PIB_growth": 0.2, "PIBpc_growth":-0.1, "Inflacion": 3.8, "Desempleo": 7.8, "Gas": 55.0, "OIS": 3.20},
            2028: {"PIB_growth": 0.5, "PIBpc_growth": 0.2, "Inflacion": 3.4, "Desempleo": 8.0, "Gas": 50.0, "OIS": 3.00},
            2029: {"PIB_growth": 0.8, "PIBpc_growth": 0.5, "Inflacion": 3.0, "Desempleo": 7.8, "Gas": 45.0, "OIS": 2.70},
            2030: {"PIB_growth": 1.0, "PIBpc_growth": 0.7, "Inflacion": 2.6, "Desempleo": 7.5, "Gas": 40.0, "OIS": 2.50},
        },
    }

    # ------------------------------------------------------------------
    # BoE — Reino Unido
    # Punto partida (dic-2025): tipo~3.75%, OIS~3.6%, Inf~2.8%, Desmp~4.2%
    # Gas: NGAS_EUR (misma fuente europea, en EUR/MWh)
    # ------------------------------------------------------------------
    boe = {
        "Base": {
            2026: {"PIB_growth": 1.2, "PIBpc_growth": 0.9, "Inflacion": 2.8, "Desempleo": 4.4, "Gas": 28.0, "OIS": 3.60},
            2027: {"PIB_growth": 1.4, "PIBpc_growth": 1.1, "Inflacion": 2.4, "Desempleo": 4.3, "Gas": 26.0, "OIS": 3.40},
            2028: {"PIB_growth": 1.6, "PIBpc_growth": 1.3, "Inflacion": 2.2, "Desempleo": 4.2, "Gas": 25.0, "OIS": 3.20},
            2029: {"PIB_growth": 1.7, "PIBpc_growth": 1.4, "Inflacion": 2.1, "Desempleo": 4.1, "Gas": 24.0, "OIS": 3.10},
            2030: {"PIB_growth": 1.8, "PIBpc_growth": 1.5, "Inflacion": 2.0, "Desempleo": 4.0, "Gas": 24.0, "OIS": 3.00},
        },
        "Optimista": {
            2026: {"PIB_growth": 2.2, "PIBpc_growth": 1.9, "Inflacion": 2.2, "Desempleo": 4.0, "Gas": 22.0, "OIS": 3.20},
            2027: {"PIB_growth": 2.5, "PIBpc_growth": 2.2, "Inflacion": 1.9, "Desempleo": 3.8, "Gas": 20.0, "OIS": 2.90},
            2028: {"PIB_growth": 2.7, "PIBpc_growth": 2.4, "Inflacion": 1.8, "Desempleo": 3.7, "Gas": 19.0, "OIS": 2.70},
            2029: {"PIB_growth": 2.8, "PIBpc_growth": 2.5, "Inflacion": 1.7, "Desempleo": 3.6, "Gas": 18.0, "OIS": 2.60},
            2030: {"PIB_growth": 2.8, "PIBpc_growth": 2.5, "Inflacion": 1.6, "Desempleo": 3.5, "Gas": 18.0, "OIS": 2.50},
        },
        "Pesimista": {
            2026: {"PIB_growth": 0.3, "PIBpc_growth": 0.0, "Inflacion": 3.8, "Desempleo": 5.0, "Gas": 45.0, "OIS": 4.20},
            2027: {"PIB_growth": 0.1, "PIBpc_growth":-0.2, "Inflacion": 4.2, "Desempleo": 5.6, "Gas": 55.0, "OIS": 4.60},
            2028: {"PIB_growth": 0.4, "PIBpc_growth": 0.1, "Inflacion": 3.8, "Desempleo": 5.8, "Gas": 50.0, "OIS": 4.40},
            2029: {"PIB_growth": 0.7, "PIBpc_growth": 0.4, "Inflacion": 3.3, "Desempleo": 5.6, "Gas": 46.0, "OIS": 4.10},
            2030: {"PIB_growth": 1.0, "PIBpc_growth": 0.7, "Inflacion": 2.8, "Desempleo": 5.4, "Gas": 42.0, "OIS": 3.80},
        },
    }

    # ------------------------------------------------------------------
    # FED — Estados Unidos
    # Punto partida (dic-2025): tipo~3.64%, OIS~3.5%, Inf~2.6%, Desmp~4.1%
    # Gas: NGAS_US (Henry Hub, USD/MMBtu; rango historico tipico: 1.5-9)
    # ------------------------------------------------------------------
    fed = {
        "Base": {
            2026: {"PIB_growth": 2.1, "PIBpc_growth": 1.8, "Inflacion": 2.6, "Desempleo": 4.2, "Gas": 3.50, "OIS": 3.50},
            2027: {"PIB_growth": 2.2, "PIBpc_growth": 1.9, "Inflacion": 2.3, "Desempleo": 4.1, "Gas": 3.40, "OIS": 3.30},
            2028: {"PIB_growth": 2.2, "PIBpc_growth": 1.9, "Inflacion": 2.1, "Desempleo": 4.0, "Gas": 3.30, "OIS": 3.20},
            2029: {"PIB_growth": 2.3, "PIBpc_growth": 2.0, "Inflacion": 2.0, "Desempleo": 3.9, "Gas": 3.20, "OIS": 3.10},
            2030: {"PIB_growth": 2.3, "PIBpc_growth": 2.0, "Inflacion": 2.0, "Desempleo": 3.9, "Gas": 3.20, "OIS": 3.00},
        },
        "Optimista": {
            2026: {"PIB_growth": 3.0, "PIBpc_growth": 2.7, "Inflacion": 2.2, "Desempleo": 3.8, "Gas": 3.20, "OIS": 3.10},
            2027: {"PIB_growth": 3.2, "PIBpc_growth": 2.9, "Inflacion": 1.9, "Desempleo": 3.6, "Gas": 3.00, "OIS": 2.90},
            2028: {"PIB_growth": 3.3, "PIBpc_growth": 3.0, "Inflacion": 1.8, "Desempleo": 3.5, "Gas": 2.90, "OIS": 2.80},
            2029: {"PIB_growth": 3.3, "PIBpc_growth": 3.0, "Inflacion": 1.8, "Desempleo": 3.4, "Gas": 2.80, "OIS": 2.70},
            2030: {"PIB_growth": 3.4, "PIBpc_growth": 3.1, "Inflacion": 1.7, "Desempleo": 3.3, "Gas": 2.80, "OIS": 2.60},
        },
        "Pesimista": {
            2026: {"PIB_growth": 0.8, "PIBpc_growth": 0.5, "Inflacion": 3.5, "Desempleo": 5.0, "Gas": 4.50, "OIS": 4.20},
            2027: {"PIB_growth": 0.5, "PIBpc_growth": 0.2, "Inflacion": 4.0, "Desempleo": 5.6, "Gas": 5.00, "OIS": 4.60},
            2028: {"PIB_growth": 0.8, "PIBpc_growth": 0.5, "Inflacion": 3.6, "Desempleo": 5.8, "Gas": 4.80, "OIS": 4.40},
            2029: {"PIB_growth": 1.2, "PIBpc_growth": 0.9, "Inflacion": 3.0, "Desempleo": 5.5, "Gas": 4.50, "OIS": 4.10},
            2030: {"PIB_growth": 1.5, "PIBpc_growth": 1.2, "Inflacion": 2.6, "Desempleo": 5.2, "Gas": 4.20, "OIS": 3.80},
        },
    }

    return {"BCE": bce, "BoE": boe, "FED": fed, "_comodidades": comodidades}


# ---------------------------------------------------------------------------
# Construccion y guardado de CSVs de escenarios
# ---------------------------------------------------------------------------

def poblar_y_guardar_escenarios() -> None:
    """
    Genera los 9 CSVs de escenarios (Base/Optimista/Pesimista x BCE/BoE/FED)
    con los valores macroeconomicos definidos en definir_escenarios().

    Corrige ademas el error de estructura de los CSV originales
    (trailing comma que desplazaba todas las columnas).

    Guarda en: escenarios/escenario_{escenario_lower}_{banco}.csv
    """
    datos = definir_escenarios()
    comodidades = datos["_comodidades"]

    nombres_esc = ["Base", "Optimista", "Pesimista"]
    anos = list(range(config.PREDICT_START, config.PREDICT_END + 1))
    meses = list(range(1, 13))

    config.SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    for banco in config.BANCOS_CENTRALES:
        rating = config.RATINGS[banco]

        for escenario in nombres_esc:
            filas = []
            datos_banco = datos[banco][escenario]

            for anio in anos:
                val_anio    = datos_banco[anio]
                val_comod   = comodidades[escenario][anio]
                for mes in meses:
                    filas.append({
                        "Anio":         anio,
                        "Mes":          mes,
                        "Banco":        banco,
                        "Escenario":    escenario,
                        # 9 features del modelo (Rating eliminado — importancia 0%)
                        "PIB_growth":   val_anio["PIB_growth"],
                        "PIBpc_growth": val_anio["PIBpc_growth"],
                        "Inflacion":    val_anio["Inflacion"],
                        "Desempleo":    val_anio["Desempleo"],
                        "VIX":          val_comod["VIX"],
                        "Oro":          val_comod["Oro"],
                        "Plata":        val_comod["Plata"],
                        "Gas":          val_anio["Gas"],
                        "OIS":          val_anio["OIS"],
                    })

            df = pd.DataFrame(filas)
            nombre_esc_lower = escenario.lower().replace("ó", "o")
            ruta = config.SCENARIOS_DIR / f"escenario_{nombre_esc_lower}_{banco}.csv"
            df.to_csv(ruta, index=False)

    print(f"  9 escenarios generados en {config.SCENARIOS_DIR}")


def cargar_escenario(banco: str, escenario: str) -> pd.DataFrame:
    """
    Carga el CSV de un escenario y devuelve un DataFrame con:
      - Indice: pd.DatetimeIndex mensual (primer dia de cada mes)
      - Columnas: exactamente config.VARIABLES_MODELO (9 features)

    Parametros:
        banco:     "BCE", "BoE" o "FED"
        escenario: "Base", "Optimista" o "Pesimista"

    Devuelve:
        (df_features, fechas)
        · df_features: DataFrame (60 x 9) listo para predecir.
        · fechas:      DatetimeIndex (60,) con fechas mensuales.
    """
    nombre_esc_lower = escenario.lower().replace("ó", "o")
    ruta = config.SCENARIOS_DIR / f"escenario_{nombre_esc_lower}_{banco}.csv"

    if not ruta.exists():
        raise FileNotFoundError(
            f"Escenario no encontrado: {ruta}\n"
            "Ejecuta primero poblar_y_guardar_escenarios()"
        )

    df = pd.read_csv(ruta)

    # Construir DatetimeIndex
    fechas = pd.to_datetime(
        df["Anio"].astype(str) + "-" + df["Mes"].astype(str).str.zfill(2) + "-01"
    )

    # Calcular OIS_taylor desde los datos del escenario (Inflacion + PIB_growth).
    # La Regla de Taylor convierte las hipotesis macro del escenario en un proxy
    # del tipo neutral, sin necesidad de fijar el OIS directamente.
    # Formula: OIS_taylor = max(0, r* + 1.5*(pi - pi*) + 0.5*(PIB_growth - y*))
    p = config.TAYLOR_PARAMS[banco]
    df["OIS_taylor"] = (
        p["r_star"]
        + 1.5 * (df["Inflacion"] - p["pi_star"])
        + 0.5 * (df["PIB_growth"] - p["y_star"])
    ).clip(lower=0.0)

    # brecha_taylor: OIS_taylor - OIS_escenario
    # Para escenarios futuros se usa el OIS del escenario como proxy del tipo
    # oficial esperado (el escenario define explicitamente donde se espera que
    # esten los tipos, y la brecha mide la desviacion respecto al equilibrio Taylor).
    df["brecha_taylor"] = df["OIS_taylor"] - df["OIS"]

    # Features de inercia para escenarios futuros:
    # Se inicializan a 0 porque para el horizonte 2026-2030 la dinamica de
    # largo plazo (Taylor Rule, brecha, macro) domina sobre la inercia mensual.
    # Asumir pausa inicial es consistente con la transicion gradual del blend.
    df["meses_sin_cambio"] = 0
    df["delta_tipo_lag1"]  = 0.0

    # Seleccionar exactamente las 12 features del modelo (Rating excluido)
    X = df[config.VARIABLES_MODELO].copy()
    X.index = fechas

    return X, fechas


# ---------------------------------------------------------------------------
# Prediccion por escenario
# ---------------------------------------------------------------------------

def predecir_escenario(banco: str, escenario: str, instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Ejecuta la prediccion completa (clasificacion + scoring Taylor Rule)
    sobre un escenario futuro.

    Para el nivel predicho se usa la Regla de Taylor (OIS_taylor) con
    transicion suave desde el tipo real de dic-2025 (tipo_oficial u OIS
    según el instrumento seleccionado).

    Parámetros:
        banco:       "BCE", "BoE" o "FED"
        escenario:   "Base", "Optimista" o "Pesimista"
        instrumento: "tipo_oficial" (default) o "OIS"

    Guarda: Resultados/predicciones_futuras_{banco}_{escenario_lower}[_OIS].csv

    Devuelve:
        pd.DataFrame con todas las columnas de construir_tabla_scores()
        mas las columnas Banco y Escenario.
    """
    from src.models.classification import (
        cargar_modelo as cargar_cls,
        predecir_probabilidades,
    )
    from src.models.scoring import construir_tabla_scores

    X_feat, fechas = cargar_escenario(banco, escenario)

    # Para el horizonte 2026-2030, se usa la Regla de Taylor directamente como
    # prediccion del NIVEL del tipo de interes, en lugar del modelo ML.
    #
    # Justificacion metodologica:
    #   · El RF entrenado sobre niveles historicos tiene un sesgo hacia cero porque
    #     el 70% de las observaciones de entrenamiento (2009-2023) corresponden a
    #     periodos con tipos oficiales en el rango 0-1%. Este sesgo hace que el RF
    #     no extrapole correctamente a escenarios con OIS_taylor > 2%.
    #   · La Regla de Taylor es el modelo teorico de referencia que los propios
    #     bancos centrales utilizan para comunicar su funcion de reaccion. Usarla
    #     directamente para el horizonte largo es mas transparente y defendible.
    #   · Para predicciones de 1 mes (test period), el modelo de delta aprende
    #     los saltos reales de politica monetaria desde el historico (MAE < 0.17pp).
    #   · Este diseno hibrido (Taylor Rule para nivel, ML para direccion) es comun
    #     en la literatura de prediccion de politica monetaria
    #     (Clarida et al. 1999; Rudebusch 2002).
    #
    # OIS_taylor ya calculado en X_feat por cargar_escenario() y sigue
    # la formula: max(0, r* + 1.5*(pi - pi*) + 0.5*(PIB_growth - y*))
    tipos_nivel = X_feat["OIS_taylor"].values.copy()

    # ── Transicion suave desde el tipo real de dic-2025 ─────────────────────
    # La Regla de Taylor puede diferir del tipo oficial en el arranque porque
    # los bancos centrales ajustan gradualmente (tipicamente 0.25 pp por reunion).
    # Interpolamos linealmente durante los primeros 12 meses desde el tipo real
    # observado en dic-2025 hacia el nivel de equilibrio de Taylor.
    # Esto evita el salto artificial en el grafico y refleja la dinamica real
    # de ajuste de politica monetaria.
    # Tipo/OIS de partida: último valor real del test (dic-2025)
    # Para OIS se usa scores_test_{banco}_OIS.csv; para tipo_oficial el estándar.
    try:
        _ruta_scores = config.RESULTS_DIR / f"scores_test_{banco}{_suf(instrumento)}.csv"
        _scores_hist = pd.read_csv(str(_ruta_scores), index_col="Fecha")
        _tipo_raw = float(_scores_hist.iloc[-1]["tipo_real"])
        if np.isnan(_tipo_raw):
            raise ValueError("tipo_real es NaN en la ultima fila del CSV de scores")
        tipo_dic25 = _tipo_raw
    except Exception:
        tipo_dic25 = tipos_nivel[0]   # fallback: sin transicion

    _N_BLEND = 12   # meses de transicion (todo el año 2026)
    for _i in range(min(_N_BLEND, len(tipos_nivel))):
        # (_i + 1) / _N_BLEND: llega exactamente a 1.0 en el mes 12 (dic-2026)
        _alpha = (_i + 1) / _N_BLEND    # 1/12 en ene-2026, 12/12=1.0 en dic-2026
        tipos_nivel[_i] = (1.0 - _alpha) * tipo_dic25 + _alpha * tipos_nivel[_i]
    tipos_nivel = np.clip(tipos_nivel, 0.0, 25.0)

    # Clasificacion — carga el modelo correcto según instrumento
    modelo_cls, scaler_cls = cargar_cls(banco, instrumento=instrumento)

    # ── Valores iniciales de features dinamicas desde el test historico ─────
    # meses_sin_cambio y delta_tipo_lag1 se recalculan iterativamente para que
    # la señal de inercia se acumule correctamente en el horizonte 2026-2030.
    # Sin este ajuste, ambas features valdrían 0 en los 60 meses, ignorando
    # que el banco puede llevar meses en pausa al inicio de 2026.
    _meses_sc_init  = 0    # contador de meses consecutivos sin cambio (desde test)
    _delta_lag_init = 0.0  # delta del ultimo mes del test (dic-2025)
    try:
        _ruta_sc_ini = config.RESULTS_DIR / f"scores_test_{banco}{_suf(instrumento)}.csv"
        _sc_ini = pd.read_csv(str(_ruta_sc_ini), index_col="Fecha")
        _tipos_hist = _sc_ini["tipo_real"].values
        # delta_tipo_lag1 para ene-2026 = cambio en dic-2025 vs nov-2025
        if len(_tipos_hist) >= 2:
            _delta_lag_init = float(_tipos_hist[-1] - _tipos_hist[-2])
        # meses_sin_cambio: contar hacia atras cuantos meses consecutivos tuvieron
        # |delta| < 0.001 (banco sin moverse) hasta dic-2025 (inclusive)
        _sc_c = 0
        for _i in range(len(_tipos_hist) - 1, 0, -1):
            _d = abs(_tipos_hist[_i] - _tipos_hist[_i - 1])
            if _d < 0.001:
                _sc_c += 1
            else:
                break
        _meses_sc_init = _sc_c
    except Exception:
        pass

    # ── Loop iterativo: predecir mes a mes actualizando features dinamicas ──
    # Para cada mes t:
    #   1. Sobreescribir meses_sin_cambio[t] y delta_tipo_lag1[t] con valores
    #      acumulados desde el test + predicciones anteriores.
    #   2. Predecir probabilidades con el clasificador.
    #   3. Actualizar el contador para el mes siguiente:
    #      · Si dir = "Estable": meses_sc += 1  (banco no mueve tipos)
    #      · Si dir = "Baja/Sube": meses_sc = 0 (cambio → reset inercia)
    #   4. Actualizar delta_tipo_lag1 como diferencia de niveles Taylor predichos.
    proba_list  = []
    meses_sc    = _meses_sc_init
    delta_lag   = _delta_lag_init
    tipo_prev   = tipo_dic25

    for t in range(len(fechas)):
        # Construir features del mes t con valores dinamicos
        X_t = X_feat.iloc[[t]].copy()
        X_t["meses_sin_cambio"] = float(meses_sc)
        X_t["delta_tipo_lag1"]  = float(delta_lag)

        # Prediccion de probabilidades para el mes t
        proba_t = predecir_probabilidades(modelo_cls, scaler_cls, X_t)
        proba_list.append(proba_t[0])

        # Direccion predicha (argmax de probabilidades)
        dir_t = config.CLASES[int(np.argmax(proba_t[0]))]

        # Actualizar contador de inercia para el mes t+1
        if dir_t == "Estable":
            meses_sc += 1
        else:
            meses_sc = 0

        # Actualizar delta para el mes t+1 (cambio en nivel Taylor predicho)
        tipo_actual = float(tipos_nivel[t])
        delta_lag   = tipo_actual - tipo_prev
        tipo_prev   = tipo_actual

    proba = np.array(proba_list)   # (60, 3)

    # Construir tabla de scores con niveles predichos (sin y_real ya que es futuro)
    tabla = construir_tabla_scores(
        fechas         = fechas,
        y_pred_reg     = tipos_nivel,
        probabilidades = proba,
    )

    # Metadatos
    tabla.insert(0, "Banco",     banco)
    tabla.insert(1, "Escenario", escenario)

    # Guardar
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nombre_esc_lower = escenario.lower().replace("ó", "o")
    ruta = config.RESULTS_DIR / f"predicciones_futuras_{banco}_{nombre_esc_lower}{_suf(instrumento)}.csv"
    tabla.to_csv(ruta)

    return tabla


# ---------------------------------------------------------------------------
# Pipeline completo: todos los bancos y escenarios
# ---------------------------------------------------------------------------

def ejecutar_predicciones_futuras(instrumento: str = "tipo_oficial") -> dict:
    """
    Ejecuta las predicciones futuras para las 9 combinaciones
    (3 bancos x 3 escenarios) y muestra un resumen por consola.

    Parámetros:
        instrumento: "tipo_oficial" (default) o "OIS".

    Devuelve:
        dict anidado: resultados[banco][escenario] = pd.DataFrame
    """
    # Paso 1: Generar y guardar CSVs de escenarios (los mismos para ambos instrumentos)
    print("Generando CSVs de escenarios...")
    poblar_y_guardar_escenarios()

    escenarios = ["Base", "Optimista", "Pesimista"]
    resultados = {b: {} for b in config.BANCOS_CENTRALES}

    for banco in config.BANCOS_CENTRALES:
        print(f"\n{'='*62}")
        print(f"  PREDICCION FUTURA — {banco} [{instrumento}]")
        print(f"{'='*62}")

        for escenario in escenarios:
            tabla = predecir_escenario(banco, escenario, instrumento=instrumento)
            resultados[banco][escenario] = tabla

            # Resumen anual: tipo predicho promedio por ano
            tabla_resumen = tabla.copy()
            tabla_resumen["Anio"] = tabla_resumen.index.year
            resumen_anual = tabla_resumen.groupby("Anio").agg(
                tipo_medio   = ("tipo_predicho",    "mean"),
                conf_media   = ("Confidence_Score", "mean"),
                net_medio    = ("Net_Score",        "mean"),
                dir_dominante = ("direccion",       lambda x: x.value_counts().index[0]),
            ).round(2)

            print(f"\n  [{escenario}]")
            print(f"  {'Ano':>4}  {'Tipo_pred(%)':>12}  {'Confidence':>10}  {'Net_Score':>9}  {'Direccion':>8}")
            for anio, row in resumen_anual.iterrows():
                print(f"  {anio:>4}  {row.tipo_medio:>12.2f}  {row.conf_media:>10.1f}"
                      f"  {row.net_medio:>9.1f}  {row.dir_dominante:>8}")

        nombre_esc_lower = "base"
        print(f"\n  Archivos guardados: predicciones_futuras_{banco}_*.csv")

    # Paso 2: Tabla comparativa final entre escenarios (tipo predicho 2030)
    print(f"\n{'='*62}")
    print("  COMPARATIVA — Tipo predicho dic-2030")
    print(f"{'='*62}")
    print(f"  {'Banco':>6}  {'Base':>8}  {'Optimista':>10}  {'Pesimista':>10}")
    for banco in config.BANCOS_CENTRALES:
        vals = {}
        for esc in escenarios:
            df_esc = resultados[banco][esc]
            # Ultimo mes = diciembre 2030
            ultimo = df_esc[df_esc.index.year == 2030].iloc[-1]
            vals[esc] = ultimo["tipo_predicho"]
        print(f"  {banco:>6}  {vals['Base']:>8.2f}  {vals['Optimista']:>10.2f}  {vals['Pesimista']:>10.2f}")

    return resultados


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    resultados = ejecutar_predicciones_futuras()
