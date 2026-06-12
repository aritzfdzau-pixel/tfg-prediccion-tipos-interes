"""
scoring.py — Sistema de scoring 0-100 (Fase 7)

Transforma las probabilidades del modelo de clasificación en dos indicadores
interpretables para el dashboard y la memoria del TFG:

  Confidence Score (0-100)
    Mide la certeza del modelo sobre la dirección predicha.
    Fórmula: (max_prob - 1/3) / (2/3) * 100
    · 0  = máxima incertidumbre (distribución uniforme: 33/33/33%)
    · 100 = certeza total (una clase con probabilidad 100%)
    Justificación: normalizar entre el mínimo teórico (1/3) y el máximo (1.0)
    da una escala intuitiva e independiente del número de clases.

  Upward Score (0-100)
    Probabilidad de subida de tipos escalada a porcentaje.
    Fórmula: P("Sube") * 100
    · 0  = el modelo asigna probabilidad cero a una subida
    · 100 = el modelo está seguro de que los tipos van a subir

  Scores complementarios (para el dashboard):
    Downward Score  = P("Baja")    * 100
    Stability Score = P("Estable") * 100
    Net Score       = (P("Sube") - P("Baja")) * 100  → rango -100 a +100

Funciones públicas:
  calcular_confidence_score(proba)          → np.ndarray
  calcular_upward_score(proba)              → np.ndarray
  calcular_downward_score(proba)            → np.ndarray
  calcular_stability_score(proba)           → np.ndarray
  calcular_net_score(proba)                 → np.ndarray
  construir_tabla_scores(...)               → pd.DataFrame
  interpretar_scores(confidence, upward, downward) → (str, str, str)
  predecir_completo(banco, X, fechas, y_real_reg, y_real_cls) → pd.DataFrame
  ejecutar_scoring_historico(banco)         → pd.DataFrame
  ejecutar_scoring_todos_bancos()           → dict

--------------------------------------------------------------------------------
NOTA PARA LA MEMORIA DEL TFG — Limitacion conocida del clasificador (2024-2025)
--------------------------------------------------------------------------------
Durante el periodo de test (2024-2025), el modelo de clasificacion presenta
tres comportamientos no idoneos que deben mencionarse en la seccion de
limitaciones metodologicas:

1. BCE y BoE: P_Baja < 3% durante todo el test pese a que ambos bancos
   realizaron recortes significativos en ese periodo.
   Causa raiz: el modelo solo usa niveles de OIS (no su derivada temporal).
   Durante el entrenamiento (2000-2023), OIS alto estaba correlado con ciclos
   de subidas (2006-2008, 2022-2023). En 2024, el OIS seguia alto pero el
   ciclo ya se habia invertido; sin una feature de momentum el modelo no
   distingue ambos contextos.

2. FED: desfase temporal de ~4-5 meses. El modelo detecta correctamente
   el entorno macroeconomico de bajadas pero con retraso, porque las
   variables macro (PIB, desempleo, inflacion) son publicadas con rezago
   y el modelo aprende su correlacion historica con los tipos.

3. ROC_AUC test = NaN: la clase "Sube" no aparece en ningun mes del test
   (2024-2025 fue un ciclo de bajadas exclusivo). La metrica OvR no puede
   calcularse para esa clase. El ROC_AUC de train (0.90-0.92) es la
   referencia valida de capacidad discriminativa.

Solucion posible (no implementada por decision metodologica del TFG):
   Anadir features de momentum: delta_tipo_3m = tipo_t - tipo_{t-3} y
   delta_OIS_1m = OIS_t - OIS_{t-1}. Esto reduciria el desfase del FED
   y permitiria a BCE/BoE cruzar la frontera de decision hacia "Baja".

Impacto en los scores del dashboard:
   El Confidence_Score y el Downward_Score para BCE/BoE en el periodo
   historico 2024-2025 subestiman la probabilidad de bajada. Para
   predicciones futuras 2026-2030 con escenarios variados, los tres
   estados son igualmente plausibles y el modelo opera con mayor
   fiabilidad.
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def _suf(instrumento: str) -> str:
    """Sufijo de archivo: '' para tipo_oficial, '_OIS' para OIS."""
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


# ---------------------------------------------------------------------------
# Indices de clases (config.CLASES = ["Baja", "Estable", "Sube"])
# ---------------------------------------------------------------------------
_IDX_BAJA    = config.CLASES.index("Baja")     # 0
_IDX_ESTABLE = config.CLASES.index("Estable")  # 1
_IDX_SUBE    = config.CLASES.index("Sube")     # 2


# ---------------------------------------------------------------------------
# Calculos de scores individuales
# ---------------------------------------------------------------------------

def calcular_confidence_score(probabilidades: np.ndarray) -> np.ndarray:
    """
    Confidence Score normalizado (0-100).

    Fórmula: (max_prob - 1/3) / (2/3) * 100

    Interpretación:
      · 0   → distribucion uniforme (33/33/33%): el modelo no sabe nada.
      · 50  → max_prob = 66.7%: confianza moderada.
      · 100 → max_prob = 100%: certeza absoluta.

    Ventaja frente a max_prob * 100: la escala empieza en 0 (incertidumbre
    maxima real) en lugar de 33.3, lo que hace los valores del dashboard
    directamente comparables entre si.

    Parámetro:
        probabilidades: array (n, 3) con columnas [P_Baja, P_Estable, P_Sube].

    Devuelve:
        array (n,) con valores en [0, 100], redondeado a 1 decimal.
    """
    proba = np.asarray(probabilidades)
    max_prob = np.max(proba, axis=1)
    score = (max_prob - 1.0 / 3.0) / (2.0 / 3.0) * 100.0
    return np.round(np.clip(score, 0.0, 100.0), 1)


def calcular_upward_score(probabilidades: np.ndarray) -> np.ndarray:
    """
    Upward Score = P("Sube") * 100  →  [0, 100].

    Indica la probabilidad especifica de subida de tipos.
    Util para el dashboard como indicador alcista puro.

    Parámetro:
        probabilidades: array (n, 3) en orden config.CLASES = [Baja, Estable, Sube].

    Devuelve:
        array (n,) con valores en [0, 100], redondeado a 1 decimal.
    """
    proba = np.asarray(probabilidades)
    return np.round(proba[:, _IDX_SUBE] * 100.0, 1)


def calcular_downward_score(probabilidades: np.ndarray) -> np.ndarray:
    """
    Downward Score = P("Baja") * 100  →  [0, 100].

    Indica la probabilidad especifica de bajada de tipos.
    Complementario al Upward Score en el dashboard.
    """
    proba = np.asarray(probabilidades)
    return np.round(proba[:, _IDX_BAJA] * 100.0, 1)


def calcular_stability_score(probabilidades: np.ndarray) -> np.ndarray:
    """
    Stability Score = P("Estable") * 100  →  [0, 100].

    Indica la probabilidad de que los tipos no cambien el siguiente mes.
    """
    proba = np.asarray(probabilidades)
    return np.round(proba[:, _IDX_ESTABLE] * 100.0, 1)


def calcular_net_score(probabilidades: np.ndarray) -> np.ndarray:
    """
    Net Score = (P("Sube") - P("Baja")) * 100  →  [-100, +100].

    Indicador de sesgo direccional neto:
      · Positivo  → sesgo alcista (mas probabilidad de subida que de bajada).
      · Cero      → sesgo neutral.
      · Negativo  → sesgo bajista.

    Util para representar la direccion esperada en una sola cifra.
    """
    proba = np.asarray(probabilidades)
    net = (proba[:, _IDX_SUBE] - proba[:, _IDX_BAJA]) * 100.0
    return np.round(net, 1)


# ---------------------------------------------------------------------------
# Tabla de scores completa
# ---------------------------------------------------------------------------

def construir_tabla_scores(
    fechas: pd.DatetimeIndex,
    y_pred_reg: np.ndarray,
    probabilidades: np.ndarray,
    y_real_reg: np.ndarray = None,
    y_real_cls: np.ndarray = None,
) -> pd.DataFrame:
    """
    Construye la tabla completa de predicciones y scores para el dashboard.

    Columnas del DataFrame resultante:
      · tipo_real       (solo si y_real_reg no es None)
      · tipo_predicho   : prediccion numerica del tipo (modelo de regresion)
      · direccion       : clase predicha ("Sube" / "Estable" / "Baja")
      · P_Baja          : probabilidad de bajada (%)
      · P_Estable       : probabilidad de estabilidad (%)
      · P_Sube          : probabilidad de subida (%)
      · Confidence_Score: certeza del modelo [0-100]
      · Upward_Score    : P(Sube) * 100 [0-100]
      · Downward_Score  : P(Baja) * 100 [0-100]
      · Stability_Score : P(Estable) * 100 [0-100]
      · Net_Score       : (P_Sube - P_Baja) * 100 [-100, +100]

    Parámetros:
        fechas:       DatetimeIndex con las fechas de las predicciones.
        y_pred_reg:   array (n,) con el tipo predicho por el regresor (%).
        probabilidades: array (n, 3) con [P_Baja, P_Estable, P_Sube].
        y_real_reg:   (opcional) array (n,) con el tipo real (% historico).
        y_real_cls:   (opcional) array (n,) con la clase real ("Sube"/"Baja"/"Estable").

    Devuelve:
        pd.DataFrame con indice = fechas.
    """
    proba = np.asarray(probabilidades)

    # Clase predicha = argmax de probabilidades
    idx_max = np.argmax(proba, axis=1)
    direccion = np.array(config.CLASES)[idx_max]

    # Scores
    confidence = calcular_confidence_score(proba)
    upward     = calcular_upward_score(proba)
    downward   = calcular_downward_score(proba)
    stability  = calcular_stability_score(proba)
    net        = calcular_net_score(proba)

    data = {}

    if y_real_reg is not None:
        data["tipo_real"]     = np.round(np.asarray(y_real_reg), 4)
    if y_real_cls is not None:
        data["clase_real"]    = np.asarray(y_real_cls)

    data["tipo_predicho"]    = np.round(np.asarray(y_pred_reg), 4)
    data["direccion"]        = direccion
    data["P_Baja"]           = np.round(proba[:, _IDX_BAJA]    * 100, 2)
    data["P_Estable"]        = np.round(proba[:, _IDX_ESTABLE] * 100, 2)
    data["P_Sube"]           = np.round(proba[:, _IDX_SUBE]    * 100, 2)
    data["Confidence_Score"] = confidence
    data["Upward_Score"]     = upward
    data["Downward_Score"]   = downward
    data["Stability_Score"]  = stability
    data["Net_Score"]        = net

    return pd.DataFrame(data, index=fechas)


# ---------------------------------------------------------------------------
# Interpretacion textual para el dashboard
# ---------------------------------------------------------------------------

def interpretar_scores(
    confidence: float,
    upward: float,
    downward: float,
) -> tuple:
    """
    Genera tres etiquetas textuales para mostrar en el dashboard.

    Parámetros:
        confidence: Confidence_Score de una sola observacion [0-100].
        upward:     Upward_Score [0-100].
        downward:   Downward_Score [0-100].

    Devuelve:
        (confianza_texto, direccion_texto, semaforo)
        · confianza_texto: "Muy alta" / "Alta" / "Moderada" / "Baja" / "Muy baja"
        · direccion_texto: "Sesgo alcista" / "Sesgo bajista" / "Neutral"
        · semaforo:        "🟢" / "🔴" / "🟡"  (para el dashboard)
    """
    # Nivel de confianza
    if confidence >= 75:
        confianza_texto = "Muy alta"
    elif confidence >= 55:
        confianza_texto = "Alta"
    elif confidence >= 35:
        confianza_texto = "Moderada"
    elif confidence >= 15:
        confianza_texto = "Baja"
    else:
        confianza_texto = "Muy baja"

    # Direccion neta
    diferencia = upward - downward
    if diferencia > 15:
        direccion_texto = "Sesgo alcista"
        semaforo = "[SUBE]"
    elif diferencia < -15:
        direccion_texto = "Sesgo bajista"
        semaforo = "[BAJA]"
    else:
        direccion_texto = "Neutral"
        semaforo = "[ESTABLE]"

    return confianza_texto, direccion_texto, semaforo


# ---------------------------------------------------------------------------
# Pipeline completo: regresion + clasificacion + scoring
# ---------------------------------------------------------------------------

def predecir_completo(
    banco: str,
    X: pd.DataFrame,
    fechas: pd.DatetimeIndex = None,
    y_real_reg: np.ndarray = None,
    y_real_cls: np.ndarray = None,
    tipo_actual=None,
    instrumento: str = "tipo_oficial",
) -> pd.DataFrame:
    """
    Carga ambos modelos serializados y produce la tabla completa de
    predicciones y scores para el banco indicado.

    El modelo de regresion predice DELTAS internamente. Si se proporciona
    tipo_actual, se reconstruye el nivel: tipo_predicho = tipo_actual + delta.
    Si tipo_actual es None (escenarios futuros), el nivel ya viene reconstruido.

    Parametros:
        banco:        "BCE", "BoE" o "FED".
        X:            DataFrame (n x 9) con las features del modelo.
        fechas:       DatetimeIndex (opcional; si None usa el indice de X).
        y_real_reg:   niveles reales (solo disponible para test historico).
        y_real_cls:   clases reales (solo disponible para test historico).
        tipo_actual:  Series/array con tipo_t actual (para reconstruir nivel).
        instrumento:  "tipo_oficial" (default) o "OIS".

    Devuelve:
        pd.DataFrame con todas las columnas de construir_tabla_scores().
    """
    from src.models.regression import (
        cargar_modelo as cargar_reg,
        predecir_regresion,
        reconstruir_nivel,
    )
    from src.models.classification import (
        cargar_modelo as cargar_cls,
        predecir_probabilidades,
    )

    if fechas is None:
        fechas = X.index

    # Cargar modelos correctos según instrumento
    modelo_reg              = cargar_reg(banco, instrumento=instrumento)
    modelo_cls, scaler_cls  = cargar_cls(banco, instrumento=instrumento)

    # Predicciones de delta
    y_pred_delta = predecir_regresion(modelo_reg, X)

    # Reconstruir nivel desde delta
    if tipo_actual is not None:
        y_pred_nivel = reconstruir_nivel(y_pred_delta, tipo_actual)
    else:
        # Fallback: si no hay tipo_actual, usar delta directamente
        # (no deberia ocurrir en uso normal — prediccion_futura.py pasa el nivel ya reconstruido)
        y_pred_nivel = y_pred_delta

    proba = predecir_probabilidades(modelo_cls, scaler_cls, X)

    return construir_tabla_scores(
        fechas         = fechas,
        y_pred_reg     = y_pred_nivel,   # tipo_predicho en la tabla = NIVEL
        probabilidades = proba,
        y_real_reg     = y_real_reg,     # y_real_reg debe ser NIVEL (pasado desde fuera)
        y_real_cls     = y_real_cls,
    )


# ---------------------------------------------------------------------------
# Scoring sobre el periodo historico de test
# ---------------------------------------------------------------------------

def ejecutar_scoring_historico(banco: str, instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Ejecuta el scoring sobre el conjunto de test (2024-01 a 2025-12).

    Carga datos procesados, aplica la division temporal y produce la
    tabla de scores completa comparando predicciones con valores reales.

    Guarda: Resultados/scores_test_{banco}[_OIS].csv

    Parámetros:
        banco:       "BCE", "BoE" o "FED".
        instrumento: "tipo_oficial" (default) o "OIS".

    Devuelve:
        pd.DataFrame con scores y predicciones del periodo de test.
    """
    from src.data.data_loader import cargar_datos_procesados
    from src.data.feature_engineering import preparar_X_y, dividir_temporal

    target_col = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    df = cargar_datos_procesados(banco)
    X, y_reg, y_cls, fechas, tipo_actual = preparar_X_y(df, banco, target_col=target_col)  # 5-tuple

    (X_train, X_test,
     y_train_reg, y_test_reg,     # deltas
     y_train_cls, y_test_cls,
     fechas_train, fechas_test) = dividir_temporal(X, y_reg, y_cls, fechas)

    # Separar tipo_actual para el test
    tipo_actual_test = tipo_actual.loc[fechas_test]

    # Nivel real del siguiente mes = tipo_actual_t + delta_real_t
    y_real_nivel_test = tipo_actual_test.values + y_test_reg.values

    tabla = predecir_completo(
        banco        = banco,
        X            = X_test,
        fechas       = fechas_test,
        y_real_reg   = y_real_nivel_test,   # nivel
        y_real_cls   = y_test_cls.values,
        tipo_actual  = tipo_actual_test,    # para reconstruccion de nivel predicho
        instrumento  = instrumento,
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTS_DIR / f"scores_test_{banco}{_suf(instrumento)}.csv"
    tabla.to_csv(ruta)

    return tabla


def ejecutar_scoring_todos_bancos(instrumento: str = "tipo_oficial") -> dict:
    """
    Ejecuta el scoring historico para los tres bancos y muestra el resumen.

    Parámetros:
        instrumento: "tipo_oficial" (default) o "OIS".

    Devuelve:
        dict {"BCE": df_BCE, "BoE": df_BoE, "FED": df_FED}
    """
    resultados = {}

    for banco in config.BANCOS_CENTRALES:
        print(f"\n{'='*62}")
        print(f"  SCORING — {banco}")
        print(f"{'='*62}")

        tabla = ejecutar_scoring_historico(banco, instrumento=instrumento)
        resultados[banco] = tabla

        # Resumen de scores medios en test
        conf_media  = tabla["Confidence_Score"].mean()
        up_media    = tabla["Upward_Score"].mean()
        down_media  = tabla["Downward_Score"].mean()
        net_media   = tabla["Net_Score"].mean()

        print(f"\n  Scores medios test (2024-2025):")
        print(f"    Confidence_Score : {conf_media:.1f}")
        print(f"    Upward_Score     : {up_media:.1f}")
        print(f"    Downward_Score   : {down_media:.1f}")
        print(f"    Net_Score        : {net_media:.1f}")

        # Tabla de predicciones con scores
        cols_show = ["tipo_real", "tipo_predicho", "direccion",
                     "Confidence_Score", "Upward_Score", "Downward_Score", "Net_Score"]
        print(f"\n  Predicciones + scores ({banco}):")
        print(tabla[cols_show].to_string())

        # Interpretacion del ultimo mes disponible
        ultimo = tabla.iloc[-1]
        conf_t, dir_t, sem = interpretar_scores(
            ultimo["Confidence_Score"],
            ultimo["Upward_Score"],
            ultimo["Downward_Score"],
        )
        print(f"\n  Ultimo mes ({tabla.index[-1].strftime('%Y-%m')}): "
              f"{sem} {dir_t} | Confianza: {conf_t} | "
              f"Net Score: {ultimo['Net_Score']:.1f}")

    return resultados


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    todos = ejecutar_scoring_todos_bancos()

    print("\n" + "="*62)
    print("  RESUMEN FINAL — SCORES MEDIOS TEST")
    print("="*62)
    filas = []
    for banco, tabla in todos.items():
        filas.append({
            "Banco":            banco,
            "Confidence_media": round(tabla["Confidence_Score"].mean(), 1),
            "Upward_media":     round(tabla["Upward_Score"].mean(), 1),
            "Downward_media":   round(tabla["Downward_Score"].mean(), 1),
            "Net_media":        round(tabla["Net_Score"].mean(), 1),
        })
    df_res = pd.DataFrame(filas)
    print(df_res.to_string(index=False))

    # Guardar resumen de scores
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(config.RESULTS_DIR / "resumen_scores.csv", index=False)
    print(f"\n  Resumen guardado: resumen_scores.csv")
