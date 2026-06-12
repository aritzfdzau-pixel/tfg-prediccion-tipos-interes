"""
feature_engineering.py — Preparación de variables objetivo y división temporal
(Fases 3 y 4)

Recibe el DataFrame mensual de data_loader.cargar_datos_procesados(banco) y:
  · Construye las variables objetivo (regresión y clasificación)
  · Calcula PIB_growth y PIBpc_growth a partir de los valores anuales brutos
  · Selecciona y alinea las 9 variables explicativas del modelo
  · Divide cronológicamente en train/test

FUGA TEMPORAL — puntos críticos:
  · y_regression = delta = tipo_{t+1} - tipo_t: se crea con shift(-1), nunca con datos del futuro.
  · PIB_growth usa solo pct_change() sobre valores pasados.
  · El StandardScaler de clasificación se ajusta SOLO sobre X_train (en classification.py).
  · dividir_temporal() usa corte cronológico; nunca shuffle aleatorio.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ---------------------------------------------------------------------------
# Variables objetivo
# ---------------------------------------------------------------------------

def crear_y_regresion(df: pd.DataFrame, target_col: str = "tipo_oficial") -> pd.Series:
    """
    Variable objetivo de regresion: CAMBIO del tipo el mes siguiente.
    y_t = tipo_{t+1} - tipo_t  (delta en puntos porcentuales)

    Justificacion: la variacion mensual es estacionaria (oscila en torno a 0,
    con saltos discretos de 0, +-0.25, +-0.50, +-1.00 pp), a diferencia del nivel
    que tiene raiz unitaria. Modelar el cambio permite al RF aprender CUANDO
    y CUANTO mueven los bancos centrales.

    El ultimo mes del dataset queda con NaN (no hay mes siguiente).
    Esa fila se elimina en preparar_X_y().

    Parametros:
        df:         DataFrame mensual con columna target_col.
        target_col: nombre de la columna del tipo oficial (por defecto "tipo_oficial").

    Devuelve:
        pd.Series alineada con df.index, dtype float64.
    """
    tipo_next = df[target_col].shift(-1)
    y = tipo_next - df[target_col]
    y.name = "y_regression"
    return y



# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _calcular_crecimiento_anual(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Calcula la tasa de crecimiento interanual de una variable forward-filleada,
    usando el valor del AÑO ANTERIOR para evitar look-ahead temporal.

    El dataset de data_loader replica el valor anual a los 12 meses del año.
    El Banco Mundial publica el PIB definitivo del año t con varios meses de
    retraso (normalmente en el año t+1). Por tanto, usar PIB_growth_t para
    predecir meses del año t constituye look-ahead: el dato no está disponible
    hasta que el año t termina. La corrección consiste en usar PIB_growth_{t-1}
    (el crecimiento del año anterior, definitivamente publicado en enero de t).

    Para obtener la tasa de crecimiento sin look-ahead:
      1. Extraer un valor por año (groupby year → first).
      2. Aplicar pct_change() entre años consecutivos → growth_t.
      3. Aplicar shift(1) → growth_lag1_t = growth_{t-1}  ← sin look-ahead.
      4. Mapear el resultado de vuelta a todos los meses del año.

    Los dos primeros años disponibles (1999 y 2000) tendrán NaN:
      · 1999: no hay dato de 1998 para calcular growth_1999.
      · 2000: growth_1999 existe pero growth_lag1_2000 = growth_1999 es NaN
              porque growth_1999 ya era NaN (no hay 1998).
    En la práctica, 1999 ya era NaN antes; ahora también lo es 2000 (24 filas
    eliminadas en preparar_X_y() en lugar de 12 — impacto negligible).

    Devuelve:
        pd.Series alineada con df.index.
    """
    serie_anual      = df.groupby(df.index.year)[col].first()
    crecimiento      = serie_anual.pct_change() * 100     # growth_t = (PIB_t - PIB_{t-1}) / PIB_{t-1}
    crecimiento_lag1 = crecimiento.shift(1)               # año t recibe growth_{t-1} — sin look-ahead
    resultado        = df.index.year.map(crecimiento_lag1.to_dict())
    return pd.Series(resultado, index=df.index, name=f"{col}_growth")


def _calcular_OIS_taylor(df: pd.DataFrame, banco: str) -> pd.Series:
    """
    Proxy del tipo de interes neutral mediante la Regla de Taylor simplificada.

    Formula: OIS_taylor_t = max(0, r* + 1.5*(pi_t - pi*) + 0.5*(PIB_growth_t - y*))

    Donde (config.TAYLOR_PARAMS[banco]):
      r*         : tipo neutral real  (BCE=1.0%, BoE=1.5%, FED=2.5%)
      pi*        : objetivo inflacion (2.0% para los tres bancos)
      y*         : crecimiento potencial (BCE=1.5%, BoE=2.0%, FED=2.0%)

    Ventajas sobre OIS bruto:
      1. Sin circularidad: OIS_taylor se deriva de inflacion y PIB (exogenos),
         no del propio tipo oficial ni de su proxy de mercado.
      2. El modelo responde a fundamentos macroeconomicos: subir inflacion en
         un escenario -> OIS_taylor sube -> prediccion de tipos sube.
      3. Calibracion economicamente motivada (Holston-Laubach-Williams 2023,
         Laubach-Williams 2023).

    Requiere columnas "Inflacion" y "PIB_growth" en df.
    Los NaN en PIB_growth (primer ano de datos) se rellenan con y* para que
    el termino de output gap sea neutro en ese periodo.
    """
    p = config.TAYLOR_PARAMS[banco]
    pib_growth = df["PIB_growth"].fillna(p["y_star"])   # NaN primer ano -> neutro
    taylor = (
        p["r_star"]
        + 1.5 * (df["Inflacion"] - p["pi_star"])
        + 0.5 * (pib_growth      - p["y_star"])
    )
    return taylor.clip(lower=0.0).rename("OIS_taylor")


def preparar_features(df: pd.DataFrame, banco: str = "BCE", target_col: str = "tipo_oficial") -> pd.DataFrame:
    """
    Construye la matriz de variables explicativas X (10 columnas).

    Transformaciones aplicadas:
      · PIB       → PIB_growth    (% crecimiento interanual, forward-filled por año)
      · PIBpc     → PIBpc_growth  (ídem)
      · OIS_taylor: Regla de Taylor derivada de Inflacion y PIB_growth.
                    Sustituye al OIS bruto eliminando la circularidad.
      · brecha_taylor = OIS_taylor - target_col (pp).
                    Gap entre el tipo neutral de Taylor y el tipo observado.
                    Negativa → tipos por encima del equilibrio (presión bajista).
                    Positiva → tipos por debajo del equilibrio (presión alcista).
      · Resto (Inflacion, Desempleo, VIX, Oro, Plata, Gas): uso directo.
      · Rating eliminado: variable estática, importancia=0% en el RF.

    No se aplica escalado aquí. El StandardScaler de clasificación se ajusta
    exclusivamente sobre X_train dentro de classification.py.

    Parámetros:
        df:         DataFrame mensual devuelto por data_loader.
        banco:      "BCE", "BoE" o "FED" (para los parametros de Taylor).
        target_col: columna objetivo del modelo — "tipo_oficial" (por defecto)
                    o "OIS" para modelar tipos OIS directamente.

    Devuelve:
        DataFrame con exactamente las 10 columnas de config.VARIABLES_MODELO:
        ["PIB_growth", "PIBpc_growth", "Inflacion", "Desempleo",
         "VIX", "Oro", "Plata", "Gas", "OIS_taylor", "brecha_taylor"]
        (Rating excluido — importancia RF = 0%, variable estática sin poder discriminativo)
    """
    df = df.copy()

    df["PIB_growth"]     = _calcular_crecimiento_anual(df, "PIB")
    df["PIBpc_growth"]   = _calcular_crecimiento_anual(df, "PIBpc")
    df["OIS_taylor"]     = _calcular_OIS_taylor(df, banco)

    # brecha_taylor: distancia entre el tipo neutral (Taylor) y el tipo objetivo
    # Positiva → tipos por debajo del equilibrio (banco debería subir)
    # Negativa → tipos por encima del equilibrio (banco debería bajar)
    df["brecha_taylor"]  = df["OIS_taylor"] - df[target_col]

    # delta_tipo_lag1: cambio del tipo objetivo el mes anterior (momentum)
    # Captura la dirección reciente de política monetaria.
    # Negativo → ciclo de bajadas en curso; positivo → ciclo de subidas.
    df["delta_tipo_lag1"] = df[target_col].diff().fillna(0.0)

    # meses_sin_cambio: contador de meses consecutivos sin mover el tipo (inercia)
    # Cuando el banco lleva varios meses parado, la probabilidad de seguir parado
    # es alta. Esta variable permite al RF aprender esa inercia de política.
    _delta_abs = df["delta_tipo_lag1"].abs()
    _conteo = []
    _c = 0
    for _d in _delta_abs:
        if _d < 0.001:
            _c += 1
        else:
            _c = 0
        _conteo.append(_c)
    df["meses_sin_cambio"] = _conteo

    # Seleccionar exactamente las 12 columnas del modelo (Rating excluido)
    return df[config.VARIABLES_MODELO].copy()


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def preparar_X_y(
    df: pd.DataFrame,
    banco: str,
    target_col: str = "tipo_oficial",
    umbral: float = config.THRESHOLD_ESTABILIDAD,
) -> tuple:
    """
    Pipeline completo: DataFrame mensual -> X, y_reg_delta, y_cls, fechas, tipo_actual.

    Pasos:
      1. Calcular features (PIB_growth, PIBpc_growth + 8 directas).
      2. Crear y_regression = delta = target_{t+1} - target_t.
      3. Crear y_classification directamente sobre el delta.
      4. Extraer tipo_actual = target_col (para reconstruccion de nivel downstream).
      5. Eliminar filas donde y_regression o alguna feature sea NaN.

    Parametros:
        df:         DataFrame de cargar_datos_procesados(banco).
        banco:      "BCE", "BoE" o "FED" (para logging y Taylor params).
        target_col: columna objetivo — "tipo_oficial" (default) o "OIS".
        umbral:     umbral de clasificacion en pp (default 0.125).

    Devuelve:
        (X, y_reg, y_cls, fechas, tipo_actual)
        - X:           DataFrame (n x 9), columnas = config.VARIABLES_MODELO.
        - y_reg:       Series float64, DELTA del tipo (tipo_{t+1} - tipo_t, en pp).
        - y_cls:       Series object,  "Sube" / "Baja" / "Estable".
        - fechas:      DatetimeIndex correspondiente a las filas limpias.
        - tipo_actual: Series float64, nivel del mes t (para reconstruir nivel).
    """
    # 1. Features (banco necesario para calcular OIS_taylor con parametros correctos)
    X = preparar_features(df, banco, target_col=target_col)

    # 2. Target de regresion: delta = target_{t+1} - target_t
    y_reg = crear_y_regresion(df, target_col=target_col)

    # 3. Clasificacion directa sobre el delta
    diff = y_reg  # y_reg ya ES el delta
    y_cls = pd.Series(np.nan, index=diff.index, dtype=object)
    y_cls[diff >  umbral] = "Sube"
    y_cls[diff < -umbral] = "Baja"
    mask_estable = (diff >= -umbral) & (diff <= umbral)
    y_cls[mask_estable] = "Estable"
    y_cls.name = "y_classification"

    # 4. tipo_actual necesario para reconstruccion de nivel downstream
    tipo_actual = df[target_col].copy()
    tipo_actual.name = "tipo_actual"

    # 5. Combinar y limpiar filas con NaN
    combined = pd.concat([X, y_reg, y_cls, tipo_actual], axis=1)

    cols_features = config.VARIABLES_MODELO
    mask_ok = (
        combined["y_regression"].notna()
        & combined[cols_features].notna().all(axis=1)
    )
    combined = combined.loc[mask_ok]

    X_limpio          = combined[cols_features]
    y_reg_limpio      = combined["y_regression"].astype(float)
    y_cls_limpio      = combined["y_classification"].astype(str)
    fechas            = combined.index
    tipo_actual_limpio = combined["tipo_actual"].astype(float)

    return X_limpio, y_reg_limpio, y_cls_limpio, fechas, tipo_actual_limpio


# ---------------------------------------------------------------------------
# División temporal train/test (Fase 4)
# ---------------------------------------------------------------------------

def dividir_temporal(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    fechas: pd.DatetimeIndex,
    corte: str = config.TEST_START,
) -> tuple:
    """
    División cronológica train/test. NUNCA aleatoria (no shuffle).

    Corte: config.TEST_START = "2024-01-01"
      · Train: desde el primer dato disponible hasta 2023-12-31.
      · Test:  desde 2024-01-01 hasta el último dato disponible (2025-12-01).

    Verificación de no-fuga temporal:
      · El corte se realiza por fecha, no por posición ni porcentaje.
      · Test siempre cronológicamente posterior al train.
      · Los datos de test no se usan en ningún paso de entrenamiento.

    Parámetros:
        X, y_reg, y_cls, fechas: salida de preparar_X_y().
        corte: fecha de inicio del test (string "YYYY-MM-DD").

    Devuelve:
        (X_train, X_test,
         y_train_reg, y_test_reg,
         y_train_cls, y_test_cls,
         fechas_train, fechas_test)
    """
    corte_ts = pd.Timestamp(corte)

    mask_train = fechas < corte_ts
    mask_test  = fechas >= corte_ts

    return (
        X.loc[mask_train],   X.loc[mask_test],
        y_reg.loc[mask_train], y_reg.loc[mask_test],
        y_cls.loc[mask_train], y_cls.loc[mask_test],
        fechas[mask_train],    fechas[mask_test],
    )
