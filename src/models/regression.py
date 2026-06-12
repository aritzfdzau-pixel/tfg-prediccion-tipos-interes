"""
regression.py — Modelo de regresion (Fase 5)

Predice el CAMBIO (delta) del tipo de interes del mes siguiente internamente,
y reporta metricas sobre el NIVEL RECONSTRUIDO (tipo_actual + delta_pred)
para que sean directamente interpretables.

Implementa dos modelos:
  - Random Forest Regressor  -> modelo PRINCIPAL (config.REGRESSOR = "random_forest")
  - Ridge CV                 -> modelo COMPARATIVO (baseline lineal escalado)

Funciones publicas:
  entrenar_regresion(X_train, y_train, tipo)   -> dict modelo
  predecir_regresion(modelo_dict, X)           -> np.ndarray (deltas)
  reconstruir_nivel(deltas, tipo_actual)       -> np.ndarray (niveles)
  evaluar_regresion(modelo_dict, X, y)         -> dict metricas (sobre deltas)
  importancia_features(modelo_dict, banco)     -> pd.Series
  guardar_modelo(modelo_dict, banco)           -> Path
  cargar_modelo(banco)                         -> dict modelo
  ejecutar_pipeline_banco(banco)               -> dict resultados completos

FUGA TEMPORAL — garantias:
  - X_train / y_train: filas con fecha < 2024-01-01 (288 observaciones).
  - X_test  / y_test:  filas con fecha >= 2024-01-01 (24 observaciones).
  - El StandardScaler de Ridge se ajusta SOLO sobre X_train.
  - Las metricas de evaluacion se calculan SOLO sobre X_test / y_test.
  - No hay ningun parametro entrenado sobre datos de test.
  - reconstruir_nivel() usa tipo_actual del MISMO periodo (no futuro).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.data.feature_engineering import preparar_X_y, dividir_temporal


def _suf(instrumento: str) -> str:
    """Sufijo de archivo: '' para tipo_oficial, '_OIS' para OIS."""
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar_regresion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tipo: str = config.REGRESSOR,
) -> dict:
    """
    Entrena el modelo de regresión sobre X_train, y_train.

    tipo = "random_forest"
        RandomForestRegressor con hiperparámetros de config.py.
        No requiere escalado (modelo basado en árboles).

    tipo = "ridge"
        RidgeCV con StandardScaler ajustado solo sobre X_train.
        Modelo lineal regularizado; útil como baseline comparativo.

    Parámetros:
        X_train: DataFrame (n_train × 9) con las features del modelo.
        y_train: Series float64 con el tipo de interés del mes siguiente (%).
        tipo:    "random_forest" (default) o "ridge".

    Devuelve:
        dict con claves:
            "modelo":  objeto sklearn ajustado
            "scaler":  StandardScaler ajustado (solo Ridge) o None
            "tipo":    string identificador ("random_forest" / "ridge")
    """
    if tipo == "random_forest":
        modelo = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,        # 200
            max_depth=config.RF_MAX_DEPTH,              # 6
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, # 10
            max_features=config.RF_MAX_FEATURES,         # 'sqrt' ≈ 3 features/split
            random_state=config.RF_RANDOM_STATE,         # 42
            n_jobs=-1,
        )
        modelo.fit(X_train, y_train)
        return {"modelo": modelo, "scaler": None, "tipo": tipo}

    elif tipo == "ridge":
        # Escalar SOLO sobre train — el scaler se reutiliza en predecir_regresion()
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        # RidgeCV selecciona alpha por validación cruzada interna (CV=5)
        modelo = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            cv=5,
        )
        modelo.fit(X_train_sc, y_train)
        return {"modelo": modelo, "scaler": scaler, "tipo": tipo}

    else:
        raise ValueError(
            f"Tipo de regresor no reconocido: '{tipo}'. "
            "Valores validos: 'random_forest', 'ridge'."
        )


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------

def predecir_regresion(modelo_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Genera predicciones numericas del tipo de interes.

    Aplica el StandardScaler guardado en modelo_dict si el modelo es Ridge.
    Para Random Forest, no se aplica escalado.

    Parámetros:
        modelo_dict: dict devuelto por entrenar_regresion().
        X:           DataFrame con las mismas columnas que X_train.

    Devuelve:
        np.ndarray de floats (valores en %, mismo largo que X).
    """
    modelo = modelo_dict["modelo"]
    scaler = modelo_dict["scaler"]

    if scaler is not None:
        X_sc = scaler.transform(X)
        return modelo.predict(X_sc)
    else:
        return modelo.predict(X)


# ---------------------------------------------------------------------------
# Reconstruccion de nivel desde deltas
# ---------------------------------------------------------------------------

def reconstruir_nivel(deltas: np.ndarray, tipo_actual) -> np.ndarray:
    """
    Reconstruye el nivel del tipo de interes desde predicciones de delta.

    Modo paralelo (test historico): tipo_predicho_t = tipo_actual_t + delta_pred_t
        tipo_actual es pd.Series o np.ndarray con el tipo real de cada mes.

    Modo encadenado (escenarios futuros): tipo_{i+1} = tipo_i + delta_i
        tipo_actual es un float escalar (tipo de partida, ej. dic-2025).

    En ambos casos, el resultado se recorta a [0.0, 25.0] pp para evitar
    valores negativos o absurdos en horizontes largos.

    Parametros:
        deltas:      array (n,) con los cambios predichos (pp).
        tipo_actual: float escalar (modo encadenado) o array/Series (modo paralelo).

    Devuelve:
        np.ndarray (n,) con niveles reconstruidos, recortados a [0, 25] pp.
    """
    deltas = np.asarray(deltas, dtype=float)
    if isinstance(tipo_actual, (int, float, np.floating)):
        # Modo encadenado: para escenarios futuros 2026-2030
        niveles = np.zeros(len(deltas))
        niveles[0] = tipo_actual + deltas[0]
        for i in range(1, len(deltas)):
            niveles[i] = niveles[i - 1] + deltas[i]
    else:
        # Modo paralelo: para test historico
        tipo_arr = np.asarray(tipo_actual, dtype=float)
        niveles = tipo_arr + deltas
    return np.round(np.clip(niveles, 0.0, 25.0), 4)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------

def evaluar_regresion(
    modelo_dict: dict,
    X: pd.DataFrame,
    y: pd.Series,
    conjunto: str = "test",
) -> dict:
    """
    Calcula MAE, RMSE y R² sobre el conjunto indicado.

    IMPORTANTE: En produccion siempre llamar con X_test / y_test para evitar
    reportar metricas infladas de entrenamiento.
    Se puede llamar con X_train / y_train para detectar sobreajuste.

    Parámetros:
        modelo_dict: dict devuelto por entrenar_regresion().
        X:           DataFrame de features.
        y:           Series con los valores reales del tipo de interes (%).
        conjunto:    etiqueta para identificar en el resumen ("train"/"test").

    Devuelve:
        dict con claves: "MAE", "RMSE", "R2", "conjunto".
        Todas las metricas redondeadas a 4 decimales.
    """
    y_pred = predecir_regresion(modelo_dict, X)
    y_arr  = np.array(y)

    mae  = mean_absolute_error(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    r2   = r2_score(y_arr, y_pred)

    return {
        "MAE":      round(mae,  4),
        "RMSE":     round(rmse, 4),
        "R2":       round(r2,   4),
        "conjunto": conjunto,
    }


# ---------------------------------------------------------------------------
# Importancia de features (solo Random Forest)
# ---------------------------------------------------------------------------

def importancia_features(modelo_dict: dict, banco: str, instrumento: str = "tipo_oficial") -> pd.Series:
    """
    Extrae las importancias de features del Random Forest y las guarda en CSV.

    Solo disponible para tipo = "random_forest".
    Guarda en: Resultados/importancia_{banco}[_OIS].csv

    Devuelve:
        pd.Series con indice = nombres de features, ordenada de mayor a menor.
    """
    if modelo_dict["tipo"] != "random_forest":
        raise ValueError("importancia_features solo esta disponible para tipo='random_forest'.")

    modelo = modelo_dict["modelo"]
    importancias = pd.Series(
        modelo.feature_importances_,
        index=config.VARIABLES_MODELO,
        name="importancia",
    ).sort_values(ascending=False)

    # Guardar CSV
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTS_DIR / f"importancia_{banco}{_suf(instrumento)}.csv"
    importancias.to_csv(ruta)

    return importancias


# ---------------------------------------------------------------------------
# Persistencia del modelo
# ---------------------------------------------------------------------------

def guardar_modelo(modelo_dict: dict, banco: str, instrumento: str = "tipo_oficial") -> Path:
    """
    Serializa el modelo (dict completo) con joblib.

    Ruta: Modelo/resultados/{banco}[_OIS]_regresion.joblib
    Incluye el scaler si el modelo es Ridge.

    Devuelve:
        Path del archivo guardado.
    """
    ruta_dir = config.MODEL_DIR / "resultados"
    ruta_dir.mkdir(parents=True, exist_ok=True)

    ruta = ruta_dir / f"{banco}{_suf(instrumento)}_regresion.joblib"
    joblib.dump(modelo_dict, ruta)
    return ruta


def cargar_modelo(banco: str, instrumento: str = "tipo_oficial") -> dict:
    """
    Carga el modelo de DELTA serializado.
    Ruta: Modelo/resultados/{banco}[_OIS]_regresion.joblib

    Devuelve:
        dict con "modelo", "scaler", "tipo" — listo para predecir_regresion().

    Lanza:
        FileNotFoundError si el archivo no existe (ejecutar entrenamiento primero).
    """
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{_suf(instrumento)}_regresion.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {ruta}\n"
            "Ejecuta primero: python src/models/regression.py"
        )
    return joblib.load(ruta)


def guardar_modelo_nivel(modelo_dict: dict, banco: str, instrumento: str = "tipo_oficial") -> Path:
    """
    Serializa el modelo de NIVEL (predice target_{t+1} directamente).
    Ruta: Modelo/resultados/{banco}[_OIS]_regresion_nivel.joblib
    """
    ruta_dir = config.MODEL_DIR / "resultados"
    ruta_dir.mkdir(parents=True, exist_ok=True)
    ruta = ruta_dir / f"{banco}{_suf(instrumento)}_regresion_nivel.joblib"
    joblib.dump(modelo_dict, ruta)
    return ruta


def cargar_modelo_nivel(banco: str, instrumento: str = "tipo_oficial") -> dict:
    """
    Carga el modelo de NIVEL serializado.
    Ruta: Modelo/resultados/{banco}[_OIS]_regresion_nivel.joblib

    Devuelve:
        dict con "modelo", "scaler", "tipo" — listo para predecir_regresion().
    """
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{_suf(instrumento)}_regresion_nivel.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo de nivel no encontrado: {ruta}\n"
            "Ejecuta primero: python src/models/regression.py"
        )
    return joblib.load(ruta)


# ---------------------------------------------------------------------------
# Pipeline completo por banco
# ---------------------------------------------------------------------------

def ejecutar_pipeline_banco(banco: str, instrumento: str = "tipo_oficial") -> dict:
    """
    Ejecuta el pipeline completo de regresion para un banco central.

    El modelo predice DELTAS internamente (target_{t+1} - target_t).
    Las metricas y el CSV de predicciones se reportan sobre NIVELES RECONSTRUIDOS
    (tipo_actual + delta_pred) para que sean directamente interpretables.

    Parámetros:
        banco:       "BCE", "BoE" o "FED".
        instrumento: "tipo_oficial" (default) o "OIS".

    Pasos:
      1. Carga features procesadas de Datos procesados/dataset_{banco}.csv
      2. Prepara X, y_reg (deltas), y_cls, fechas, tipo_actual (5-tuple)
      3. Division temporal train/test
      4. Entrena Random Forest (modelo principal) y Ridge CV (comparativo)
      5. Reconstruye niveles para metricas interpretables
      6. Extrae importancia de features (RF)
      7. Guarda modelos en Modelo/resultados/{banco}[_OIS]_regresion*.joblib
      8. Guarda predicciones en Resultados/predicciones_test_regresion_{banco}[_OIS].csv

    Devuelve:
        dict con todos los modelos, metricas (sobre nivel), predicciones y datos de split.
    """
    from src.data.data_loader import cargar_datos_procesados

    print(f"\n{'='*62}")
    print(f"  REGRESION — {banco} [{instrumento}]")
    print(f"{'='*62}")

    # 1. Cargar datos
    target_col = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    df = cargar_datos_procesados(banco)
    X, y_reg, y_cls, fechas, tipo_actual = preparar_X_y(df, banco, target_col=target_col)  # 5-tuple

    # 2. Division temporal
    (X_train, X_test,
     y_train_reg, y_test_reg,
     y_train_cls, y_test_cls,
     fechas_train, fechas_test) = dividir_temporal(X, y_reg, y_cls, fechas)

    # Separar tipo_actual en train/test (para reconstruccion de nivel)
    tipo_actual_train = tipo_actual.loc[fechas_train]
    tipo_actual_test  = tipo_actual.loc[fechas_test]

    print(f"  Train: {len(X_train)} obs  "
          f"({fechas_train[0].strftime('%Y-%m')} a {fechas_train[-1].strftime('%Y-%m')})")
    print(f"  Test:  {len(X_test)} obs  "
          f"({fechas_test[0].strftime('%Y-%m')} a {fechas_test[-1].strftime('%Y-%m')})")
    print(f"  Target: delta (tipo_{{t+1}} - tipo_t); metricas sobre nivel reconstruido")

    # Niveles reales (para calcular metricas sobre el nivel)
    nivel_real_train = tipo_actual_train.values + y_train_reg.values
    nivel_real_test  = tipo_actual_test.values  + y_test_reg.values

    # 3. Random Forest (entrena sobre deltas)
    print(f"\n  [Random Forest]  "
          f"n_estimators={config.RF_N_ESTIMATORS}  "
          f"max_depth={config.RF_MAX_DEPTH}  "
          f"min_samples_leaf={config.RF_MIN_SAMPLES_LEAF}")

    modelo_rf = entrenar_regresion(X_train, y_train_reg, tipo="random_forest")

    # Predicciones de delta -> reconstruccion de nivel
    delta_pred_rf_train = predecir_regresion(modelo_rf, X_train)
    delta_pred_rf_test  = predecir_regresion(modelo_rf, X_test)
    nivel_pred_rf_train = reconstruir_nivel(delta_pred_rf_train, tipo_actual_train.values)
    nivel_pred_rf_test  = reconstruir_nivel(delta_pred_rf_test,  tipo_actual_test.values)

    # Metricas sobre nivel reconstruido
    met_train_rf = {
        "MAE":      round(mean_absolute_error(nivel_real_train, nivel_pred_rf_train), 4),
        "RMSE":     round(np.sqrt(mean_squared_error(nivel_real_train, nivel_pred_rf_train)), 4),
        "R2":       round(r2_score(nivel_real_train, nivel_pred_rf_train), 4),
        "conjunto": "train",
    }
    met_test_rf = {
        "MAE":      round(mean_absolute_error(nivel_real_test, nivel_pred_rf_test), 4),
        "RMSE":     round(np.sqrt(mean_squared_error(nivel_real_test, nivel_pred_rf_test)), 4),
        "R2":       round(r2_score(nivel_real_test, nivel_pred_rf_test), 4),
        "conjunto": "test",
    }

    print(f"  Train -> MAE={met_train_rf['MAE']:.4f} pp | "
          f"RMSE={met_train_rf['RMSE']:.4f} pp | R2={met_train_rf['R2']:.4f}")
    print(f"  Test  -> MAE={met_test_rf['MAE']:.4f} pp | "
          f"RMSE={met_test_rf['RMSE']:.4f} pp | R2={met_test_rf['R2']:.4f}")

    # Importancia de features
    importancias = importancia_features(modelo_rf, banco, instrumento=instrumento)
    top3 = ", ".join(importancias.head(3).index.tolist())
    print(f"  Top-3 features: {top3}")

    # 4. Ridge (entrena sobre deltas)
    print(f"\n  [Ridge CV]  alphas=[0.01, 0.1, 1, 10, 100, 1000]  cv=5")

    modelo_ridge = entrenar_regresion(X_train, y_train_reg, tipo="ridge")
    alpha_opt    = modelo_ridge["modelo"].alpha_

    delta_pred_ridge_train = predecir_regresion(modelo_ridge, X_train)
    delta_pred_ridge_test  = predecir_regresion(modelo_ridge, X_test)
    nivel_pred_ridge_train = reconstruir_nivel(delta_pred_ridge_train, tipo_actual_train.values)
    nivel_pred_ridge_test  = reconstruir_nivel(delta_pred_ridge_test,  tipo_actual_test.values)

    met_train_ridge = {
        "MAE":      round(mean_absolute_error(nivel_real_train, nivel_pred_ridge_train), 4),
        "RMSE":     round(np.sqrt(mean_squared_error(nivel_real_train, nivel_pred_ridge_train)), 4),
        "R2":       round(r2_score(nivel_real_train, nivel_pred_ridge_train), 4),
        "conjunto": "train",
    }
    met_test_ridge = {
        "MAE":      round(mean_absolute_error(nivel_real_test, nivel_pred_ridge_test), 4),
        "RMSE":     round(np.sqrt(mean_squared_error(nivel_real_test, nivel_pred_ridge_test)), 4),
        "R2":       round(r2_score(nivel_real_test, nivel_pred_ridge_test), 4),
        "conjunto": "test",
    }

    print(f"  Alpha optimo: {alpha_opt}")
    print(f"  Train -> MAE={met_train_ridge['MAE']:.4f} pp | "
          f"RMSE={met_train_ridge['RMSE']:.4f} pp | R2={met_train_ridge['R2']:.4f}")
    print(f"  Test  -> MAE={met_test_ridge['MAE']:.4f} pp | "
          f"RMSE={met_test_ridge['RMSE']:.4f} pp | R2={met_test_ridge['R2']:.4f}")

    # 5a. Entrenar modelo de NIVEL (para predicciones futuras 2026-2030)
    # y_nivel = tipo_actual + delta = target_{t+1}
    # Se usa en prediccion_futura.py para evitar la acumulacion de errores en
    # 60 meses encadenados (iterated one-step forecasting seria inestable).
    y_nivel_train = pd.Series(nivel_real_train, index=fechas_train, name="y_nivel")
    modelo_rf_nivel = entrenar_regresion(X_train, y_nivel_train, tipo="random_forest")
    ruta_nivel = guardar_modelo_nivel(modelo_rf_nivel, banco, instrumento=instrumento)
    print(f"\n  Modelo RF NIVEL guardado: {ruta_nivel.name}  (para escenarios 2026-2030)")

    # 5b. Guardar modelo de DELTA (para scoring del test historico)
    ruta_modelo = guardar_modelo(modelo_rf, banco, instrumento=instrumento)
    print(f"  Modelo RF DELTA guardado: {ruta_modelo.name}  (para test 2024-2025)")

    # 6. Guardar predicciones del test con niveles reconstruidos y deltas
    df_pred = pd.DataFrame({
        "fecha":           fechas_test,
        "y_real":          nivel_real_test,        # nivel real del siguiente mes
        "y_pred_rf":       nivel_pred_rf_test,     # nivel predicho RF
        "y_pred_ridge":    nivel_pred_ridge_test,  # nivel predicho Ridge
        "delta_real":      y_test_reg.values,      # cambio real (pp)
        "delta_pred_rf":   delta_pred_rf_test,     # cambio predicho RF (pp)
        "delta_pred_ridge": delta_pred_ridge_test, # cambio predicho Ridge (pp)
        "error_nivel_rf":  nivel_real_test - nivel_pred_rf_test,
        "error_nivel_ridge": nivel_real_test - nivel_pred_ridge_test,
    }).set_index("fecha")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_pred = config.RESULTS_DIR / f"predicciones_test_regresion_{banco}{_suf(instrumento)}.csv"
    df_pred.to_csv(ruta_pred)
    print(f"  Predicciones test guardadas: {ruta_pred.name}")

    return {
        "banco":             banco,
        "instrumento":       instrumento,
        "modelo_rf":         modelo_rf,
        "modelo_ridge":      modelo_ridge,
        "met_train_rf":      met_train_rf,
        "met_test_rf":       met_test_rf,
        "met_train_ridge":   met_train_ridge,
        "met_test_ridge":    met_test_ridge,
        "importancias":      importancias,
        "X_train":           X_train,
        "X_test":            X_test,
        "y_train_reg":       y_train_reg,     # deltas
        "y_test_reg":        y_test_reg,      # deltas
        "y_train_cls":       y_train_cls,
        "y_test_cls":        y_test_cls,
        "fechas_train":      fechas_train,
        "fechas_test":       fechas_test,
        "tipo_actual_train": tipo_actual_train,
        "tipo_actual_test":  tipo_actual_test,
        "predicciones_test": df_pred,
    }


# ---------------------------------------------------------------------------
# Resumen consolidado de los 3 bancos
# ---------------------------------------------------------------------------

def _calcular_mase(mae_test: float, y_train_reg: pd.Series) -> float:
    """
    MASE — Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Escala el MAE del modelo por el MAE del forecast naïve random walk calculado
    dentro del periodo de entrenamiento. El denominador se obtiene de los deltas
    de entrenamiento (ya disponibles como y_train_reg), sin look-ahead.

        MASE = MAE_test / MAE_naive_insample

        MAE_naive_insample = mean(|delta_t|) para t en train
                           = mean(|tipo_{t+1} - tipo_t|) en el periodo de entrenamiento

    Interpretación:
        MASE < 1  → el modelo bate al random walk  (deseable)
        MASE = 1  → equivalente al random walk
        MASE > 1  → peor que el random walk

    Ventaja académica: no está inflado por autocorrelación de niveles (a diferencia
    del R² sobre niveles) y es comparable entre series con escalas distintas.
    Referencia estándar para benchmarking de modelos de series temporales.

    Parámetros:
        mae_test:    MAE del modelo sobre el test (pp, sobre nivel reconstruido).
        y_train_reg: Series de deltas del entrenamiento (tipo_{t+1} - tipo_t, pp).

    Devuelve:
        float redondeado a 4 decimales, o NaN si el denominador es degenerado.
    """
    mae_naive = float(np.mean(np.abs(y_train_reg.values)))
    if mae_naive < 1e-10:
        return float("nan")
    return round(mae_test / mae_naive, 4)


def actualizar_metricas_mase(instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Calcula el MASE para cada banco (RF y Ridge) y añade la columna al CSV existente.

    No reentrena los modelos. Lee el resumen_regresion[_OIS].csv ya generado,
    recalcula y_train_reg en memoria (operación rápida, solo feature_engineering),
    y sobreescribe el CSV con la columna MASE añadida.

    Parámetros:
        instrumento: "tipo_oficial" o nombre OIS ("ESTR"/"SONIA"/"SOFR").

    Devuelve:
        pd.DataFrame actualizado con la columna MASE.
    """
    from src.data.data_loader import cargar_datos_procesados
    from src.data.feature_engineering import preparar_X_y

    target_col  = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    ruta_resumen = config.RESULTS_DIR / f"resumen_regresion{_suf(instrumento)}.csv"

    if not ruta_resumen.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_resumen.name}. "
            "Ejecuta regression.py o run_ois_model.py primero."
        )

    df_resumen = pd.read_csv(ruta_resumen)

    print(f"\n  Calculando MASE para instrumento='{instrumento}'...")

    mase_map: dict[tuple, float] = {}

    for banco in config.BANCOS_CENTRALES:
        df   = cargar_datos_procesados(banco)
        X, y_reg, _, fechas, _ = preparar_X_y(df, banco, target_col=target_col)

        mask_train  = fechas < pd.Timestamp(config.TEST_START)
        y_train_reg = y_reg.loc[mask_train]

        mae_naive = float(np.mean(np.abs(y_train_reg.values)))

        for modelo_nombre in ["RF", "Ridge"]:
            row = df_resumen[
                (df_resumen["Banco"] == banco) &
                (df_resumen["Modelo"] == modelo_nombre)
            ]
            if not row.empty:
                mae_test  = float(row.iloc[0]["MAE_test"])
                mase_val  = round(mae_test / mae_naive, 4) if mae_naive > 1e-10 else float("nan")
                mase_map[(banco, modelo_nombre)] = mase_val
                print(f"    {banco} [{modelo_nombre}]: "
                      f"MAE_test={mae_test:.4f}  MAE_naive={mae_naive:.4f}  "
                      f"MASE={mase_val:.4f}"
                      + (" [bate al naive]" if mase_val < 1 else ""))

    df_resumen["MASE"] = df_resumen.apply(
        lambda r: mase_map.get((r["Banco"], r["Modelo"]), float("nan")),
        axis=1,
    )

    df_resumen.to_csv(ruta_resumen, index=False)
    print(f"  Guardado: {ruta_resumen.name}  (columna MASE añadida)")
    return df_resumen


def guardar_resumen_metricas(todos_resultados: list, instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Consolida las metricas de regresion de los 3 bancos en un unico CSV.

    Guarda: Resultados/resumen_regresion[_OIS].csv
    Devuelve: DataFrame con las metricas de todos los bancos y modelos.
    """
    filas = []
    for res in todos_resultados:
        banco = res["banco"]
        for tipo, mt, mc in [
            ("RF",    "met_train_rf",    "met_test_rf"),
            ("Ridge", "met_train_ridge", "met_test_ridge"),
        ]:
            filas.append({
                "Banco":      banco,
                "Modelo":     tipo,
                "MAE_train":  res[mt]["MAE"],
                "RMSE_train": res[mt]["RMSE"],
                "R2_train":   res[mt]["R2"],
                "MAE_test":   res[mc]["MAE"],
                "RMSE_test":  res[mc]["RMSE"],
                "R2_test":    res[mc]["R2"],
            })

    df_resumen = pd.DataFrame(filas)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTS_DIR / f"resumen_regresion{_suf(instrumento)}.csv"
    df_resumen.to_csv(ruta, index=False)
    print(f"\n  Resumen guardado: {ruta.name}")
    return df_resumen


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    todos = []
    for banco in config.BANCOS_CENTRALES:
        res = ejecutar_pipeline_banco(banco)
        todos.append(res)

    df_resumen = guardar_resumen_metricas(todos)

    print("\n" + "="*62)
    print("  RESUMEN FINAL — METRICAS TEST (Random Forest)")
    print("="*62)
    tabla_rf = df_resumen[df_resumen["Modelo"] == "RF"][
        ["Banco", "MAE_test", "RMSE_test", "R2_test"]
    ].reset_index(drop=True)
    print(tabla_rf.to_string(index=False))

    print("\n  RESUMEN FINAL — METRICAS TEST (Ridge CV)")
    print("="*62)
    tabla_ridge = df_resumen[df_resumen["Modelo"] == "Ridge"][
        ["Banco", "MAE_test", "RMSE_test", "R2_test"]
    ].reset_index(drop=True)
    print(tabla_ridge.to_string(index=False))
