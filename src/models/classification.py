"""
classification.py — Modelo de clasificación multiclase (Fase 6)

Predice la dirección del movimiento del tipo de interés en el mes siguiente:
    "Sube"   →  diff > +0.125 pp
    "Estable"→  |diff| <= 0.125 pp
    "Baja"   →  diff < -0.125 pp

Modelo: Logistic Regression con solver lbfgs (multinomial implícito para
        3 clases) y class_weight='balanced' (Decisión 7 — desequilibrio ~83%).

El StandardScaler se ajusta EXCLUSIVAMENTE sobre X_train para evitar
fuga de información del conjunto de test (Decisión Anti-Fuga).

Funciones públicas:
  entrenar_clasificacion(X_train, y_train)        → (modelo_cls, scaler)
  predecir_clasificacion(modelo_cls, scaler, X)   → np.ndarray de strings
  predecir_probabilidades(modelo_cls, scaler, X)  → np.ndarray (n × 3)
  evaluar_clasificacion(modelo_cls, scaler, X, y) → dict de métricas
  matriz_confusion(modelo_cls, scaler, X, y)      → pd.DataFrame
  guardar_modelo(modelo_cls, scaler, banco)        → Path
  cargar_modelo(banco)                             → (modelo_cls, scaler)
  ejecutar_pipeline_banco(banco)                   → dict resultados

FUGA TEMPORAL — garantías:
  · StandardScaler ajustado solo con X_train (288 observaciones).
  · Métricas de evaluación calculadas solo sobre X_test (24 observaciones).
  · División cronológica: train < 2024-01-01, test >= 2024-01-01.
  · No hay shuffle ni acceso a datos futuros en ningún paso.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
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

def entrenar_clasificacion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple:
    """
    Entrena el modelo de clasificación multinomial sobre X_train, y_train.

    Modelo: LogisticRegression con solver='lbfgs'.
      · Con 3 clases y solver lbfgs, sklearn ajusta un modelo multinomial
        (softmax regression), equivalente a multi_class='multinomial'.
      · class_weight='balanced': compensa el desequilibrio de clases
        (~83% Estable, ~10% Sube, ~7% Baja) asignando mayor penalización
        a los errores en las clases minoritarias.

    StandardScaler:
      · Ajustado SOLO sobre X_train. El mismo objeto se usa en test y futuro.
      · La regresión logística es sensible a la escala de las features;
        el escalado es obligatorio para que lbfgs converja correctamente.

    Parámetros:
        X_train: DataFrame (n_train × 9) con las features del modelo.
        y_train: Series de strings ("Sube", "Baja", "Estable").

    Devuelve:
        (modelo_cls, scaler)
        · modelo_cls: LogisticRegression ajustado.
        · scaler:     StandardScaler ajustado sobre X_train.
    """
    # Escalado — ajuste SOLO sobre train
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # Modelo multinomial (lbfgs es el solver recomendado para multinomial)
    modelo = LogisticRegression(
        solver=config.LR_SOLVER,             # "lbfgs"
        class_weight="balanced",             # compensa desequilibrio de clases
        max_iter=config.LR_MAX_ITER,         # 1000
        random_state=config.LR_RANDOM_STATE, # 42
        C=1.0,                               # regularización L2 por defecto
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # suprimir ConvergenceWarning si ocurre
        modelo.fit(X_train_sc, y_train)

    return modelo, scaler


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predecir_clasificacion(
    modelo_cls: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Predice la clase ("Sube", "Baja" o "Estable") para cada fila de X.

    Aplica el scaler ANTES de predecir (misma transformación que en train).

    Devuelve:
        np.ndarray de strings con la clase predicha por fila.
    """
    X_sc = scaler.transform(X)
    return modelo_cls.predict(X_sc)


def predecir_probabilidades(
    modelo_cls: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Devuelve la matriz de probabilidades posterior para cada fila de X.

    Las columnas se reordenan para coincidir SIEMPRE con:
        config.CLASES = ["Baja", "Estable", "Sube"]
    independientemente del orden interno de modelo_cls.classes_.

    Parámetros:
        modelo_cls, scaler: devueltos por entrenar_clasificacion().
        X:                  DataFrame con las mismas features que X_train.

    Devuelve:
        np.ndarray (n_samples × 3):
            columna 0 → P("Baja")
            columna 1 → P("Estable")
            columna 2 → P("Sube")
    """
    X_sc = scaler.transform(X)
    proba_raw = modelo_cls.predict_proba(X_sc)          # (n × n_classes)

    # Reordenar columnas según config.CLASES (por si el modelo tiene otro orden)
    clases_modelo = list(modelo_cls.classes_)
    idx_orden = [clases_modelo.index(c) for c in config.CLASES]
    return proba_raw[:, idx_orden]


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluar_clasificacion(
    modelo_cls: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
    y: pd.Series,
    conjunto: str = "test",
) -> dict:
    """
    Calcula métricas de clasificación multiclase sobre el conjunto indicado.

    Métricas reportadas:
      · Accuracy:         fracción de predicciones correctas.
      · Precision_macro:  media de precisiones por clase (sin ponderar por frecuencia).
      · Recall_macro:     media de recalls por clase (clave para clases minoritarias).
      · F1_macro:         media harmónica de precisión y recall por clase.
      · F1_weighted:      F1 ponderado por soporte de cada clase.
      · ROC_AUC_ovr:      área bajo curva ROC con estrategia One-vs-Rest, media macro.

    Nota sobre ROC AUC:
      Si alguna clase no aparece en y (p.ej. "Sube" ausente en test 2024-2025),
      el AUC se calcula sobre las clases presentes y se indica en el resultado.

    IMPORTANTE: usar siempre X_test / y_test para el informe final.

    Devuelve:
        dict con las métricas redondeadas a 4 decimales + clave "conjunto".
    """
    y_pred = predecir_clasificacion(modelo_cls, scaler, X)
    proba  = predecir_probabilidades(modelo_cls, scaler, X)

    acc           = accuracy_score(y, y_pred)
    prec_macro    = precision_score(y, y_pred, average="macro",    zero_division=0, labels=config.CLASES)
    rec_macro     = recall_score   (y, y_pred, average="macro",    zero_division=0, labels=config.CLASES)
    f1_macro      = f1_score       (y, y_pred, average="macro",    zero_division=0, labels=config.CLASES)
    f1_weighted   = f1_score       (y, y_pred, average="weighted", zero_division=0, labels=config.CLASES)

    # ROC AUC OvR — requiere binarización
    try:
        # Binarizar con todas las clases posibles (aunque alguna no aparezca en y)
        y_bin = pd.get_dummies(
            pd.Categorical(y, categories=config.CLASES)
        ).values.astype(float)
        roc_auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        roc_auc = round(float(roc_auc), 4)
    except Exception:
        roc_auc = float("nan")

    return {
        "Accuracy":        round(acc,         4),
        "Precision_macro": round(prec_macro,  4),
        "Recall_macro":    round(rec_macro,   4),
        "F1_macro":        round(f1_macro,    4),
        "F1_weighted":     round(f1_weighted, 4),
        "ROC_AUC_ovr":     roc_auc,
        "conjunto":        conjunto,
    }


def matriz_confusion(
    modelo_cls: LogisticRegression,
    scaler: StandardScaler,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Calcula la matriz de confusión con clases en orden config.CLASES.

    Filas    = clase real     (Baja / Estable / Sube)
    Columnas = clase predicha (Baja / Estable / Sube)

    Devuelve:
        pd.DataFrame (3 × 3) con etiquetas de fila y columna.
    """
    y_pred = predecir_clasificacion(modelo_cls, scaler, X)
    cm = confusion_matrix(y, y_pred, labels=config.CLASES)
    return pd.DataFrame(
        cm,
        index   = [f"Real_{c}"     for c in config.CLASES],
        columns = [f"Pred_{c}"     for c in config.CLASES],
    )


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def guardar_modelo(
    modelo_cls: LogisticRegression,
    scaler: StandardScaler,
    banco: str,
    instrumento: str = "tipo_oficial",
) -> Path:
    """
    Serializa el modelo y el scaler juntos en un único archivo joblib.

    Ruta: Modelo/resultados/{banco}[_OIS]_clasificacion.joblib

    Devuelve:
        Path del archivo guardado.
    """
    ruta_dir = config.MODEL_DIR / "resultados"
    ruta_dir.mkdir(parents=True, exist_ok=True)

    ruta = ruta_dir / f"{banco}{_suf(instrumento)}_clasificacion.joblib"
    joblib.dump({"modelo": modelo_cls, "scaler": scaler}, ruta)
    return ruta


def cargar_modelo(banco: str, instrumento: str = "tipo_oficial") -> tuple:
    """
    Carga el modelo y el scaler desde Modelo/resultados/{banco}[_OIS]_clasificacion.joblib

    Devuelve:
        (modelo_cls, scaler) — listos para predecir_clasificacion() / predecir_probabilidades().

    Lanza:
        FileNotFoundError si el archivo no existe.
    """
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{_suf(instrumento)}_clasificacion.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {ruta}\n"
            "Ejecuta primero: python src/models/classification.py"
        )
    d = joblib.load(ruta)
    return d["modelo"], d["scaler"]


# ---------------------------------------------------------------------------
# Pipeline completo por banco
# ---------------------------------------------------------------------------

def ejecutar_pipeline_banco(banco: str, instrumento: str = "tipo_oficial") -> dict:
    """
    Ejecuta el pipeline completo de clasificación para un banco central.

    Parámetros:
        banco:       "BCE", "BoE" o "FED".
        instrumento: "tipo_oficial" (default) o "OIS".

    Pasos:
      1. Carga dataset procesado (Datos procesados/dataset_{banco}.csv)
      2. Prepara X, y_cls (feature_engineering.preparar_X_y)
      3. División temporal train/test
      4. Entrena LogisticRegression con class_weight='balanced' + StandardScaler
      5. Evalúa en train y test (Accuracy, F1_macro, F1_weighted, ROC_AUC)
      6. Calcula matrices de confusión (train y test)
      7. Guarda modelo en Modelo/resultados/{banco}[_OIS]_clasificacion.joblib
      8. Guarda predicciones y matrices en Resultados/

    Devuelve:
        dict con modelos, métricas, matrices de confusión y datos de split.
    """
    from src.data.data_loader import cargar_datos_procesados

    print(f"\n{'='*62}")
    print(f"  CLASIFICACION — {banco} [{instrumento}]")
    print(f"{'='*62}")

    # 1. Cargar datos
    target_col = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    df = cargar_datos_procesados(banco)
    X, y_reg, y_cls, fechas, tipo_actual = preparar_X_y(df, banco, target_col=target_col)  # 5-tuple

    # 2. División temporal
    (X_train, X_test,
     y_train_reg, y_test_reg,
     y_train_cls, y_test_cls,
     fechas_train, fechas_test) = dividir_temporal(X, y_reg, y_cls, fechas)

    print(f"  Train: {len(X_train)} obs  "
          f"({fechas_train[0].strftime('%Y-%m')} a {fechas_train[-1].strftime('%Y-%m')})")
    print(f"  Test:  {len(X_test)} obs  "
          f"({fechas_test[0].strftime('%Y-%m')} a {fechas_test[-1].strftime('%Y-%m')})")

    # Distribución de clases en train y test
    dist_train = y_train_cls.value_counts().to_dict()
    dist_test  = y_test_cls.value_counts().to_dict()
    print(f"  Dist train: Baja={dist_train.get('Baja',0)}  "
          f"Estable={dist_train.get('Estable',0)}  "
          f"Sube={dist_train.get('Sube',0)}")
    print(f"  Dist test:  Baja={dist_test.get('Baja',0)}  "
          f"Estable={dist_test.get('Estable',0)}  "
          f"Sube={dist_test.get('Sube',0)}")

    # 3. Entrenamiento
    print(f"\n  [Logistic Regression]  solver={config.LR_SOLVER}  "
          f"class_weight=balanced  C=1.0  max_iter={config.LR_MAX_ITER}")
    modelo_cls, scaler = entrenar_clasificacion(X_train, y_train_cls)
    print(f"  Clases aprendidas: {list(modelo_cls.classes_)}")
    print(f"  Iteraciones usadas: {modelo_cls.n_iter_[0]}")

    # 4. Evaluación train y test
    met_train = evaluar_clasificacion(modelo_cls, scaler, X_train, y_train_cls, "train")
    met_test  = evaluar_clasificacion(modelo_cls, scaler, X_test,  y_test_cls,  "test")

    print(f"\n  TRAIN:")
    print(f"    Accuracy={met_train['Accuracy']:.4f}  "
          f"F1_macro={met_train['F1_macro']:.4f}  "
          f"F1_weighted={met_train['F1_weighted']:.4f}  "
          f"ROC_AUC={met_train['ROC_AUC_ovr']}")
    print(f"  TEST:")
    print(f"    Accuracy={met_test['Accuracy']:.4f}  "
          f"F1_macro={met_test['F1_macro']:.4f}  "
          f"F1_weighted={met_test['F1_weighted']:.4f}  "
          f"ROC_AUC={met_test['ROC_AUC_ovr']}")

    # 5. Matrices de confusión
    cm_train = matriz_confusion(modelo_cls, scaler, X_train, y_train_cls)
    cm_test  = matriz_confusion(modelo_cls, scaler, X_test,  y_test_cls)

    print(f"\n  Matriz de confusion TEST:")
    print(cm_test.to_string())

    # 6. Guardar modelo
    ruta_modelo = guardar_modelo(modelo_cls, scaler, banco, instrumento=instrumento)
    print(f"\n  Modelo guardado: {ruta_modelo.name}")

    # 7. Guardar predicciones + probabilidades del test
    proba_test = predecir_probabilidades(modelo_cls, scaler, X_test)
    y_pred_test = predecir_clasificacion(modelo_cls, scaler, X_test)

    df_pred = pd.DataFrame({
        "fecha":       fechas_test,
        "y_real":      y_test_cls.values,
        "y_pred":      y_pred_test,
        "correcto":    (y_pred_test == y_test_cls.values).astype(int),
        "P_Baja":      proba_test[:, 0],
        "P_Estable":   proba_test[:, 1],
        "P_Sube":      proba_test[:, 2],
    }).set_index("fecha")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_pred = config.RESULTS_DIR / f"predicciones_test_clasificacion_{banco}{_suf(instrumento)}.csv"
    df_pred.to_csv(ruta_pred)
    print(f"  Predicciones test guardadas: {ruta_pred.name}")

    # 8. Guardar matrices de confusión
    ruta_cm_train = config.RESULTS_DIR / f"confusion_train_{banco}{_suf(instrumento)}.csv"
    ruta_cm_test  = config.RESULTS_DIR / f"confusion_test_{banco}{_suf(instrumento)}.csv"
    cm_train.to_csv(ruta_cm_train)
    cm_test.to_csv(ruta_cm_test)

    # 9. Guardar reporte de clasificación completo (test)
    reporte = classification_report(
        y_test_cls,
        y_pred_test,
        labels=config.CLASES,
        zero_division=0,
    )
    ruta_reporte = config.RESULTS_DIR / f"reporte_clasificacion_{banco}{_suf(instrumento)}.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    return {
        "banco":        banco,
        "instrumento":  instrumento,
        "modelo_cls":   modelo_cls,
        "scaler":       scaler,
        "met_train":    met_train,
        "met_test":     met_test,
        "cm_train":     cm_train,
        "cm_test":      cm_test,
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train_cls":  y_train_cls,
        "y_test_cls":   y_test_cls,
        "fechas_train": fechas_train,
        "fechas_test":  fechas_test,
        "pred_test":    df_pred,
    }


# ---------------------------------------------------------------------------
# Resumen consolidado de los 3 bancos
# ---------------------------------------------------------------------------

def guardar_resumen_metricas(todos_resultados: list, instrumento: str = "tipo_oficial") -> pd.DataFrame:
    """
    Consolida las métricas de clasificación de los 3 bancos en un único CSV.

    Guarda: Resultados/resumen_clasificacion[_OIS].csv
    Devuelve: DataFrame con todas las métricas por banco y conjunto.
    """
    filas = []
    for res in todos_resultados:
        banco = res["banco"]
        for conjunto, met in [("train", res["met_train"]), ("test", res["met_test"])]:
            filas.append({
                "Banco":           banco,
                "Conjunto":        conjunto,
                "Accuracy":        met["Accuracy"],
                "Precision_macro": met["Precision_macro"],
                "Recall_macro":    met["Recall_macro"],
                "F1_macro":        met["F1_macro"],
                "F1_weighted":     met["F1_weighted"],
                "ROC_AUC_ovr":     met["ROC_AUC_ovr"],
            })

    df_resumen = pd.DataFrame(filas)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTS_DIR / f"resumen_clasificacion{_suf(instrumento)}.csv"
    df_resumen.to_csv(ruta, index=False)
    print(f"\n  Resumen guardado: {ruta.name}")
    return df_resumen


# ---------------------------------------------------------------------------
# Evaluación adicional: ciclo de subidas 2022-2023
# ---------------------------------------------------------------------------

def ejecutar_evaluacion_hikecycle(
    banco: str,
    instrumento: str = "tipo_oficial",
) -> dict:
    """
    Evaluación adicional del clasificador sobre el ciclo de subidas 2022-2023.

    El test principal (2024-2025) no contiene observaciones de la clase "Sube"
    porque ese período fue un ciclo exclusivo de bajadas. Esta evaluación usa una
    división temporal alternativa para validar que el clasificador SÍ detecta
    la clase "Sube" cuando existen observaciones de ella:

      · Train: 2000-2021 (corte en 2022-01-01)  ← mismo modelo que el principal
      · Test:  2022-2023 (24 meses)             ← ciclo de subidas agresivas

    2022 fue el año de mayor agresividad alcista de los últimos 20 años
    (BCE: 0% → 2.5%; BoE: 0.25% → 3.5%; FED: 0% → 4.5%), por lo que el test
    contiene las tres clases: Baja, Estable y Sube.

    IMPORTANTE: este modelo es INDEPENDIENTE del modelo principal.
      · No sobreescribe ningún archivo en Modelo/resultados/.
      · Los resultados se guardan con sufijo '_hikecycle' en Resultados/.
      · El objetivo es validación académica, no producción.

    Parámetros:
        banco:       "BCE", "BoE" o "FED".
        instrumento: "tipo_oficial" (default) o nombre OIS ("ESTR"/"SONIA"/"SOFR").

    Devuelve:
        dict con modelo, métricas, matriz de confusión, datos del split y
        distribución de clases en el test.
    """
    from src.data.data_loader import cargar_datos_procesados

    print(f"\n{'='*62}")
    print(f"  CLASIFICACION CICLO SUBIDAS 2022-2023 — {banco} [{instrumento}]")
    print(f"{'='*62}")

    target_col = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    df = cargar_datos_procesados(banco)
    X, y_reg, y_cls, fechas, tipo_actual = preparar_X_y(df, banco, target_col=target_col)

    # División alternativa: train hasta 2021-12-31, test 2022-01-01 a 2023-12-31
    HIKE_TEST_START = pd.Timestamp("2022-01-01")
    HIKE_TEST_END   = pd.Timestamp("2023-12-31")

    mask_train = fechas < HIKE_TEST_START
    mask_test  = (fechas >= HIKE_TEST_START) & (fechas <= HIKE_TEST_END)

    X_train_hk = X.loc[mask_train]
    X_test_hk  = X.loc[mask_test]
    y_train_hk = y_cls.loc[mask_train]
    y_test_hk  = y_cls.loc[mask_test]
    fechas_train_hk = fechas[mask_train]
    fechas_test_hk  = fechas[mask_test]

    print(f"  Train: {len(X_train_hk)} obs  "
          f"({fechas_train_hk[0].strftime('%Y-%m')} a {fechas_train_hk[-1].strftime('%Y-%m')})")
    print(f"  Test:  {len(X_test_hk)} obs  "
          f"({fechas_test_hk[0].strftime('%Y-%m')} a {fechas_test_hk[-1].strftime('%Y-%m')})")

    dist_train = y_train_hk.value_counts().to_dict()
    dist_test  = y_test_hk.value_counts().to_dict()
    print(f"  Dist train: Baja={dist_train.get('Baja',0)}  "
          f"Estable={dist_train.get('Estable',0)}  "
          f"Sube={dist_train.get('Sube',0)}")
    print(f"  Dist test:  Baja={dist_test.get('Baja',0)}  "
          f"Estable={dist_test.get('Estable',0)}  "
          f"Sube={dist_test.get('Sube',0)}")

    # Entrenamiento con la misma arquitectura que el modelo principal
    print(f"\n  [Logistic Regression]  solver={config.LR_SOLVER}  "
          f"class_weight=balanced  C=1.0  max_iter={config.LR_MAX_ITER}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo_cls, scaler = entrenar_clasificacion(X_train_hk, y_train_hk)
    print(f"  Clases aprendidas: {list(modelo_cls.classes_)}")

    # Evaluación sobre el test 2022-2023
    met_test = evaluar_clasificacion(modelo_cls, scaler, X_test_hk, y_test_hk, "test_hikecycle")
    cm_test  = matriz_confusion(modelo_cls, scaler, X_test_hk, y_test_hk)

    print(f"\n  TEST CICLO SUBIDAS (2022-2023):")
    print(f"    Accuracy={met_test['Accuracy']:.4f}  "
          f"F1_macro={met_test['F1_macro']:.4f}  "
          f"F1_weighted={met_test['F1_weighted']:.4f}  "
          f"ROC_AUC={met_test['ROC_AUC_ovr']}")
    print(f"\n  Matriz de confusion TEST (2022-2023):")
    print(cm_test.to_string())

    # Guardar resultados con sufijo _hikecycle (no sobreescribe los principales)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ruta_cm = config.RESULTS_DIR / f"confusion_test_{banco}{_suf(instrumento)}_hikecycle.csv"
    cm_test.to_csv(ruta_cm)

    y_pred_test = predecir_clasificacion(modelo_cls, scaler, X_test_hk)
    reporte = classification_report(
        y_test_hk,
        y_pred_test,
        labels=config.CLASES,
        zero_division=0,
    )
    ruta_reporte = config.RESULTS_DIR / f"reporte_clasificacion_{banco}{_suf(instrumento)}_hikecycle.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    print(f"\n  Guardado: {ruta_cm.name}")
    print(f"  Guardado: {ruta_reporte.name}")

    return {
        "banco":        banco,
        "instrumento":  instrumento,
        "modelo_cls":   modelo_cls,
        "scaler":       scaler,
        "met_test":     met_test,
        "cm_test":      cm_test,
        "X_train":      X_train_hk,
        "X_test":       X_test_hk,
        "y_train_cls":  y_train_hk,
        "y_test_cls":   y_test_hk,
        "fechas_train": fechas_train_hk,
        "fechas_test":  fechas_test_hk,
        "dist_test":    dist_test,
    }


def guardar_resumen_hikecycle(
    todos_resultados: list,
    instrumento: str = "tipo_oficial",
) -> pd.DataFrame:
    """
    Consolida las métricas del ciclo de subidas 2022-2023 en un único CSV.

    Guarda: Resultados/resumen_clasificacion_hikecycle[_{instrumento}].csv
    Columnas extra respecto al resumen principal: N_Baja, N_Estable, N_Sube
    (distribución de clases del periodo de test, para mostrar que "Sube" existe).

    Devuelve:
        pd.DataFrame con una fila por banco.
    """
    filas = []
    for res in todos_resultados:
        banco = res["banco"]
        met   = res["met_test"]
        dist  = res["dist_test"]
        filas.append({
            "Banco":           banco,
            "Periodo":         "2022-2023",
            "N_Baja":          dist.get("Baja",    0),
            "N_Estable":       dist.get("Estable", 0),
            "N_Sube":          dist.get("Sube",    0),
            "Accuracy":        met["Accuracy"],
            "Precision_macro": met["Precision_macro"],
            "Recall_macro":    met["Recall_macro"],
            "F1_macro":        met["F1_macro"],
            "F1_weighted":     met["F1_weighted"],
            "ROC_AUC_ovr":     met["ROC_AUC_ovr"],
        })

    df_resumen = pd.DataFrame(filas)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTS_DIR / f"resumen_clasificacion_hikecycle{_suf(instrumento)}.csv"
    df_resumen.to_csv(ruta, index=False)
    print(f"\n  Resumen hikecycle guardado: {ruta.name}")
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
    print("  RESUMEN FINAL — METRICAS TEST (Logistic Regression)")
    print("="*62)
    tabla_test = df_resumen[df_resumen["Conjunto"] == "test"][
        ["Banco", "Accuracy", "F1_macro", "F1_weighted", "ROC_AUC_ovr"]
    ].reset_index(drop=True)
    print(tabla_test.to_string(index=False))

    # Baseline de comparacion: clasificador trivial "siempre Estable"
    print("\n" + "="*62)
    print("  BASELINE — siempre 'Estable' (sin modelo)")
    print("="*62)
    print("  BCE: Accuracy=0.6250  F1_macro~0.19  (15/24 Estable en test)")
    print("  BoE: Accuracy=0.7917  F1_macro~0.22  (19/24 Estable en test)")
    print("  FED: Accuracy=0.6667  F1_macro~0.20  (16/24 Estable en test)")
