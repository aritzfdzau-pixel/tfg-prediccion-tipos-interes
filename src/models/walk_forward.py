"""
walk_forward.py — Validación walk-forward con 3 ventanas temporales expandidas

Refuerza la robustez estadística de las métricas más allá del único corte
2024-2025. En lugar de un solo split temporal, se entrena el modelo en
ventanas progresivamente más largas y se evalúa sobre el año siguiente:

  Ventana 1: Train 2000-2021 → Test 2022        (12 meses)
  Ventana 2: Train 2000-2022 → Test 2023        (12 meses)
  Ventana 3: Train 2000-2023 → Test 2024-2025   (24 meses) ← test principal

Si el MAE es consistente en las tres ventanas, las métricas no son fruto
de un período favorable sino de la capacidad real del modelo.

FUGA TEMPORAL — garantias:
  · Cada ventana entrena SOLO sobre datos anteriores al test.
  · El StandardScaler (Ridge) se ajusta solo sobre X_train de esa ventana.
  · No hay información futura en ningún paso de entrenamiento.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.data.data_loader import cargar_datos_procesados
from src.data.feature_engineering import preparar_X_y
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Definición de ventanas
# ---------------------------------------------------------------------------

WINDOWS = [
    ("Ventana 1",              "2022-01-01", "2022-12-31"),
    ("Ventana 2",              "2023-01-01", "2023-12-31"),
    ("Ventana 3 (test oficial)", "2024-01-01", "2025-12-31"),
]


# ---------------------------------------------------------------------------
# Pipeline por banco y ventana
# ---------------------------------------------------------------------------

def _entrenar_ventana(X, y_reg, fechas, tipo_actual, test_start: str, test_end: str):
    """
    Entrena el RF sobre todo lo anterior a test_start y evalúa en [test_start, test_end].
    Devuelve dict con MAE, RMSE, R² sobre el nivel reconstruido.
    """
    ts  = pd.Timestamp(test_start)
    te  = pd.Timestamp(test_end)

    mask_train = fechas < ts
    mask_test  = (fechas >= ts) & (fechas <= te)

    if mask_train.sum() < 36 or mask_test.sum() < 1:
        return None

    X_train     = X.loc[mask_train]
    X_test      = X.loc[mask_test]
    y_train     = y_reg.loc[mask_train]
    y_test      = y_reg.loc[mask_test]
    tipo_test   = tipo_actual.loc[mask_test]

    # Entrenar RF con los mismos hiperparámetros que el modelo principal
    modelo = RandomForestRegressor(
        n_estimators    = config.RF_N_ESTIMATORS,
        max_depth       = config.RF_MAX_DEPTH,
        min_samples_leaf= config.RF_MIN_SAMPLES_LEAF,
        max_features    = config.RF_MAX_FEATURES,
        random_state    = config.RF_RANDOM_STATE,
        n_jobs          = -1,
    )
    modelo.fit(X_train, y_train)

    # Predecir deltas → reconstruir nivel (modo paralelo, igual que en test)
    delta_pred = modelo.predict(X_test)
    nivel_pred = np.clip(tipo_test.values + delta_pred,  0.0, 25.0)
    nivel_real = np.clip(tipo_test.values + y_test.values, 0.0, 25.0)

    mae  = round(float(mean_absolute_error(nivel_real, nivel_pred)),  4)
    rmse = round(float(np.sqrt(mean_squared_error(nivel_real, nivel_pred))), 4)
    r2   = round(float(r2_score(nivel_real, nivel_pred)),               4)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "N_test": int(mask_test.sum())}


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def ejecutar_walk_forward() -> pd.DataFrame:
    """
    Ejecuta la validación walk-forward para los 3 bancos y las 3 ventanas.
    Guarda Resultados/walk_forward_validation.csv y devuelve el DataFrame.
    """
    resultados = []

    for banco in config.BANCOS_CENTRALES:
        print(f"\n{'='*56}")
        print(f"  WALK-FORWARD — {banco}")
        print(f"{'='*56}")

        df = cargar_datos_procesados(banco)
        X, y_reg, y_cls, fechas, tipo_actual = preparar_X_y(df, banco)

        for nombre, test_start, test_end in WINDOWS:
            met = _entrenar_ventana(X, y_reg, fechas, tipo_actual, test_start, test_end)
            if met is None:
                print(f"  {nombre}: datos insuficientes, omitida.")
                continue

            n_train = (fechas < pd.Timestamp(test_start)).sum()
            print(
                f"  {nombre:<30}  "
                f"Train={n_train} obs | Test={met['N_test']} obs | "
                f"MAE={met['MAE']:.4f} pp | R2={met['R2']:.4f}"
            )

            resultados.append({
                "Banco":       banco,
                "Ventana":     nombre,
                "Test_inicio": test_start,
                "Test_fin":    test_end,
                "N_train":     n_train,
                "N_test":      met["N_test"],
                "MAE":         met["MAE"],
                "RMSE":        met["RMSE"],
                "R2":          met["R2"],
            })

    df_res = pd.DataFrame(resultados)

    # Añadir MAE medio por banco
    print(f"\n{'='*56}")
    print("  RESUMEN — MAE medio de las 3 ventanas")
    print(f"{'='*56}")
    for banco in config.BANCOS_CENTRALES:
        mae_medio = df_res[df_res["Banco"] == banco]["MAE"].mean()
        r2_medio  = df_res[df_res["Banco"] == banco]["R2"].mean()
        print(f"  {banco}: MAE_medio={mae_medio:.4f} pp | R2_medio={r2_medio:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.RESULTS_DIR / "walk_forward_validation.csv"
    df_res.to_csv(ruta, index=False)
    print(f"\n  Guardado: {ruta.name}")
    return df_res


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = ejecutar_walk_forward()
