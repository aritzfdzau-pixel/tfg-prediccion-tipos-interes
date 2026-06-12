"""
test_iterativo_ois.py — Prueba en paralelo del enfoque iterativo RF para OIS

Compara dos enfoques para la predicción futura del escenario personalizado OIS:

  A) Taylor Rule (actual): nivel predicho = OIS_taylor constante
  B) RF iterativo (nuevo):  nivel_t = nivel_{t-1} + RF.predict(delta_t)

Para evaluar cuál es mejor se simula el escenario "personalizado con features
constantes" sobre el PERIODO DE TEST real (2024-2025, 24 meses), donde sí
tenemos los valores verdaderos del OIS para comparar.

Configuración del test: las features se fijan a la media del año 2023
(equivalente a lo que haría un slider con valores "normalizados"),
y se predice iterativamente desde dic-2023.

Ejecutar:
    python test_iterativo_ois.py
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.data_loader import cargar_datos_procesados
from src.data.feature_engineering import preparar_X_y
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suf(instrumento: str) -> str:
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


def cargar_rf(banco: str, instrumento: str):
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{_suf(instrumento)}_regresion.joblib"
    d = joblib.load(ruta)
    return d["rf"]   # RandomForestRegressor entrenado


def get_features_constantes(banco: str, instrumento: str) -> pd.Series:
    """
    Devuelve las features promedio del año 2023 para simular
    los sliders con valores 'representativos'.
    """
    target_col = config.INSTRUMENT_TO_COL.get(instrumento, "OIS")
    df = cargar_datos_procesados(banco)
    X_full, _, _, fechas_full, _ = preparar_X_y(df, banco, target_col=target_col)
    mask_2023 = (fechas_full.year == 2023)
    return X_full[mask_2023].mean()


def predecir_taylor_rule(banco: str, features: pd.Series, tipo_inicio: float,
                          n_meses: int = 24) -> np.ndarray:
    """
    Enfoque A (actual): nivel = OIS_taylor constante tras blend de 12 meses.
    """
    p = config.TAYLOR_PARAMS[banco]
    ois_taylor = max(0.0,
        p["r_star"]
        + 1.5 * (features["Inflacion"] - p["pi_star"])
        + 0.5 * (features["PIB_growth"] - p["y_star"])
    )
    niveles = np.full(n_meses, ois_taylor)

    # Blend de 12 meses desde tipo_inicio
    N_BLEND = 12
    for i in range(min(N_BLEND, n_meses)):
        alpha = (i + 1) / N_BLEND
        niveles[i] = (1 - alpha) * tipo_inicio + alpha * ois_taylor

    return np.clip(niveles, 0.0, 25.0)


def predecir_rf_iterativo(banco: str, instrumento: str,
                           features_base: pd.Series,
                           tipo_inicio: float,
                           n_meses: int = 24) -> np.ndarray:
    """
    Enfoque B (nuevo): nivel_t = nivel_{t-1} + RF.predict(Δ_t)
    Las features macro son constantes (= media 2023).
    Las features dinámicas (brecha_taylor, delta_tipo_lag1, meses_sin_cambio)
    se actualizan en cada paso.
    """
    rf = cargar_rf(banco, instrumento)
    p  = config.TAYLOR_PARAMS[banco]

    # Calcular OIS_taylor de las features constantes
    ois_taylor_const = max(0.0,
        p["r_star"]
        + 1.5 * (features_base["Inflacion"] - p["pi_star"])
        + 0.5 * (features_base["PIB_growth"] - p["y_star"])
    )

    niveles         = np.zeros(n_meses)
    nivel_actual    = tipo_inicio
    delta_lag1      = 0.0
    meses_sin_cambio = 0

    for t in range(n_meses):
        # Construir vector de features para este mes
        row = features_base.copy()

        # Features dinámicas actualizadas
        row["OIS_taylor"]      = ois_taylor_const
        row["brecha_taylor"]   = ois_taylor_const - nivel_actual
        row["delta_tipo_lag1"] = delta_lag1
        row["meses_sin_cambio"] = meses_sin_cambio

        X_t = row[config.VARIABLES_MODELO].values.reshape(1, -1)
        delta_pred = float(rf.predict(X_t)[0])

        nuevo_nivel = np.clip(nivel_actual + delta_pred, 0.0, 25.0)
        niveles[t]  = nuevo_nivel

        # Actualizar estado
        cambio = nuevo_nivel - nivel_actual
        delta_lag1  = cambio
        meses_sin_cambio = 0 if abs(cambio) > config.THRESHOLD_ESTABILIDAD else meses_sin_cambio + 1
        nivel_actual = nuevo_nivel

    return niveles


def evaluar_banco(banco: str, instrumento: str) -> dict:
    """
    Evalúa ambos enfoques sobre el test real 2024-2025.
    """
    target_col = config.INSTRUMENT_TO_COL.get(instrumento, "OIS")
    df = cargar_datos_procesados(banco)
    _, _, _, fechas_full, tipo_full = preparar_X_y(df, banco, target_col=target_col)

    # Dividir train/test
    mask_train = fechas_full <= pd.Timestamp(config.TRAIN_END)
    mask_test  = fechas_full >= pd.Timestamp(config.TEST_START)

    tipo_inicio = float(tipo_full[mask_train].iloc[-1])   # dic-2023
    tipo_real   = tipo_full[mask_test].values              # 24 valores reales OIS 2024-2025

    # Features constantes = media 2023
    features_const = get_features_constantes(banco, instrumento)

    # Enfoque A: Taylor Rule
    pred_taylor = predecir_taylor_rule(banco, features_const, tipo_inicio, n_meses=24)

    # Enfoque B: RF iterativo
    pred_rf_iter = predecir_rf_iterativo(banco, instrumento, features_const,
                                          tipo_inicio, n_meses=24)

    # Métricas
    mae_taylor   = mean_absolute_error(tipo_real, pred_taylor)
    mae_rf_iter  = mean_absolute_error(tipo_real, pred_rf_iter)
    rmse_taylor  = np.sqrt(mean_squared_error(tipo_real, pred_taylor))
    rmse_rf_iter = np.sqrt(mean_squared_error(tipo_real, pred_rf_iter))

    mejora_mae   = (1 - mae_rf_iter / mae_taylor) * 100   # % de mejora

    # Serie real para contexto
    tipo_inicio_str = f"{tipo_inicio:.3f}%"
    tipo_real_final = f"{tipo_real[-1]:.3f}%"
    pred_taylor_fin = f"{pred_taylor[-1]:.3f}%"
    pred_rf_fin     = f"{pred_rf_iter[-1]:.3f}%"

    return {
        "banco":          banco,
        "instrumento":    instrumento,
        "tipo_inicio":    tipo_inicio,
        "tipo_real_fin":  tipo_real[-1],
        "MAE_taylor":     round(mae_taylor,  4),
        "MAE_rf_iter":    round(mae_rf_iter, 4),
        "RMSE_taylor":    round(rmse_taylor,  4),
        "RMSE_rf_iter":   round(rmse_rf_iter, 4),
        "mejora_MAE_pct": round(mejora_mae, 1),
        "pred_taylor_fin": pred_taylor[-1],
        "pred_rf_fin":     pred_rf_iter[-1],
        "_pred_taylor":   pred_taylor,
        "_pred_rf_iter":  pred_rf_iter,
        "_tipo_real":     tipo_real,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  TEST PARALELO: Taylor Rule vs RF Iterativo — OIS (test 2024-2025)")
    print("=" * 70)
    print("\nConfiguración: features fijadas a la media de 2023 (= sliders típicos)")
    print("Objetivo: medir cuál predice mejor el OIS real cuando los inputs son constantes\n")

    resultados = []
    for banco in config.BANCOS_CENTRALES:
        instrumento = config.OIS_INSTRUMENT_NAMES[banco]
        print(f"{'─'*70}")
        print(f"  {banco} — {instrumento}")
        print(f"{'─'*70}")

        r = evaluar_banco(banco, instrumento)
        resultados.append(r)

        print(f"  Punto de partida (dic-2023):      {r['tipo_inicio']:.3f}%")
        print(f"  Valor real final (dic-2025):       {r['tipo_real_fin']:.3f}%")
        print(f"  Predicción final Taylor Rule:      {r['pred_taylor_fin']:.3f}%")
        print(f"  Predicción final RF iterativo:     {r['pred_rf_fin']:.3f}%")
        print()
        print(f"  {'Métrica':<20} {'Taylor Rule':>14} {'RF Iterativo':>14} {'Mejora':>10}")
        print(f"  {'─'*60}")
        print(f"  {'MAE (pp)':<20} {r['MAE_taylor']:>14.4f} {r['MAE_rf_iter']:>14.4f} {r['mejora_MAE_pct']:>+9.1f}%")
        print(f"  {'RMSE (pp)':<20} {r['RMSE_taylor']:>14.4f} {r['RMSE_rf_iter']:>14.4f}")

        if r["mejora_MAE_pct"] > 5:
            print(f"\n  ✓  RF iterativo MEJOR en {r['mejora_MAE_pct']:+.1f}% de MAE")
        elif r["mejora_MAE_pct"] < -5:
            print(f"\n  ✗  Taylor Rule MEJOR en {-r['mejora_MAE_pct']:.1f}% de MAE")
        else:
            print(f"\n  ~  Diferencia marginal ({r['mejora_MAE_pct']:+.1f}%)")

    # ── Resumen global ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN GLOBAL")
    print("=" * 70)
    print(f"\n  {'Banco':<6} {'Inst':<6} {'MAE_Taylor':>12} {'MAE_RF_iter':>12} {'Mejora':>10}")
    print(f"  {'─'*50}")
    for r in resultados:
        print(f"  {r['banco']:<6} {r['instrumento']:<6} "
              f"{r['MAE_taylor']:>12.4f} {r['MAE_rf_iter']:>12.4f} "
              f"{r['mejora_MAE_pct']:>+9.1f}%")

    maes_taylor  = [r["MAE_taylor"]  for r in resultados]
    maes_rf      = [r["MAE_rf_iter"] for r in resultados]
    media_mejora = np.mean([r["mejora_MAE_pct"] for r in resultados])

    print(f"\n  MAE medio Taylor Rule : {np.mean(maes_taylor):.4f} pp")
    print(f"  MAE medio RF Iterativo: {np.mean(maes_rf):.4f} pp")
    print(f"  Mejora media MAE      : {media_mejora:+.1f}%")

    if media_mejora > 5:
        print("\n  CONCLUSION: RF iterativo es superior. Recomendable implementar.")
    elif media_mejora < -5:
        print("\n  CONCLUSION: Taylor Rule es superior. Mantener enfoque actual.")
    else:
        print("\n  CONCLUSION: Diferencia no significativa. Ambos enfoques son equivalentes.")

    print()
