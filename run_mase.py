"""
run_mase.py — Añade la métrica MASE a los resúmenes de regresión existentes

MASE (Mean Absolute Scaled Error, Hyndman & Koehler 2006) responde a la
pregunta: ¿cuánto mejor que el random walk es el modelo?

    MASE = MAE_test / MAE_naive_insample
    MAE_naive_insample = mean(|delta_t|) en el periodo de entrenamiento
                       = MAE del forecast "el tipo no cambia" medido en train

    MASE < 1 → el modelo bate al random walk
    MASE = 1 → equivalente al random walk

Ventaja académica: no está inflado por autocorrelación (a diferencia de R²
sobre niveles) y es la métrica estándar de benchmarking en series temporales.

Este script NO reentrena ningún modelo. Sólo:
  1. Lee los CSVs de resumen ya existentes (resumen_regresion.csv y resumen_regresion_OIS.csv)
  2. Recalcula y_train_reg en memoria a partir de los datos procesados
  3. Añade la columna MASE a los CSVs y los sobreescribe

Ejecutar:
    python run_mase.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.models.regression import actualizar_metricas_mase


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CALCULO DE MASE — Mean Absolute Scaled Error (Hyndman & Koehler 2006)")
    print("  Sin reentrenamiento — usa MAE_test de los CSVs existentes")
    print("=" * 70)

    # ── Tipos oficiales ──────────────────────────────────────────────────────
    print("\n[1/2] Tipos oficiales (BCE / BoE / FED)")
    df_oficial = actualizar_metricas_mase(instrumento="tipo_oficial")

    # ── Tipos OIS ────────────────────────────────────────────────────────────
    # El resumen OIS es único (resumen_regresion_OIS.csv), instrumento="OIS"
    print("\n[2/2] Tipos OIS (ESTR / SONIA / SOFR)")
    df_ois = actualizar_metricas_mase(instrumento="OIS")

    # ── Resumen final ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN FINAL — MASE por banco y modelo")
    print("=" * 70)

    import pandas as pd

    print("\n  Tipos Oficiales (RF):")
    df_rf = df_oficial[df_oficial["Modelo"] == "RF"][["Banco", "MAE_test", "MASE"]].copy()
    df_rf["Interpreta"] = df_rf["MASE"].apply(
        lambda v: "bate al naïve" if v < 1 else ("igual al naïve" if abs(v - 1) < 0.05 else "peor que naïve")
    )
    print(df_rf.to_string(index=False))

    print("\n  Tipos OIS (RF):")
    ruta_ois = config.RESULTS_DIR / "resumen_regresion_OIS.csv"
    if ruta_ois.exists():
        df_ois_rf = pd.read_csv(ruta_ois)
        df_ois_rf = df_ois_rf[df_ois_rf["Modelo"] == "RF"][["Banco", "MAE_test", "MASE"]].copy()
        _ois_names = config.OIS_INSTRUMENT_NAMES
        df_ois_rf["Instrumento"] = df_ois_rf["Banco"].map(_ois_names)
        df_ois_rf["Interpreta"] = df_ois_rf["MASE"].apply(
            lambda v: "bate al naive" if v < 1 else ("igual" if abs(v-1) < 0.05 else "peor que naive")
        )
        print(df_ois_rf[["Banco", "Instrumento", "MAE_test", "MASE", "Interpreta"]].to_string(index=False))

    print("\n  MASE < 1 = el modelo bate al forecast naïve 'sin cambio' en el test.\n")
