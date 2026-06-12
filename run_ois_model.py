"""
run_ois_model.py — Pipeline completo del modelo de tipos OIS

Modela los tipos OIS de cada zona económica como variable objetivo:
  BCE → €STR  (European Short-Term Rate)
  BoE → SONIA (Sterling Overnight Index Average)
  FED → SOFR  (Secured Overnight Financing Rate)

Es un modelo distinto al de tipos oficiales de bancos centrales.
La variable objetivo es el tipo de mercado overnight (OIS), no
la decisión de política monetaria del banco central.

Archivos generados (sufijo por instrumento, no por banco):
  Modelo/resultados/BCE_ESTR_regresion.joblib
  Modelo/resultados/BCE_ESTR_clasificacion.joblib
  Modelo/resultados/BoE_SONIA_regresion.joblib
  Modelo/resultados/BoE_SONIA_clasificacion.joblib
  Modelo/resultados/FED_SOFR_regresion.joblib
  Modelo/resultados/FED_SOFR_clasificacion.joblib
  Resultados/importancia_{banco}_{instrumento}.csv
  Resultados/scores_test_{banco}_{instrumento}.csv
  Resultados/predicciones_futuras_{banco}_{escenario}_{instrumento}.csv
  Resultados/resumen_regresion_OIS.csv
  Resultados/resumen_clasificacion_OIS.csv
  Resultados/resumen_scores_OIS.csv

Ejecutar con:
    python run_ois_model.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config
from src.models.regression import (
    ejecutar_pipeline_banco as run_regresion,
    guardar_resumen_metricas as save_resumen_reg,
)
from src.models.classification import (
    ejecutar_pipeline_banco as run_clasificacion,
    guardar_resumen_metricas as save_resumen_cls,
)
from src.models.scoring import ejecutar_scoring_todos_bancos, ejecutar_scoring_historico
from src.models.prediccion_futura import ejecutar_predicciones_futuras, predecir_escenario


def main():
    print("\n" + "=" * 62)
    print("  PIPELINE — MODELOS DE TIPOS OIS")
    print("  BCE: €STR  |  BoE: SONIA  |  FED: SOFR")
    print("=" * 62)

    # ── Fase 5: Regresión ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FASE 5 — REGRESIÓN (Random Forest + Ridge CV)")
    print("=" * 62)

    todos_reg = []
    for banco in config.BANCOS_CENTRALES:
        instrumento = config.OIS_INSTRUMENT_NAMES[banco]   # ESTR / SONIA / SOFR
        res = run_regresion(banco, instrumento=instrumento)
        todos_reg.append(res)

    # Resumen conjunto con etiqueta genérica "OIS"
    df_resumen_reg = save_resumen_reg(todos_reg, instrumento="OIS")

    print("\n  RESUMEN FINAL — MÉTRICAS TEST (Random Forest)")
    print("=" * 62)
    tabla_rf = df_resumen_reg[df_resumen_reg["Modelo"] == "RF"][
        ["Banco", "MAE_test", "RMSE_test", "R2_test"]
    ].reset_index(drop=True)
    print(tabla_rf.to_string(index=False))

    # ── Fase 6: Clasificación ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FASE 6 — CLASIFICACIÓN (Logistic Regression Multinomial)")
    print("=" * 62)

    todos_cls = []
    for banco in config.BANCOS_CENTRALES:
        instrumento = config.OIS_INSTRUMENT_NAMES[banco]
        res = run_clasificacion(banco, instrumento=instrumento)
        todos_cls.append(res)

    df_resumen_cls = save_resumen_cls(todos_cls, instrumento="OIS")

    print("\n  RESUMEN FINAL — MÉTRICAS TEST")
    print("=" * 62)
    tabla_test = df_resumen_cls[df_resumen_cls["Conjunto"] == "test"][
        ["Banco", "Accuracy", "F1_macro", "F1_weighted", "ROC_AUC_ovr"]
    ].reset_index(drop=True)
    print(tabla_test.to_string(index=False))

    # ── Fase 7: Scoring ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FASE 7 — SCORING (Confidence / Upward / Downward / Net)")
    print("=" * 62)

    todos_scores = {}
    for banco in config.BANCOS_CENTRALES:
        instrumento = config.OIS_INSTRUMENT_NAMES[banco]
        print(f"\n{'='*62}")
        print(f"  SCORING — {banco} [{instrumento}]")
        print(f"{'='*62}")
        tabla = ejecutar_scoring_historico(banco, instrumento=instrumento)
        todos_scores[banco] = tabla

        conf_media = tabla["Confidence_Score"].mean()
        up_media   = tabla["Upward_Score"].mean()
        down_media = tabla["Downward_Score"].mean()
        net_media  = tabla["Net_Score"].mean()
        print(f"\n  Scores medios test (2024-2025):")
        print(f"    Confidence_Score : {conf_media:.1f}")
        print(f"    Upward_Score     : {up_media:.1f}")
        print(f"    Downward_Score   : {down_media:.1f}")
        print(f"    Net_Score        : {net_media:.1f}")

    # Guardar resumen scores OIS
    filas_sc = []
    for banco, tabla in todos_scores.items():
        instrumento = config.OIS_INSTRUMENT_NAMES[banco]
        filas_sc.append({
            "Banco":            banco,
            "Instrumento":      instrumento,
            "Confidence_media": round(tabla["Confidence_Score"].mean(), 1),
            "Upward_media":     round(tabla["Upward_Score"].mean(), 1),
            "Downward_media":   round(tabla["Downward_Score"].mean(), 1),
            "Net_media":        round(tabla["Net_Score"].mean(), 1),
        })
    df_sc_res = pd.DataFrame(filas_sc)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_sc_res.to_csv(config.RESULTS_DIR / "resumen_scores_OIS.csv", index=False)
    print(f"\n  Resumen scores guardado: resumen_scores_OIS.csv")

    # ── Fase 8: Predicciones futuras ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  FASE 8 — PREDICCIONES FUTURAS 2026-2030 (3 escenarios)")
    print("=" * 62)

    from src.models.prediccion_futura import poblar_y_guardar_escenarios
    print("Generando CSVs de escenarios...")
    poblar_y_guardar_escenarios()

    escenarios = ["Base", "Optimista", "Pesimista"]
    for banco in config.BANCOS_CENTRALES:
        instrumento = config.OIS_INSTRUMENT_NAMES[banco]
        print(f"\n{'='*62}")
        print(f"  PREDICCION FUTURA — {banco} [{instrumento}]")
        print(f"{'='*62}")
        for escenario in escenarios:
            tabla = predecir_escenario(banco, escenario, instrumento=instrumento)
            tabla_resumen = tabla.copy()
            tabla_resumen["Anio"] = tabla_resumen.index.year
            resumen_anual = tabla_resumen.groupby("Anio").agg(
                tipo_medio    = ("tipo_predicho",    "mean"),
                conf_media    = ("Confidence_Score", "mean"),
                net_medio     = ("Net_Score",        "mean"),
                dir_dominante = ("direccion", lambda x: x.value_counts().index[0]),
            ).round(2)
            print(f"\n  [{escenario}]")
            print(f"  {'Ano':>4}  {'Tipo_pred(%)':>12}  {'Confidence':>10}  {'Net_Score':>9}  {'Direccion':>8}")
            for anio, row in resumen_anual.iterrows():
                print(f"  {anio:>4}  {row.tipo_medio:>12.2f}  {row.conf_media:>10.1f}"
                      f"  {row.net_medio:>9.1f}  {row.dir_dominante:>8}")

    # ── Resumen final ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  PIPELINE OIS COMPLETADO")
    print("=" * 62)
    print(f"\n  Instrumentos modelados:")
    for banco, instrumento in config.OIS_INSTRUMENT_NAMES.items():
        print(f"    {banco}: {instrumento}")
    print(f"\n  Archivos en: {config.RESULTS_DIR}")
    print(f"  Modelos en:  {config.MODEL_DIR / 'resultados'}")


if __name__ == "__main__":
    main()
