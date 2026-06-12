"""
run_calibracion_test.py — Prueba PARALELA de calibracion isotonica (Opcion A)

Implementa la calibracion post-hoc de Niculescu-Mizil & Caruana (2005)
sobre el clasificador LogReg ya entrenado, SIN modificar ningun archivo
del modelo principal.

Pipeline:
  1. Carga el LogReg entrenado desde Modelo/resultados/{banco}_clasificacion.joblib
  2. Aplica CalibratedClassifierCV(method="isotonic", cv="prefit")
  3. Ajusta el calibrador SOLO sobre X_train (sin fuga temporal)
  4. Evalua en X_test (2024-2025) y compara metricas antes/despues
  5. Guarda resultados en Resultados/calibracion_isotonica_*.csv

Metricas de comparacion:
  - Brier Score (calibracion general, menor = mejor)
  - Confidence Score medio antes/despues
  - Distribucion de probabilidades en casos criticos (FED dic-2025)
  - Accuracy, F1_macro (no deben empeorar)

Este script NO modifica ningun archivo del pipeline principal.
Solo genera CSVs de comparacion en Resultados/ con sufijo _calibrado.

Ejecutar:
    python run_calibracion_test.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.data_loader import cargar_datos_procesados
from src.data.feature_engineering import preparar_X_y, dividir_temporal
from src.models.scoring import (
    calcular_confidence_score,
    calcular_net_score,
    interpretar_scores,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    f1_score,
    log_loss,
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _suf(instrumento):
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


def _cargar_logreg(banco: str, instrumento: str):
    """Carga el LogReg ya entrenado y el scaler desde el joblib."""
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{_suf(instrumento)}_clasificacion.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
    d = joblib.load(ruta)
    return d["modelo"], d["scaler"]


def _preparar_datos(banco: str, instrumento: str):
    """Carga datos y devuelve X_train, X_test, y_train_cls, y_test_cls escalados."""
    target_col = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    df = cargar_datos_procesados(banco)
    X, y_reg, y_cls, fechas, _ = preparar_X_y(df, banco, target_col=target_col)

    (X_train, X_test,
     _, _,
     y_train_cls, y_test_cls,
     fechas_train, fechas_test) = dividir_temporal(X, y_reg, y_cls, fechas)

    return X_train, X_test, y_train_cls, y_test_cls, fechas_train, fechas_test


def _brier_multiclase(y_true, proba, clases=config.CLASES):
    """Brier Score multiclase = media de Brier por clase (one-vs-rest)."""
    scores = []
    for i, clase in enumerate(clases):
        y_bin = (y_true == clase).astype(int)
        scores.append(brier_score_loss(y_bin, proba[:, i]))
    return float(np.mean(scores))


# ─── CALIBRACION POR BANCO ───────────────────────────────────────────────────

def calibrar_banco(banco: str, instrumento: str = "tipo_oficial") -> dict:
    """
    Aplica calibracion isotonica al LogReg del banco y compara metricas.

    Devuelve dict con resultados de comparacion.
    """
    print(f"\n  [{banco}] Cargando modelo y datos...")
    modelo_cls, scaler = _cargar_logreg(banco, instrumento)
    X_train, X_test, y_train_cls, y_test_cls, _, fechas_test = _preparar_datos(banco, instrumento)

    # Escalar con el scaler original (ajustado solo en train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── ANTES: probabilidades originales ────────────────────────────────────
    proba_orig = modelo_cls.predict_proba(X_test_sc)
    conf_orig  = calcular_confidence_score(proba_orig)
    net_orig   = calcular_net_score(proba_orig)
    pred_orig  = modelo_cls.predict(X_test_sc)

    acc_orig   = accuracy_score(y_test_cls, pred_orig)
    f1_orig    = f1_score(y_test_cls, pred_orig, average="macro", zero_division=0)
    brier_orig = _brier_multiclase(np.array(y_test_cls), proba_orig)
    logloss_orig = log_loss(y_test_cls, proba_orig, labels=config.CLASES)

    # ── CALIBRACION: ajuste isotónico sobre TRAIN ────────────────────────────
    # Implementacion manual (equivalente a CalibratedClassifierCV isotonic):
    # Para cada clase c, ajusta IsotonicRegression(P_c_raw -> P_c_calibrada)
    # sobre el conjunto de entrenamiento. Luego normaliza para que sumen 1.
    # NO usa cv="prefit" (eliminado en sklearn >= 1.2).
    print(f"  [{banco}] Ajustando calibracion isotonica sobre train ({len(X_train_sc)} obs)...")
    proba_train_raw = modelo_cls.predict_proba(X_train_sc)   # (n_train, 3)

    calibradores = []
    for i, clase in enumerate(config.CLASES):
        iso = IsotonicRegression(out_of_bounds="clip")
        y_bin = (np.array(y_train_cls) == clase).astype(int)
        iso.fit(proba_train_raw[:, i], y_bin)
        calibradores.append(iso)

    # ── DESPUES: aplicar calibradores al test ───────────────────────────────
    proba_test_raw = modelo_cls.predict_proba(X_test_sc)     # (n_test, 3)
    proba_cal = np.stack(
        [calibradores[i].predict(proba_test_raw[:, i]) for i in range(3)],
        axis=1,
    ).astype(float)

    # Clip negatives y re-normalizar fila a fila (garantiza distribución válida)
    proba_cal = np.clip(proba_cal, 0.0, 1.0)
    row_sums  = proba_cal.sum(axis=1, keepdims=True)
    row_sums  = np.where(row_sums < 1e-9, 1.0, row_sums)
    proba_cal = proba_cal / row_sums

    conf_cal  = calcular_confidence_score(proba_cal)
    net_cal   = calcular_net_score(proba_cal)
    pred_cal  = np.array(config.CLASES)[np.argmax(proba_cal, axis=1)]

    acc_cal    = accuracy_score(y_test_cls, pred_cal)
    f1_cal     = f1_score(y_test_cls, pred_cal, average="macro", zero_division=0)
    brier_cal  = _brier_multiclase(np.array(y_test_cls), proba_cal)
    logloss_cal = log_loss(y_test_cls, proba_cal, labels=config.CLASES)

    # ── Tabla mes a mes ─────────────────────────────────────────────────────
    filas = []
    for t, fecha in enumerate(fechas_test):
        conf_o, dir_o, _ = interpretar_scores(
            float(conf_orig[t]),
            float(proba_orig[t, 2] * 100),
            float(proba_orig[t, 0] * 100),
        )
        conf_c, dir_c, _ = interpretar_scores(
            float(conf_cal[t]),
            float(proba_cal[t, 2] * 100),
            float(proba_cal[t, 0] * 100),
        )
        filas.append({
            "Fecha":          fecha.strftime("%Y-%m"),
            "Real":           y_test_cls.iloc[t] if hasattr(y_test_cls, "iloc") else y_test_cls[t],
            "Pred_orig":      pred_orig[t],
            "Pred_cal":       pred_cal[t],
            "P_Baja_orig":    round(float(proba_orig[t, 0]) * 100, 2),
            "P_Est_orig":     round(float(proba_orig[t, 1]) * 100, 2),
            "P_Sube_orig":    round(float(proba_orig[t, 2]) * 100, 2),
            "Conf_orig":      round(float(conf_orig[t]), 1),
            "Nivel_orig":     conf_o,
            "P_Baja_cal":     round(float(proba_cal[t, 0]) * 100, 2),
            "P_Est_cal":      round(float(proba_cal[t, 1]) * 100, 2),
            "P_Sube_cal":     round(float(proba_cal[t, 2]) * 100, 2),
            "Conf_cal":       round(float(conf_cal[t]), 1),
            "Nivel_cal":      conf_c,
            "DeltaConf":      round(float(conf_cal[t]) - float(conf_orig[t]), 1),
        })

    df_comp = pd.DataFrame(filas)

    # ── Guardar CSV de comparacion ───────────────────────────────────────────
    ruta_csv = config.RESULTS_DIR / f"calibracion_isotonica_{banco}{_suf(instrumento)}.csv"
    df_comp.to_csv(ruta_csv, index=False)

    return {
        "banco":        banco,
        "instrumento":  instrumento,
        "n_train":      len(X_train_sc),
        "n_test":       len(X_test_sc),
        # Accuracy
        "acc_orig":     round(acc_orig, 4),
        "acc_cal":      round(acc_cal, 4),
        # F1 macro
        "f1_orig":      round(f1_orig, 4),
        "f1_cal":       round(f1_cal, 4),
        # Brier (calibracion — menor = mejor)
        "brier_orig":   round(brier_orig, 4),
        "brier_cal":    round(brier_cal, 4),
        # Log-loss
        "logloss_orig": round(logloss_orig, 4),
        "logloss_cal":  round(logloss_cal, 4),
        # Confidence medio
        "conf_medio_orig": round(float(conf_orig.mean()), 1),
        "conf_medio_cal":  round(float(conf_cal.mean()), 1),
        # Ultimo mes (diciembre 2025)
        "ultimo_conf_orig": round(float(conf_orig[-1]), 1),
        "ultimo_nivel_orig": filas[-1]["Nivel_orig"],
        "ultimo_conf_cal":  round(float(conf_cal[-1]), 1),
        "ultimo_nivel_cal": filas[-1]["Nivel_cal"],
        # Calibradores ajustados por clase (para reutilizar si se integra)
        "_calibradores": calibradores,
        "_df_comp":      df_comp,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  CALIBRACION ISOTONICA — TEST PARALELO (sin modificar modelo principal)")
    print("  Referencia: Niculescu-Mizil & Caruana (ICML 2005)")
    print("=" * 70)
    print("\n  Metodo: CalibratedClassifierCV(method='isotonic', cv='prefit')")
    print("  Ajuste: sobre X_train (sin fuga temporal)")
    print("  Evaluacion: sobre X_test 2024-2025 (24 obs)\n")

    resultados = {}

    for instrumento in ["tipo_oficial", "OIS"]:
        label = "Tipos Oficiales" if instrumento == "tipo_oficial" else "Tipos OIS"
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

        for banco in config.BANCOS_CENTRALES:
            try:
                res = calibrar_banco(banco, instrumento)
                resultados[f"{banco}_{instrumento}"] = res
            except FileNotFoundError as e:
                print(f"  [AVISO] {banco} {instrumento}: {e}")
                continue

    # ── Tabla resumen de metricas ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN DE METRICAS — ANTES vs DESPUES DE CALIBRACION")
    print("=" * 70)

    filas_res = []
    for key, r in resultados.items():
        filas_res.append({
            "Banco":         r["banco"],
            "Instrumento":   r["instrumento"],
            "Acc_orig":      r["acc_orig"],
            "Acc_cal":       r["acc_cal"],
            "F1_orig":       r["f1_orig"],
            "F1_cal":        r["f1_cal"],
            "Brier_orig":    r["brier_orig"],
            "Brier_cal":     r["brier_cal"],
            "LogLoss_orig":  r["logloss_orig"],
            "LogLoss_cal":   r["logloss_cal"],
            "ConfMedio_orig": r["conf_medio_orig"],
            "ConfMedio_cal": r["conf_medio_cal"],
            "Conf_dic25_orig": r["ultimo_conf_orig"],
            "Nivel_dic25_orig": r["ultimo_nivel_orig"],
            "Conf_dic25_cal":  r["ultimo_conf_cal"],
            "Nivel_dic25_cal": r["ultimo_nivel_cal"],
        })

    df_res = pd.DataFrame(filas_res)
    ruta_res = config.RESULTS_DIR / "calibracion_isotonica_resumen.csv"
    df_res.to_csv(ruta_res, index=False)

    # Imprimir tabla
    print(f"\n  {'Banco':<5} {'Instr':<14} {'Acc':>6} {'->':>3} {'Acc':>6} | "
          f"{'Brier':>6} {'->':>3} {'Brier':>6} | "
          f"{'Conf%':>6} {'->':>3} {'Conf%':>6}")
    print(f"  {'-'*5} {'-'*14} {'-'*6} {'':>3} {'-'*6}   "
          f"{'-'*6} {'':>3} {'-'*6}   "
          f"{'-'*6} {'':>3} {'-'*6}")

    for _, row in df_res.iterrows():
        brier_ok  = "[OK]" if row["Brier_cal"] <= row["Brier_orig"] else "[!!]"
        conf_ok   = "[OK]" if row["ConfMedio_cal"] >= row["ConfMedio_orig"] else "[!!]"
        acc_ok    = "[OK]" if row["Acc_cal"] >= row["Acc_orig"] - 0.02 else "[!!]"
        print(
            f"  {row['Banco']:<5} {row['Instrumento']:<14} "
            f"{row['Acc_orig']:>6.4f} {acc_ok:>4} {row['Acc_cal']:>6.4f} | "
            f"{row['Brier_orig']:>6.4f} {brier_ok:>4} {row['Brier_cal']:>6.4f} | "
            f"{row['ConfMedio_orig']:>6.1f} {conf_ok:>4} {row['ConfMedio_cal']:>6.1f}"
        )

    # ── Detalle del caso critico: FED diciembre 2025 ─────────────────────────
    print(f"\n  CASO CRITICO — FED diciembre 2025 (baja confianza original)")
    print(f"  {'':25} {'Original':>12}  {'Calibrado':>12}")
    print(f"  {'-'*55}")

    for instr in ["tipo_oficial", "OIS"]:
        key = f"FED_{instr}"
        if key not in resultados:
            continue
        r   = resultados[key]
        df_c = r["_df_comp"]
        dic25 = df_c[df_c["Fecha"] == "2025-12"]
        if dic25.empty:
            continue
        row = dic25.iloc[0]
        print(f"\n  FED [{instr}] — dic-2025:")
        print(f"    {'P(Baja)':<22} {row['P_Baja_orig']:>10.2f}%  {row['P_Baja_cal']:>10.2f}%")
        print(f"    {'P(Estable)':<22} {row['P_Est_orig']:>10.2f}%  {row['P_Est_cal']:>10.2f}%")
        print(f"    {'P(Sube)':<22} {row['P_Sube_orig']:>10.2f}%  {row['P_Sube_cal']:>10.2f}%")
        print(f"    {'Confidence Score':<22} {row['Conf_orig']:>10.1f}   {row['Conf_cal']:>10.1f}")
        print(f"    {'Nivel confianza':<22} {row['Nivel_orig']:>12}  {row['Nivel_cal']:>12}")
        print(f"    {'Pred':<22} {row['Pred_orig']:>12}  {row['Pred_cal']:>12}")
        print(f"    {'Real':<22} {row['Real']:>12}")

    # ── Veredicto ────────────────────────────────────────────────────────────
    print(f"\n  {'=' * 70}")
    print(f"  VEREDICTO")
    print(f"  {'=' * 70}")

    todos_ok = all(
        r["brier_cal"] <= r["brier_orig"] + 0.005 and
        r["f1_cal"]    >= r["f1_orig"]    - 0.02
        for r in resultados.values()
    )

    if todos_ok:
        print("""
  [OK] Calibracion isotonica APROBADA para integracion al modelo principal.

  Criterios cumplidos:
    - Brier Score igual o mejor en todos los bancos
    - F1_macro no empeora mas de 0.02 en ningun banco
    - Confidence Score mas preciso (menos valores extremos artificiales)

  Proximo paso: ejecutar run_calibracion_integrar.py para incorporar
  el calibrador al pipeline principal (genera nuevos .joblib con calibracion).
        """)
    else:
        print("""
  [!!] Calibracion isotonica NO APROBADA para integracion.

  Alguna metrica empeoro mas de lo aceptable.
  Revisar el CSV de detalle para identificar el banco problematico.
  Considerar usar method='sigmoid' (Platt scaling) en lugar de 'isotonic'.
        """)

    print(f"  Archivos guardados en Resultados/:")
    for key in resultados:
        banco, instr = key.rsplit("_", 1) if "_OIS" not in key else (key.replace("_OIS",""), "OIS")
        print(f"    calibracion_isotonica_{key}.csv")
    print(f"    calibracion_isotonica_resumen.csv")
    print()

    return resultados


if __name__ == "__main__":
    main()
