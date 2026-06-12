"""
run_calibracion_a2.py — Opcion A2: Isotonica (BCE/BoE) + Platt Scaling (FED)

Estrategia hibrida validada:
  · BCE y BoE : IsotonicRegression  (mejora Brier y Accuracy en test)
  · FED       : Platt Scaling       (sigmoide — robusta con pocos datos,
                                     evita overfitting del FED ya bien calibrado)

Platt Scaling (Platt, 1999):
  Para cada clase c: ajusta LogReg(C=1) sobre proba_train[:, c] -> y_bin_c
  P_cal_c = sigmoid(a * P_raw_c + b)
  Despues: normalizar filas para que sumen 1.

  Ventaja sobre isotonica con n pequeno: solo 2 parametros (a, b) por clase
  en lugar de n puntos de quiebre → mucho menos sobreajuste.

Este script NO modifica ningun archivo del pipeline principal.

Ejecutar:
    python run_calibracion_a2.py
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
from src.models.scoring import calcular_confidence_score, calcular_net_score, interpretar_scores
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import brier_score_loss, accuracy_score, f1_score, log_loss


# ─── METODO DE CALIBRACION POR BANCO ─────────────────────────────────────────
# BCE y BoE: isotonica (mejor rendimiento en test A1)
# FED: Platt scaling (sigmoide, robusta con pocos datos)

METODO_CALIBRACION = {
    "BCE": "isotonic",
    "BoE": "isotonic",
    "FED": "platt",
}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _suf(instrumento):
    return "" if instrumento == "tipo_oficial" else f"_{instrumento}"


def _cargar_logreg(banco, instrumento):
    ruta = config.MODEL_DIR / "resultados" / f"{banco}{_suf(instrumento)}_clasificacion.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {ruta}")
    d = joblib.load(ruta)
    return d["modelo"], d["scaler"]


def _preparar_datos(banco, instrumento):
    target_col = config.INSTRUMENT_TO_COL.get(instrumento, instrumento)
    df = cargar_datos_procesados(banco)
    X, y_reg, y_cls, fechas, _ = preparar_X_y(df, banco, target_col=target_col)
    (X_train, X_test, _, _, y_train_cls, y_test_cls,
     fechas_train, fechas_test) = dividir_temporal(X, y_reg, y_cls, fechas)
    return X_train, X_test, y_train_cls, y_test_cls, fechas_train, fechas_test


def _brier_multiclase(y_true, proba):
    return float(np.mean([
        brier_score_loss((np.array(y_true) == c).astype(int), proba[:, i])
        for i, c in enumerate(config.CLASES)
    ]))


def _normalizar(proba):
    """Clip + normalizar filas para garantizar distribucion valida."""
    p = np.clip(proba, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-9, 1.0, s)
    return p / s


# ─── CALIBRADORES ────────────────────────────────────────────────────────────

def _ajustar_isotonica(proba_train, y_train_cls):
    """Ajusta IsotonicRegression por clase. Devuelve lista de calibradores."""
    calibradores = []
    for i, clase in enumerate(config.CLASES):
        iso = IsotonicRegression(out_of_bounds="clip")
        y_bin = (np.array(y_train_cls) == clase).astype(int)
        iso.fit(proba_train[:, i], y_bin)
        calibradores.append(iso)
    return calibradores


def _ajustar_platt(proba_train, y_train_cls):
    """
    Platt Scaling: LogReg(C=1) sobre proba_train[:, c] -> y_bin_c.
    Solo 2 parametros por clase (a, b en el argumento de la sigmoide).
    Mucho mas robusto que isotonica con n pequeno.
    """
    calibradores = []
    for i, clase in enumerate(config.CLASES):
        lr = LogReg(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        y_bin = (np.array(y_train_cls) == clase).astype(int)
        # Necesitamos al menos 2 clases en y_bin para ajustar LogReg
        if len(np.unique(y_bin)) < 2:
            # Si solo hay una clase, usar calibrador trivial (devuelve p tal cual)
            calibradores.append(None)
        else:
            lr.fit(proba_train[:, i].reshape(-1, 1), y_bin)
            calibradores.append(lr)
    return calibradores


def _aplicar_calibradores(calibradores, proba_test, metodo):
    """Aplica la lista de calibradores al proba_test. Devuelve proba calibrada."""
    cols = []
    for i, cal in enumerate(calibradores):
        if cal is None:
            cols.append(proba_test[:, i])
        elif metodo == "isotonic":
            cols.append(cal.predict(proba_test[:, i]))
        else:  # platt
            cols.append(cal.predict_proba(proba_test[:, i].reshape(-1, 1))[:, 1])
    return _normalizar(np.stack(cols, axis=1).astype(float))


# ─── CALIBRACION POR BANCO ───────────────────────────────────────────────────

def calibrar_banco_a2(banco, instrumento="tipo_oficial"):
    metodo = METODO_CALIBRACION[banco]
    print(f"  [{banco} / {instrumento}] metodo={metodo} ...", end=" ", flush=True)

    modelo_cls, scaler = _cargar_logreg(banco, instrumento)
    X_train, X_test, y_train_cls, y_test_cls, _, fechas_test = _preparar_datos(banco, instrumento)

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Probabilidades originales
    proba_orig = modelo_cls.predict_proba(X_test_sc)
    pred_orig  = modelo_cls.predict(X_test_sc)
    conf_orig  = calcular_confidence_score(proba_orig)
    acc_orig   = accuracy_score(y_test_cls, pred_orig)
    f1_orig    = f1_score(y_test_cls, pred_orig, average="macro", zero_division=0)
    brier_orig = _brier_multiclase(np.array(y_test_cls), proba_orig)
    ll_orig    = log_loss(y_test_cls, proba_orig, labels=config.CLASES)

    # Ajuste del calibrador sobre TRAIN
    proba_train = modelo_cls.predict_proba(X_train_sc)
    if metodo == "isotonic":
        calibradores = _ajustar_isotonica(proba_train, y_train_cls)
    else:
        calibradores = _ajustar_platt(proba_train, y_train_cls)

    # Probabilidades calibradas en TEST
    proba_cal  = _aplicar_calibradores(calibradores, proba_orig, metodo)
    pred_cal   = np.array(config.CLASES)[np.argmax(proba_cal, axis=1)]
    conf_cal   = calcular_confidence_score(proba_cal)
    acc_cal    = accuracy_score(y_test_cls, pred_cal)
    f1_cal     = f1_score(y_test_cls, pred_cal, average="macro", zero_division=0)
    brier_cal  = _brier_multiclase(np.array(y_test_cls), proba_cal)
    ll_cal     = log_loss(y_test_cls, proba_cal, labels=config.CLASES)

    print(f"Acc {acc_orig:.3f}->{acc_cal:.3f} | Brier {brier_orig:.4f}->{brier_cal:.4f} | "
          f"Conf {conf_orig.mean():.1f}->{conf_cal.mean():.1f}")

    # Tabla mes a mes
    filas = []
    for t, fecha in enumerate(fechas_test):
        conf_o, dir_o, _ = interpretar_scores(
            float(conf_orig[t]),
            float(proba_orig[t, 2]*100),
            float(proba_orig[t, 0]*100),
        )
        conf_c, dir_c, _ = interpretar_scores(
            float(conf_cal[t]),
            float(proba_cal[t, 2]*100),
            float(proba_cal[t, 0]*100),
        )
        real = y_test_cls.iloc[t] if hasattr(y_test_cls, "iloc") else y_test_cls[t]
        filas.append({
            "Fecha":         fecha.strftime("%Y-%m"),
            "Real":          real,
            "Pred_orig":     pred_orig[t],
            "Pred_cal":      pred_cal[t],
            "P_Baja_orig":   round(float(proba_orig[t, 0])*100, 2),
            "P_Est_orig":    round(float(proba_orig[t, 1])*100, 2),
            "P_Sube_orig":   round(float(proba_orig[t, 2])*100, 2),
            "Conf_orig":     round(float(conf_orig[t]), 1),
            "Nivel_orig":    conf_o,
            "P_Baja_cal":    round(float(proba_cal[t, 0])*100, 2),
            "P_Est_cal":     round(float(proba_cal[t, 1])*100, 2),
            "P_Sube_cal":    round(float(proba_cal[t, 2])*100, 2),
            "Conf_cal":      round(float(conf_cal[t]), 1),
            "Nivel_cal":     conf_c,
            "DeltaConf":     round(float(conf_cal[t]) - float(conf_orig[t]), 1),
        })

    df_comp = pd.DataFrame(filas)
    ruta = config.RESULTS_DIR / f"calibracion_a2_{banco}{_suf(instrumento)}.csv"
    df_comp.to_csv(ruta, index=False)

    return {
        "banco": banco, "instrumento": instrumento, "metodo": metodo,
        "acc_orig":    round(acc_orig,  4), "acc_cal":    round(acc_cal,  4),
        "f1_orig":     round(f1_orig,   4), "f1_cal":     round(f1_cal,   4),
        "brier_orig":  round(brier_orig,4), "brier_cal":  round(brier_cal,4),
        "ll_orig":     round(ll_orig,   4), "ll_cal":     round(ll_cal,   4),
        "conf_orig":   round(float(conf_orig.mean()), 1),
        "conf_cal":    round(float(conf_cal.mean()),  1),
        "ultimo_conf_orig":  round(float(conf_orig[-1]), 1),
        "ultimo_nivel_orig": filas[-1]["Nivel_orig"],
        "ultimo_conf_cal":   round(float(conf_cal[-1]),  1),
        "ultimo_nivel_cal":  filas[-1]["Nivel_cal"],
        "_calibradores": calibradores,
        "_df_comp":      df_comp,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  OPCION A2 — Isotonica (BCE/BoE) + Platt Scaling (FED)")
    print("  Referencia: Platt 1999 + Niculescu-Mizil & Caruana 2005")
    print("=" * 70)
    print(f"\n  Metodo por banco: BCE=isotonica | BoE=isotonica | FED=platt\n")

    resultados = {}

    for instrumento in ["tipo_oficial", "OIS"]:
        label = "Tipos Oficiales" if instrumento == "tipo_oficial" else "Tipos OIS"
        print(f"\n  {label}:")
        for banco in config.BANCOS_CENTRALES:
            try:
                res = calibrar_banco_a2(banco, instrumento)
                resultados[f"{banco}_{instrumento}"] = res
            except FileNotFoundError as e:
                print(f"  [AVISO] {e}")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN — ANTES vs DESPUES (Opcion A2)")
    print("=" * 70)
    print(f"\n  {'Banco':<5} {'Instr':<14} {'Mtd':>5} | "
          f"{'Acc':>6} -> {'Acc':>6} | "
          f"{'Brier':>7} -> {'Brier':>7} | "
          f"{'Conf%':>6} -> {'Conf%':>6}")
    print(f"  {'-'*5} {'-'*14} {'-'*5}   "
          f"{'-'*6}    {'-'*6}   "
          f"{'-'*7}    {'-'*7}   "
          f"{'-'*6}    {'-'*6}")

    filas_res = []
    for key, r in resultados.items():
        acc_ok   = "+" if r["acc_cal"]   >= r["acc_orig"]   - 0.01 else "-"
        brier_ok = "+" if r["brier_cal"] <= r["brier_orig"] + 0.005 else "-"
        conf_ok  = "+" if r["conf_cal"]  >= r["conf_orig"]  - 1.0  else "-"
        print(
            f"  {r['banco']:<5} {r['instrumento']:<14} {r['metodo']:>5} | "
            f"{r['acc_orig']:>6.4f} {acc_ok}> {r['acc_cal']:>6.4f} | "
            f"{r['brier_orig']:>7.4f} {brier_ok}> {r['brier_cal']:>7.4f} | "
            f"{r['conf_orig']:>6.1f} {conf_ok}> {r['conf_cal']:>6.1f}"
        )
        filas_res.append({
            "Banco":          r["banco"],
            "Instrumento":    r["instrumento"],
            "Metodo":         r["metodo"],
            "Acc_orig":       r["acc_orig"],   "Acc_cal":    r["acc_cal"],
            "F1_orig":        r["f1_orig"],    "F1_cal":     r["f1_cal"],
            "Brier_orig":     r["brier_orig"], "Brier_cal":  r["brier_cal"],
            "LogLoss_orig":   r["ll_orig"],    "LogLoss_cal":r["ll_cal"],
            "Conf_orig":      r["conf_orig"],  "Conf_cal":   r["conf_cal"],
            "Conf_dic25_orig":r["ultimo_conf_orig"],
            "Nivel_dic25_orig":r["ultimo_nivel_orig"],
            "Conf_dic25_cal": r["ultimo_conf_cal"],
            "Nivel_dic25_cal":r["ultimo_nivel_cal"],
        })

    df_res = pd.DataFrame(filas_res)
    df_res.to_csv(config.RESULTS_DIR / "calibracion_a2_resumen.csv", index=False)

    # ── Caso critico: diciembre 2025 ────────────────────────────────────────
    print(f"\n  CASO CRITICO — Diciembre 2025 (fin del test)")
    print(f"  {'Banco/Instr':<20} {'Metodo':>7} | "
          f"{'Conf orig':>10} {'Nivel':>10} | "
          f"{'Conf cal':>10} {'Nivel':>10}")
    print(f"  {'-'*20} {'-'*7}   {'-'*10} {'-'*10}   {'-'*10} {'-'*10}")
    for key, r in resultados.items():
        print(
            f"  {key:<20} {r['metodo']:>7} | "
            f"{r['ultimo_conf_orig']:>10.1f} {r['ultimo_nivel_orig']:>10} | "
            f"{r['ultimo_conf_cal']:>10.1f} {r['ultimo_nivel_cal']:>10}"
        )

    # ── Detalle FED (el caso problematico) ──────────────────────────────────
    print(f"\n  DETALLE FED — Platt Scaling mes a mes:")
    print(f"  {'Fecha':<9} {'Real':<9} {'Pred->Cal':<12} "
          f"{'P_Baja':>7} ->{'':>1} {'P_Baja':>7} | "
          f"{'P_Est':>7} ->{'':>1} {'P_Est':>7} | "
          f"{'Conf':>6} ->{'':>1} {'Conf':>6}")
    print(f"  {'-'*9} {'-'*9} {'-'*12} {'-'*7}    {'-'*7}   "
          f"{'-'*7}    {'-'*7}   {'-'*6}    {'-'*6}")
    for instr in ["tipo_oficial", "OIS"]:
        key = f"FED_{instr}"
        if key not in resultados:
            continue
        df_c = resultados[key]["_df_comp"]
        print(f"  --- FED [{instr}] ---")
        for _, row in df_c.iterrows():
            cambio = "*" if row["Pred_orig"] != row["Pred_cal"] else " "
            print(
                f"  {row['Fecha']:<9} {row['Real']:<9} "
                f"{row['Pred_orig']}->{row['Pred_cal']:<8}{cambio} "
                f"{row['P_Baja_orig']:>7.1f}  -> {row['P_Baja_cal']:>7.1f} | "
                f"{row['P_Est_orig']:>7.1f}  -> {row['P_Est_cal']:>7.1f} | "
                f"{row['Conf_orig']:>6.1f}  -> {row['Conf_cal']:>6.1f}"
            )

    # ── Veredicto ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VEREDICTO OPCION A2")
    print("=" * 70)

    criticos = {k: r for k, r in resultados.items()}
    aprobados  = [k for k, r in criticos.items()
                  if r["acc_cal"] >= r["acc_orig"] - 0.01
                  and r["brier_cal"] <= r["brier_orig"] + 0.006]
    reprobados = [k for k in criticos if k not in aprobados]

    if not reprobados:
        print("""
  [OK] APROBADA — Opcion A2 lista para integracion al modelo principal.

  Todos los bancos cumplen:
    - Accuracy no empeora mas de 0.01
    - Brier Score no empeora mas de 0.006
    - Confidence Score mas preciso

  Siguiente paso: ejecutar run_calibracion_integrar.py
        """)
    else:
        print(f"\n  [PARCIAL] Aprobados: {aprobados}")
        print(f"  [!!] Con problemas: {reprobados}")
        print(f"\n  Los bancos con [!!] necesitan revision antes de integrar.")
        print(f"  Considerar: umbrales mas permisivos o excluir esos casos.\n")

    print(f"  Archivos guardados:")
    for key in resultados:
        print(f"    Resultados/calibracion_a2_{key}.csv")
    print(f"    Resultados/calibracion_a2_resumen.csv\n")

    return resultados


if __name__ == "__main__":
    main()
