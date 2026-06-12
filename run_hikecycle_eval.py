"""
run_hikecycle_eval.py — Evaluación adicional del clasificador en el ciclo de subidas 2022-2023

El test principal del modelo (2024-2025) no contiene observaciones de la clase
"Sube" porque ese período fue un ciclo exclusivo de bajadas de tipos. Esto impide
validar si el clasificador es capaz de detectar subidas.

Este script realiza una evaluación paralela con una división temporal diferente:
  · Train: 2000-2021  (misma arquitectura LogReg + class_weight=balanced)
  · Test:  2022-2023  (24 meses — ciclo de subidas agresivas post-COVID)

2022 fue el año de mayor agresividad alcista de los últimos 20 años:
  · BCE:  0.00% → 2.50% (7 subidas)
  · BoE:  0.25% → 3.50% (8 subidas)
  · FED:  0.25% → 4.50% (7 subidas)

El test 2022-2023 contiene las 3 clases (Baja, Estable, Sube), permitiendo
calcular el F1_macro real con soporte en todas las clases.

Nota: este script NO modifica los modelos de Modelo/resultados/.
Los resultados se guardan con sufijo '_hikecycle' en Resultados/.

Ejecutar:
    python run_hikecycle_eval.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.models.classification import ejecutar_evaluacion_hikecycle, guardar_resumen_hikecycle


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  EVALUACION ADICIONAL — CLASIFICADOR EN CICLO DE SUBIDAS 2022-2023")
    print("  Objetivo: validar la clase 'Sube' (ausente en el test 2024-2025)")
    print("=" * 70)
    print("\n  Division temporal: Train=2000-2021 | Test=2022-2023")
    print("  Arquitectura: Logistic Regression, class_weight=balanced, C=1.0\n")

    todos = []
    for banco in config.BANCOS_CENTRALES:
        res = ejecutar_evaluacion_hikecycle(banco, instrumento="tipo_oficial")
        todos.append(res)

    df_resumen = guardar_resumen_hikecycle(todos, instrumento="tipo_oficial")

    print("\n" + "=" * 70)
    print("  RESUMEN GLOBAL — CICLO SUBIDAS 2022-2023")
    print("=" * 70)
    print("\n  Train: 2000-2021 | Test: 2022-2023")
    print("  A diferencia del test principal, aquí la clase 'Sube' tiene soporte.\n")

    cols_show = ["Banco", "N_Baja", "N_Estable", "N_Sube",
                 "Accuracy", "F1_macro", "ROC_AUC_ovr"]
    print(df_resumen[cols_show].to_string(index=False))

    print("\n" + "=" * 70)
    print("  INTERPRETACION")
    print("=" * 70)

    for _, row in df_resumen.iterrows():
        banco = row["Banco"]
        n_sube = int(row["N_Sube"])
        f1 = float(row["F1_macro"])
        roc = row["ROC_AUC_ovr"]
        roc_str = f"{roc:.3f}" if str(roc) != "nan" else "N/A"

        if n_sube > 0:
            estado = "OK" if f1 > 0.40 else "~"
            print(f"\n  [{estado}] {banco}: {n_sube} meses 'Sube' en test | "
                  f"F1_macro={f1:.3f} | ROC_AUC={roc_str}")
        else:
            print(f"\n  [KO] {banco}: sin meses 'Sube' incluso en 2022-2023 (revisar datos)")

    print("\n  Archivos generados:")
    print("    Resultados/resumen_clasificacion_hikecycle.csv")
    for banco in config.BANCOS_CENTRALES:
        print(f"    Resultados/confusion_test_{banco}_hikecycle.csv")
        print(f"    Resultados/reporte_clasificacion_{banco}_hikecycle.txt")
    print()
