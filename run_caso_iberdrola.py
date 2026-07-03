# -*- coding: utf-8 -*-
"""
Caso práctico Iberdrola: comparación de 4 estrategias de refinanciación
usando las predicciones del modelo (tipos oficiales + OIS).

Toda la lógica de negocio vive en src/models/caso_iberdrola.py (módulo
compartido con la tab "Caso Iberdrola" del dashboard). Este script solo
ejecuta el caso con los supuestos por defecto y guarda los CSVs.

Salidas: Resultados/caso_iberdrola_pesos.csv,
         Resultados/caso_iberdrola_costes.csv,
         Resultados/caso_iberdrola_giros_modelo.csv
"""

import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from src.models.caso_iberdrola import cargar_predicciones, calcular_caso

RES = DIR / "Resultados"


def main() -> None:
    datos = cargar_predicciones(RES)
    res = calcular_caso(datos)

    res["df_pesos"].to_csv(RES / "caso_iberdrola_pesos.csv", index=False)
    res["df_costes"].to_csv(RES / "caso_iberdrola_costes.csv", index=False)
    res["df_giros"].to_csv(RES / "caso_iberdrola_giros_modelo.csv", index=False)

    print("PESOS DE ESCENARIOS (derivados del clasificador)")
    print(res["df_pesos"].to_string(index=False))
    print()
    print("COSTE FINANCIERO TOTAL 2026-2030 POR ESTRATEGIA (M€)")
    print(res["df_costes"].to_string(index=False))
    print()
    print(f">> Estrategia óptima por coste esperado: {res['mejor']}")
    print()
    print("AÑO DE GIRO A FIJO DE LA ESTRATEGIA GUIADA POR EL MODELO")
    print(res["df_giros"].to_string(index=False))


if __name__ == "__main__":
    main()
