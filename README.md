# Dashboard de Predicción de Tipos de Interés — TFG

Modelo híbrido de Machine Learning para predecir los tipos de interés oficiales del BCE, BoE y FED.

## Requisitos

Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

## Cómo ejecutar el dashboard

```bash
streamlit run app.py
```

## Estructura del proyecto

```
MODELO V2/
├── config.py                  # Parámetros centralizados (rutas, umbrales, fechas)
├── app.py                     # Dashboard principal (Streamlit)
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── data_loader.py     # Carga y limpieza de DATASET.xlsx
│   │   └── feature_engineering.py  # Variables objetivo y features
│   ├── models/
│   │   ├── regression.py      # Random Forest / Ridge
│   │   ├── classification.py  # Regresión Logística Multiclase
│   │   └── scoring.py         # Confidence Score y Upward Score
│   └── visualization/
│       └── charts.py          # Gráficos Plotly reutilizables
├── escenarios/                # Plantillas de variables futuras 2026–2030
├── Datos originales/          # Archivos fuente (DATASET.xlsx, VIX, Ratings)
├── Datos procesados/          # Datasets intermedios generados por el pipeline
├── Modelo/
│   └── resultados/            # Modelos serializados y métricas
└── Resultados/                # Tablas y gráficos exportados
```

## Pipeline de ejecución

El proyecto se construye por fases independientes:

| Fase | Módulo | Descripción |
|------|--------|-------------|
| 2 | `src/data/data_loader.py` | Cargar y limpiar datos desde DATASET.xlsx |
| 3 | `src/data/feature_engineering.py` | Crear variables objetivo (y_reg, y_cls) |
| 4 | `src/data/feature_engineering.py` | División temporal train/test |
| 5 | `src/models/regression.py` | Entrenar Random Forest Regressor |
| 6 | `src/models/classification.py` | Entrenar Regresión Logística Multiclase |
| 7 | `src/models/scoring.py` | Calcular Confidence Score y Upward Score |
| 8 | `src/models/` | Predicciones futuras 2026–2030 por escenario |
| 9-10 | `src/visualization/` + `app.py` | Gráficos y dashboard Streamlit |

## División temporal

- **Train:** 1999M01 – 2023M12 (~300 meses)
- **Test:** 2024M01 – 2025M12 (~24 meses)
- **Predicción futura:** 2026 – 2030 mediante escenarios

## Parámetros configurables

Todos los parámetros del modelo se pueden modificar en `config.py`:
- `THRESHOLD_ESTABILIDAD`: umbral mensual para la clase "Estable" (por defecto 0.125 pp)
- `REGRESSOR`: selección de modelo de regresión (`"random_forest"` o `"ridge"`)
- `TRAIN_END` / `TEST_START`: fechas de corte temporal

## Advertencia

Las predicciones futuras (2026–2030) son **estimaciones basadas en escenarios hipotéticos**, no previsiones certeras. El modelo es un ejercicio académico y no debe usarse para decisiones financieras.
