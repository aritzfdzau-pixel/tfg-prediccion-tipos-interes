# Documentación del Modelo Predictivo de Tipos de Interés
## Trabajo de Fin de Grado — Aritz Fernández Aulestiarte

---

## 1. Cómo usar el modelo

### Archivo principal
El modelo está en:
```
Resultados/Modelo_Tipos_Interes_TFG.xlsx
```

### Pasos básicos
1. Abrir `Modelo_Tipos_Interes_TFG.xlsx`
2. Ir a la hoja **Panel_Control** y revisar alertas del modelo
3. Consultar **Resultados** para el resumen ejecutivo por bloque
4. Ver **Prediccion_5_Anios** para las predicciones 2026–2030
5. Consultar **Graficos** para la visualización

Para regenerar el modelo con nuevos datos o hipótesis:
```
cd MODELO/modelo
python 04_exportar_excel.py
```

---

## 2. Estructura del modelo

### Módulos Python
| Archivo | Función |
|---------|---------|
| `01_cargar_datos.py` | Carga y limpia todos los archivos fuente |
| `02_scoring_dataset.py` | Construye el dataset maestro y aplica scoring 1-10 |
| `03_regresion_prediccion.py` | Regresión Ridge + predicción 5 años |
| `04_exportar_excel.py` | Genera el Excel completo |

### Hojas del Excel
| Hoja | Contenido |
|------|-----------|
| Panel_Control | Parámetros editables y alertas |
| Datos_Anuales | Dataset anual por bloque (1993–2024) |
| Datos_Mensuales | Materias primas mensuales (1993–2026) |
| Diccionario_Variables | Definición, unidad y peso de cada variable |
| Reglas_Scoring | Tablas de intervalos para convertir a puntuación 1-10 |
| Pesos_Modelo | Ponderaciones por variable (editables) |
| Scoring_Anual | Puntuaciones históricas con código de color |
| Modelo_Predictivo | Coeficientes Ridge + predicciones históricas con acierto |
| Prediccion_5_Anios | Hipótesis + predicciones 2026–2030 para 3 escenarios |
| Resultados | Resumen ejecutivo con señal y métricas por bloque |
| Validacion | MAE, RMSE, R², acierto direccional y limitaciones |
| Graficos | 6 gráficos embebidos generados automáticamente |

---

## 3. Lógica del scoring macrofinanciero

Cada variable se transforma a una escala 1-10 con la siguiente interpretación:

| Puntuación | Significado |
|-----------|-------------|
| 1 | Situación críticamente mala |
| 3 | Situación negativa |
| 5 | Situación moderadamente débil |
| 7 | Situación buena |
| 9-10 | Situación excelente o perfecta |

### Ejemplos de reglas clave
- **Inflación**: Puntuación máxima (10) si está entre 2-3%; cae bruscamente si supera el 5% o cae en deflación.
- **Desempleo**: Penaliza tanto el paro alto (>10%) como el excesivamente bajo con presión salarial.
- **Petróleo**: Niveles estables 40-80 $/bbl reciben puntuación moderada-alta; picos >120 $/bbl se penalizan.
- **Tipos oficiales**: Niveles moderados (1-3%) reciben puntuación máxima; tipos cero o negativos, o muy altos (>7%), se penalizan.

La **puntuación agregada** se calcula como suma ponderada:
```
Score = Σ (Puntuación_variable × Peso_variable)
```

Los pesos suman siempre 100% (verificado automáticamente).

---

## 4. Cómo se calcula la predicción

El modelo combina dos componentes:

### A. Regresión Ridge con variables estandarizadas
Se estima para cada bloque (Eurozona, EEUU, UK):

```
Tipo_t = α + β₁·Inflación_t + β₂·ΔPIBpc_t + β₃·ΔPIB_t +
          β₄·Desempleo_t + β₅·OIS_t + β₆·Petróleo_t +
          β₇·Gas_t + β₈·Oro_t + β₉·Rating_t + ε
```

- Se usa **Ridge CV** (regularización L2) con α seleccionado por validación cruzada para evitar sobreajuste.
- Todas las variables se estandarizan (z-score) antes de la regresión para comparabilidad de coeficientes.
- Variables OIS sin historia (EUR €STR) se rellenan con la media histórica.

### B. Escenarios (Prediccion_5_Anios)
Para cada año 2026-2030, el usuario puede especificar hipótesis sobre cada variable. El modelo aplica los coeficientes estimados a esas hipótesis para obtener el tipo predicho.

Los tres escenarios predefinidos son:
- **Base**: evolución gradual moderada, consolidación macroeconómica.
- **Optimista**: crecimiento sólido, inflación controlada, tipos a la baja.
- **Pesimista**: rebrote inflacionario, debilidad económica, tipos con más resistencia.

---

## 5. Cómo interpretar los resultados

### Señal del modelo
La señal compara el tipo predicho en el escenario base para 2030 con el tipo actual (2024):

| Señal | Condición |
|-------|-----------|
| "Subida esperada" | Predicción > Tipo actual + 0.25 pp |
| "Bajada esperada" | Predicción < Tipo actual − 0.25 pp |
| "Estabilidad esperada" | Diferencia ≤ ±0.25 pp |

El umbral de 0.25 pp es editable en **Panel_Control**.

### Métricas de validación
- **R²**: mide cuánta varianza del tipo oficial explica el modelo (1.0 = perfecto; >0.5 = aceptable para datos macroeconómicos).
- **RMSE**: error promedio en puntos porcentuales (pp).
- **Acierto direccional**: % de veces que el modelo predice correctamente si el tipo sube, baja o se mantiene.

---

## 6. Limitaciones del modelo

1. **Variable dependiente**: El modelo predice los **tipos oficiales de banco central** (BCE, Fed, BoE), no el rendimiento del bono soberano a 10 años (variable no disponible en los archivos locales).

2. **Tamaño muestral limitado**: 26–32 observaciones anuales por bloque dificultan alcanzar el 80% de acierto direccional. UK alcanza el 95%; Eurozona y EEUU no superan el umbral con los datos disponibles.

3. **VIX/VSTOXX/VFTSE sin datos históricos**: Esta variable de riesgo financiero (peso 5%) no se incorporó al entrenamiento por falta de datos.

4. **Ratings estáticos**: Se usan los ratings actuales de 2026, no la evolución histórica.

5. **OIS incompleto para Eurozona**: EUR €STR solo disponible desde 2022; antes se usa NaN (imputado con media).

### Para mejorar el modelo
- Añadir yields soberanos 10 años (Bund, UST, UK Gilt) desde FRED o Bloomberg.
- Incorporar series mensuales de VIX/VSTOXX desde CBOE/Investing.com.
- Añadir datos de deuda pública (FMI/WEO, `General Government Debt, % of GDP`).
- Usar frecuencia mensual para mayor tamaño muestral.
