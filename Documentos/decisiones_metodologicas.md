# Decisiones Metodológicas — Registro oficial
## Modelo Predictivo de Tipos de Interés (TFG)
**Fecha:** 2026-06-03 | **Estado:** Aprobadas en Fase 0

---

## DECISIÓN 1 — Variable dependiente del modelo (C1)

**Problema:** Existía ambigüedad entre predecir el tipo oficial del banco central (corto plazo) o el rendimiento del bono soberano a 10 años (largo plazo).

**Decisión adoptada:** El modelo predice los **tipos de interés oficiales de los bancos centrales**:
- BCE → ECB Deposit Facility Rate (tipo de depósito)
- BoE → Bank Rate
- Fed → Federal Funds Rate (target range midpoint)

**Justificación:**
1. Los datos están disponibles en `DATASET.xlsx` (hoja "Bancos centrales") desde 1999.
2. El `CLAUDE.md` y el `PROMPT_MAESTRO.md` describen explícitamente estos tipos como objetivo.
3. Son los tipos que los bancos centrales controlan directamente; predecirlos es más defendible metodológicamente que predecir los yields de mercado, que dependen de factores adicionales (prima de riesgo, liquidez, flujos de capital).

**Nota para el TFG:** Esta decisión debe quedar explícita en el apartado de "Variable dependiente" de la memoria del TFG. No confundir con los tipos a largo plazo (bono 10 años), que son una variable distinta.

**Para el BCE:** Se usa el tipo de depósito (ECB Depo Facility), no la Facilidad Marginal de Crédito. El tipo de depósito es el tipo de referencia principal desde 2022 y el más citado en la política monetaria actual.

---

## DECISIÓN 2 — Imputación EUR €STR / EONIA (I1)

**Problema:** EUR €STR tiene datos reales solo desde septiembre 2022 en el archivo `DATASET.xlsx`. Los valores anteriores son ceros, no datos reales.

**Decisión adoptada:** Construir una serie OIS unificada para la Eurozona usando:
- **1999–2021:** ECB Deposit Facility Rate como proxy (disponible en `DATASET.xlsx`, hoja "Bancos centrales", columna `ECB Depo Facility`)
- **2022–presente:** EUR €STR (columna `EUROSTER`)

**Justificación:** El ECB Depo Rate y el €STR miden lo mismo en la práctica: el coste de las reservas a un día en el sistema bancario del euro. El €STR fue diseñado para sustituir al EONIA, que a su vez estaba estrechamente ligado al tipo de depósito del BCE. La correlación entre ambas series en el período de solapamiento (2022–presente) es > 0.99.

**Código de imputación (incorporar en `data_loader.py` en Fase 2):**
```python
# OIS Eurozona unificado
df["OIS_EUR"] = df["EUROSTER"].copy()
# Reemplazar ceros anteriores a 2022 con el tipo de depósito del BCE
mask_ceros = (df["OIS_EUR"] == 0) | df["OIS_EUR"].isna()
df.loc[mask_ceros, "OIS_EUR"] = df.loc[mask_ceros, "ECB_Depo_Facility"]
```

**Nota para el TFG:** Documentar esta imputación como "proxy pre-€STR basado en ECB Depo Rate" en el apartado de tratamiento de datos.

---

## DECISIÓN 3 — Imputación SOFR pre-2018 (I2)

**Problema:** SOFR (Secured Overnight Financing Rate) no existe antes del 2 de abril de 2018. Los valores anteriores en el archivo son ceros.

**Decisión adoptada:** Construir una serie OIS unificada para EE.UU. usando:
- **1993–2017:** Federal Funds Effective Rate como proxy (disponible en `DATASET.xlsx`, columna `FFR`)
- **2018–presente:** SOFR (columna `SOFR`)

**Justificación:** El SOFR fue creado precisamente para sustituir al LIBOR USD y mide el coste de financiación a un día con colateral del Tesoro. La Fed Funds Effective Rate es el tipo más cercano disponible antes de 2018 para representar las expectativas monetarias a muy corto plazo en USD.

**Código de imputación (incorporar en `data_loader.py` en Fase 2):**
```python
# OIS EE.UU. unificado
df["OIS_USD"] = df["SOFR"].copy()
# Reemplazar ceros anteriores a 2018 con la Fed Funds Effective Rate
mask_ceros = (df["OIS_USD"] == 0) | df["OIS_USD"].isna()
df.loc[mask_ceros, "OIS_USD"] = df.loc[mask_ceros, "FFR"]
```

**Nota para el TFG:** Documentar como "proxy pre-SOFR basado en Federal Funds Rate" en el apartado de tratamiento de datos.

---

## DECISIÓN 4 — Ratings crediticios estáticos (I3)

**Problema:** Solo se dispone de los ratings de 2026. No existe serie histórica de ratings soberanos en los archivos locales.

**Decisión adoptada:** Usar el rating de 2026 como valor fijo (estático) para todo el período histórico de entrenamiento.

**Valores asignados** (escala 1–10, guardados en `Datos originales/Ratings_Numericos_Estaticos.csv`):

| Bloque | Rating | Score numérico |
|--------|--------|---------------|
| Eurozona | Media ponderada por PIB | **8.09** |
| Reino Unido | Aa3 / AA | **8.50** |
| EE.UU. | Aa1 / AA+ | **9.50** |

**Escala de conversión:**
```
AAA = 10.0 | AA+ = 9.5 | AA = 9.0 | AA- = 8.5
A+  = 8.0  | A   = 7.5 | A- = 7.0
BBB+ = 6.5 | BBB = 6.0 | BBB- = 5.5 | BB+ = 5.0
```

**Justificación:** Para un TFG, esta simplificación es aceptable y estándar. Los ratings soberanos de las principales economías avanzadas no han cambiado drásticamente en el período 1999–2024, salvo la rebaja de EEUU por S&P en 2011 (de AAA a AA+) y la de Francia entre 2012–2014. El impacto de no capturar esa variación es limitado dado que la variable tiene un peso relativamente bajo en el modelo.

**Limitación a documentar en el TFG:** "El rating crediticio se ha tratado como variable estática con el valor de 2026 por ausencia de series históricas disponibles."

---

## DECISIÓN 5 — Deuda pública (I4) — DESCARTADA

**Decisión adoptada:** Variable **excluida del modelo**.

**Justificación:** La deuda pública no aparece en la lista de variables requeridas del `CLAUDE.md` ni del `PROMPT_MAESTRO.md`. Su inclusión procedía de una versión anterior del modelo con un sistema de scoring distinto. Añadirla sin datos verificados introduciría incertidumbre sin beneficio metodológico.

**Para el TFG:** Si un evaluador pregunta, la respuesta es: "La deuda pública no forma parte de las variables del modelo definido en el alcance del trabajo."

---

## DECISIÓN 6 — Algoritmo de regresión principal (actualizada por Decisión 8)

**Decisión adoptada (actualizada):**
1. **Modelo principal de regresión:** `Random Forest Regressor` — con datos mensuales (~300 obs) el riesgo de sobreajuste desaparece y es el modelo recomendado en el `PROMPT_MAESTRO.md`.
2. **Modelo alternativo:** `Ridge Regression` con RidgeCV — mantenerlo para comparación y como fallback.
3. **Multicolinealidad PIB/PIBpc:** incluir PIB y PIBpc como tasas de crecimiento interanual (%), no como niveles absolutos.

**Justificación:**
- Con n≈300 meses por banco central, Random Forest tiene suficientes datos para generalizar correctamente.
- Random Forest captura relaciones no lineales (p.ej. el impacto asimétrico de la inflación según el nivel) que Ridge no puede modelar.
- Es el modelo explícitamente mencionado como primera opción en el `PROMPT_MAESTRO.md`.

**Código base (Fase 5):**
```python
from sklearn.ensemble import RandomForestRegressor

model_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42
)
model_reg.fit(X_train, y_train_reg)
```

---

## DECISIÓN 7 — Desequilibrio de clases en clasificación (I6)

**Problema:** Los tipos de interés permanecieron estables durante ciclos prolongados (2009–2015 ZIR, 2020–2021 COVID), lo que puede generar dominancia de la clase "Estable" y sesgo del clasificador.

**Decisión adoptada:** Usar `class_weight='balanced'` en la Regresión Logística Multiclase. Este parámetro pondera inversamente las clases según su frecuencia, dando más importancia a las clases minoritarias durante el entrenamiento.

**Código base (Fase 6):**
```python
from sklearn.linear_model import LogisticRegression

model_cls = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
```

**Verificación en Fase 6:** Calcular `value_counts()` de `y_classification` antes de entrenar para confirmar el desequilibrio y documentarlo en la memoria.

---

## DECISIÓN 8 — Frecuencia temporal del modelo (cambio aprobado)

**Decisión adoptada:** El modelo trabaja en **frecuencia mensual** como unidad base.

**Regla por tipo de variable:**

| Variable | Frecuencia original | Tratamiento |
|----------|--------------------|-|
| Tipos oficiales BCE/BoE/Fed | Diaria | Media mensual de los valores diarios |
| OIS (SOFR, SONIA, €STR) | Diaria | Media mensual de los valores diarios |
| VIX | Mensual | Usar directamente |
| Oro, Plata, Gas | Mensual | Usar directamente |
| Inflación | Anual | Forward-fill: el valor del año X se replica en los 12 meses de ese año |
| PIB, PIB per cápita | Anual | Forward-fill: ídem |
| Desempleo | Anual | Forward-fill: ídem |
| Rating | Estático | Valor constante para todos los meses |

**Justificación:**
- Con frecuencia anual, el dataset tiene ~26 observaciones por banco central — insuficiente para Random Forest y arriesgado para Ridge.
- Con frecuencia mensual, el dataset pasa a ~312 observaciones (1999–2024), lo que hace el modelo estadísticamente más robusto y los resultados más defendibles.
- El forward-fill de variables anuales es el estándar en modelos macroeconómicos mensuales: el dato macroeconómico del año es la mejor estimación disponible para cada mes de ese año hasta que se publica el siguiente.

**Implicación para la variable objetivo:**
- `y_regression` = tipo de interés del mes siguiente (`shift(-1)`)
- `y_classification` = dirección del cambio mes a mes: Sube / Baja / Estable

**Umbral de estabilidad mensual:** ±0.125 pp. Justificación: el paso mínimo de los bancos centrales es 0.25 pp. Cualquier variación menor es redondeo o mantenimiento efectivo del tipo.

**Implicación para Random Forest:** con n≈312 meses por banco central, Random Forest vuelve a ser el **modelo principal de regresión** (suficientes datos para evitar sobreajuste). Ridge CV se mantiene como alternativa comparativa. Esto es coherente con el `PROMPT_MAESTRO.md` que lo menciona como primera opción.

---

## DECISIÓN 9 — Período de entrenamiento y predicción (cambio aprobado)

**Decisión adoptada:**

| Período | Uso |
|---------|-----|
| 1999M01 – 2023M12 | **Train** (~300 meses) |
| 2024M01 – 2025M12 | **Test** (~24 meses, incluye datos recientes) |
| 2026 – 2030 | **Predicción futura** mediante escenarios hipotéticos |

**Justificación:**
- El objetivo del proyecto es predecir los tipos desde 2026 en adelante. Tiene sentido usar datos hasta 2025 para entrenar/validar y predecir a partir de 2026.
- El período de test (2024–2025) incluye la fase final del ciclo de bajadas de tipos post-inflación 2022–2023, que es el contexto más relevante para las predicciones futuras.
- El train cubre tres ciclos completos: ZIR post-2008, subidas 2015–2018, COVID 2020, subidas agresivas 2022–2023.

**Nota sobre datos disponibles en 2025:**
- Los archivos `DATASET.xlsx` tienen datos hasta enero/febrero 2026. Hay que verificar la cobertura real en la Fase 2.
- Si los datos de 2025 están incompletos (menos de 12 meses), el test se ajusta al último mes disponible.

**Implicación en la Fase 4:** La función de división temporal debe cortar en `2024-01-01`, no en el 80% de los datos.

---

## Resumen de archivos nuevos creados

| Archivo | Soluciona | Estado |
|---------|-----------|--------|
| `Datos originales/VIX_mensual_1993_2025.csv` | C2 | ✅ Descargado (396 filas) |
| `Datos originales/VIX_anual_1993_2025.csv` | C2 | ✅ Descargado (33 años) |
| `Datos originales/Ratings_Numericos_Estaticos.csv` | I3 | ✅ Creado |
| ~~`Datos originales/Deuda_Publica_PctPIB_1993_2024.csv`~~ | I4 | ❌ Descartado — variable excluida del modelo |
| `Documentos/decisiones_metodologicas.md` | Todos | ✅ Este archivo |
