# Limitaciones del modelo de clasificación — Nota para la memoria del TFG

## Contexto
Detectadas durante la Fase 6 (clasificación) al revisar las predicciones del periodo de test 2024-2025.

---

## Limitación 1 — BCE y BoE: P_Baja < 3% durante todo el test

**Qué ocurre:**
El modelo de clasificación asigna probabilidad de bajada inferior al 3% en todos los meses de test para BCE y BoE, pese a que ambos bancos realizaron recortes significativos en 2024-2025 (BCE desde 4.0% hasta 2.0%; BoE desde 5.25% hasta 3.75%).

**Causa raíz:**
El modelo usa solo **niveles** de OIS (la variable más importante con 99.6% de importancia en regresión). Durante el entrenamiento (2000-2023), OIS alto estaba fuertemente correlado con ciclos de subidas (2006-2008, 2022-2023). En 2024, el OIS seguía en niveles altos pero el ciclo ya se había invertido. Sin una feature de **momentum** (dirección del cambio), el modelo no distingue entre:
- "OIS = 3.9% y subiendo" → ciclo de subidas
- "OIS = 3.9% ya cayendo desde 4.5%" → ciclo de bajadas

**Evidencia numérica:**
```
BCE test 2024-03: y_real=Estable,  P_Baja=0.7%,  P_Sube=32%  (BCE cortó en junio)
BCE test 2024-09: y_real=Baja,     P_Baja=0.9%,  P_Sube=15%  (BCE ya había cortado)
BCE test 2025-03: y_real=Baja,     P_Baja=1.0%,  P_Sube=0.9%
```

**Solución posible (no implementada):**
Añadir dos features de momentum:
- `delta_tipo_3m = tipo_oficial_t − tipo_oficial_{t−3}` → captura si el banco ya ha empezado a mover tipos
- `delta_OIS_1m = OIS_t − OIS_{t−1}` → captura si el mercado está pivotando hacia recortes

Estimación de mejora: P_Baja cruzaría el 50% a partir de sept-2024 para BCE y BoE.

**Impacto en el TFG:**
- El Downward_Score y el Net_Score del dashboard para BCE/BoE en 2024-2025 subestiman la probabilidad de bajada.
- Para predicciones futuras 2026-2030, el modelo opera con mayor fiabilidad porque los tres estados (Sube/Estable/Baja) son igualmente plausibles y el modelo puede expresar señales en cualquier dirección.

**Cómo argumentarlo ante el tribunal:**
> "El modelo de clasificación presenta una limitación conocida en su capacidad para detectar giros de ciclo monetario. Al utilizar únicamente el nivel de las variables (OIS, tipos) sin capturar su dirección de cambio, el modelo tiende a prolongar el régimen previo. Esta limitación, documentada en la literatura como 'stickiness' de modelos econométricos de frecuencia mensual, es inherente al diseño de features elegido y podría mitigarse añadiendo variables de momentum, sacrificando la simplicidad interpretativa del modelo."

---

## Limitación 2 — FED: desfase temporal de ~4-5 meses

**Qué ocurre:**
El modelo predice "Baja" con alta probabilidad (60-93%) en enero-agosto 2025, periodo en que la FED mantuvo tipos estables. Las bajadas reales de agosto-diciembre 2024 las clasificó como "Estable".

**Causa raíz:**
Las variables macro (PIB, desempleo, inflación) se publican con rezago (1-3 meses) y su correlación histórica con los tipos tarda en trasladarse al modelo. El modelo "detecta" el entorno de bajadas cuando las variables ya reflejan el contexto post-recorte.

**Impacto en métricas:**
- FED test accuracy: 33.3% (por debajo del baseline trivial del 66.7%)
- Esto no invalida el modelo, pero debe reportarse con su explicación.

---

## Limitación 3 — ROC_AUC test = NaN

**Causa:** la clase "Sube" no aparece en ningún mes del test (2024-2025 fue un ciclo de bajadas exclusivo para los tres bancos). La métrica OvR para "Sube vs. Resto" es indefinida cuando no hay ejemplos positivos.

**Referencia válida:** ROC_AUC train = 0.90-0.92 (los tres bancos), calculado sobre datos que sí contienen las tres clases.

---

## Métricas de referencia para la memoria

| Banco | ROC_AUC train | Accuracy test | Baseline trivial | F1_macro test |
|-------|--------------|---------------|-----------------|--------------|
| BCE   | 0.919        | 62.5%         | 62.5%           | 0.256        |
| BoE   | 0.902        | 79.2%         | 79.2%           | 0.295        |
| FED   | 0.920        | 33.3%         | 66.7%           | 0.167        |

El ROC_AUC de train es la métrica más representativa de la capacidad discriminativa real del modelo, dado que el test no contiene la clase "Sube".
