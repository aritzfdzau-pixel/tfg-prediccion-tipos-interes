# Caso práctico: aplicación del modelo a la problemática financiera de Iberdrola

Propuesta de diseño para la defensa del TFG — borrador de trabajo (julio 2026)

## 1. Planteamiento del caso

Iberdrola ejecuta un plan inversor de ~58.000 M€ (2025-2028) intensivo en financiación externa, con inversión concentrada en Eurozona, Reino Unido y EEUU — exactamente las tres zonas monetarias que cubre el modelo (BCE, BoE, FED). La pregunta del caso:

> **¿Cómo debería estructurar Iberdrola su financiación 2026-2028 (momento de emisión, divisa y tipo fijo/variable) según las predicciones del modelo?**

El caso convierte el modelo en una herramienta de decisión de tesorería corporativa, respondiendo directamente a la problemática planteada en el apartado 3.1.3 del TFG.

## 2. Datos reales de anclaje (Iberdrola, resultados FY2025)

- Deuda neta: **50.200 M€** (reducida en 1.500 M€ en 2025).
- Coste medio de la deuda: **4,81%** (3,7% excluyendo Neoenergia/Brasil).
- Vencimientos de deuda: **5.392 M€ en 2026, 4.123 M€ en 2027, 4.424 M€ en 2028** → ~14.000 M€ a refinanciar en el horizonte de predicción del modelo.
- Emisiones recientes reales (para dar realismo al caso):
  - Mayo 2025: bono verde 750 M€, cupón 3,50%, vencimiento 2035.
  - Octubre 2025: híbrido verde 1.000 M€, cupón 3,75%, perpetuo (call 2031).
  - Enero 2026: híbrido verde UE 600 M€, cupón 3,95%.
  - Marzo 2026: bono verde en dos tramos: 750 M€ al 3,125% (venc. 2030) y 750 M€ al 3,75% (venc. 2036).
- Pendiente de verificar en la presentación FY25 (PDF resultados-25FY): % deuda a tipo fijo vs. variable y desglose por divisa (EUR/GBP/USD/BRL).

Fuentes: [Resultados FY2025 Iberdrola](https://www.iberdrola.com/documents/20125/5693151/resultados-25FY.pdf), [Estrategia financiera](https://www.iberdrola.com/shareholders-investors/investors/fixed-income/financial-strategy), [Nota de prensa resultados 2025](https://www.iberdrola.com/sala-comunicacion/noticias/detalle/iberdrola-invierte-14460-millones-en-2025-y-aumenta-el-beneficio-neto-un-12-por-ciento), [Emisión híbrido 1.000 M€](https://www.iberdrola.com/press-room/news/detail/iberdrola-issues-1-billion-first-eu-green-hybrid-bond), [Emisión híbrido 600 M€](https://www.iberdrola.com/press-room/news/detail/iberdrola-issues-600-million-euros-eu-green-hybrid-bond), [Bono verde 1.500 M€ marzo 2026](https://www.esgtoday.com/iberdrola-issues-e1-5-billion-green-bond-to-finance-grid-renewables-investments/).

## 3. Predicciones del modelo (media anual del tipo oficial predicho, %)

| Banco | Escenario | 2026 | 2027 | 2028 | 2029 | 2030 |
|---|---|---|---|---|---|---|
| BCE | Base | 1,68 | 1,20 | 1,10 | 1,15 | 1,15 |
| BCE | Optimista | 1,65 | 1,20 | 1,10 | 1,15 | 1,00 |
| BCE | Pesimista | 2,22 | 3,05 | 2,60 | 2,15 | 1,65 |
| FED | Base | 3,54 | 3,05 | 2,75 | 2,65 | 2,65 |
| FED | Optimista | 3,46 | 2,95 | 2,85 | 2,85 | 2,75 |
| FED | Pesimista | 3,92 | 4,75 | 4,30 | 3,60 | 3,15 |
| BoE | Base | 2,96 | 1,80 | 1,60 | 1,50 | 1,40 |
| BoE | Optimista | 2,75 | 1,60 | 1,55 | 1,45 | 1,30 |
| BoE | Pesimista | 3,53 | 3,85 | 3,40 | 2,80 | 2,20 |

Lecturas clave para el caso:

1. **EUR es la divisa de financiación más barata en todos los escenarios** (base 2027: BCE 1,20% vs. BoE 1,80% vs. FED 3,05%).
2. **En escenario base, los tres bancos centrales bajan tipos hasta 2028** → conviene retrasar la fijación de tipos / mantener exposición variable a corto plazo.
3. **El escenario pesimista invierte la recomendación** (BCE en 3,05% en 2027) → el clasificador direccional y sus probabilidades calibradas determinan cuánta cobertura contratar.

## 4. Bloque A — Decisión de emisión: timing y divisa

**Situación:** Iberdrola debe refinanciar ~5.400 M€ que vencen en 2026. Simplificación de trabajo: una emisión de 1.000 M€.

**Decisión 1 (timing):** ¿emitir en enero 2026 o esperar 12-18 meses?
- Escenario base BCE: 1,68% (2026) → 1,20% (2027), −48 pb. Sobre 1.000 M€ ≈ **4,8 M€/año de ahorro en intereses** si el coste de emisión se traslada 1:1; en un bono a 10 años, ~48 M€ de ahorro acumulado (sin descontar).
- Contraste con el escenario pesimista (BCE 3,05% en 2027): esperar costaría +18,5 M€/año. La decisión se pondera con las probabilidades del clasificador (P_Estable ~70-75% en 2026 para BCE base).

**Decisión 2 (divisa):** comparar las tres trayectorias → emitir en EUR concentra el ahorro; deuda en GBP/USD se justifica como cobertura natural de los ingresos regulados en UK/EEUU (matching divisa ingresos-deuda), no por coste. Esto conecta con la estrategia real de Iberdrola (mayor peso de la libra en su estructura de deuda en 2025).

**Validación con la realidad:** las emisiones reales de Iberdrola de 2025-2026 (cupones 3,95% → 3,125% entre enero y marzo 2026) son coherentes con la senda descendente que predice el modelo — argumento potente para la defensa.

## 5. Bloque B — Fijo vs. variable y cobertura con IRS

Usa la parte de clasificación (la "otra mitad" del modelo híbrido):

- Si P(Baja) + P(Estable) es alta (escenario base BCE) → mantener/aumentar el tramo de deuda a tipo variable, que se abaratará con la senda descendente.
- Si P(Sube) gana peso (escenario pesimista, FED) → fijar tipo mediante swaps (IRS: pagar fijo, recibir variable) sobre el tramo USD.
- **Regla de decisión propuesta** (aporte original del TFG): dimensionar el porcentaje de cobertura en función del Confidence Score / probabilidades calibradas. Ejemplo: cubrir un % del nocional proporcional a P(Sube). Así el modelo no solo predice: prescribe.

Aquí luces la calibración isotónica y las matrices de confusión ya calculadas — demuestras que las probabilidades son fiables, no decorativas.

## 6. Bloque C — Sensibilidad WACC / VAN de un proyecto tipo

**Proyecto ilustrativo:** inversión en redes reguladas de 2.000 M€ (dos tercios del plan van a redes), financiada 50% con deuda.

- Kd por escenario = tipo oficial predicho + spread corporativo (~130-190 pb, calibrado con las emisiones reales: cupón 3,125% en marzo 2026 vs. tipo BCE ~2,0%).
- WACC = E/V·Ke + D/V·Kd·(1−t) → calcular VAN del proyecto bajo los tipos de los 3 escenarios.
- Resultado esperado: mostrar cuántos M€ de VAN separa el escenario optimista del pesimista, y si algún escenario compromete la rentabilidad mínima. El pesimista funciona además como stress test (impacto en gasto financiero y en ratios crediticios FFO/deuda neta que sostienen el rating).

## 7. Estructura narrativa para la defensa

1. Problema real: 58.000 M€ de plan inversor + 14.000 M€ de vencimientos 2026-2028 + incertidumbre de tipos (cita de la propia Iberdrola).
2. Herramienta: el modelo híbrido (regresión = cuánto, clasificación = hacia dónde y con qué confianza).
3. Decisión A: cuándo y en qué divisa emitir → ahorro cuantificado en M€.
4. Decisión B: cuánto cubrir con swaps → regla basada en probabilidades calibradas.
5. Decisión C: impacto en WACC/VAN y stress test → viabilidad del plan bajo el peor escenario.
6. Cierre: contraste con las emisiones reales de Iberdrola 2025-2026 (el mercado se comportó como predecía el escenario base).

## 8. Limitaciones a reconocer proactivamente (preguntas probables del tribunal)

- El modelo predice **tipos oficiales (corto plazo)**, no el coste de emisión a 10 años. Puente defendible: (a) la deuda variable está indexada a Euribor/€STR, que sigue al tipo oficial; (b) el tramo corto de la curva ancla las decisiones de timing; (c) para el coste a largo se añade un spread supuesto constante — decláralo como supuesto.
- La comparación entre divisas ignora el coste de cobertura cambiaria (paridad cubierta de intereses). Se justifica porque Iberdrola tiene ingresos naturales en GBP/USD (cobertura natural).
- Los escenarios 2027-2030 dependen de supuestos macro exógenos, no de datos observados — es una fortaleza si lo presentas como análisis de escenarios, no como predicción puntual.
- El coste real de Iberdrola (4,81%) incluye Brasil; usar el 3,7% ex-Brasil como referencia comparable.

## 9. Próximos pasos

1. Verificar en la presentación FY25 el mix fijo/variable y por divisa (los gráficos Datawrapper de la web de Iberdrola contienen el dato exacto a 31/03/2026).
2. Implementar los cálculos de los bloques A y C como script reproducible (`run_caso_iberdrola.py`) que lea directamente `Resultados/predicciones_futuras_*.csv`.
3. Añadir una pestaña "Caso Iberdrola" al dashboard Streamlit para mostrarlo en vivo en la defensa.
