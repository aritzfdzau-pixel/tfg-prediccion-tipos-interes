# Prompt para Claude Code — Apartado "Caso Iberdrola" en el dashboard

Copia desde aquí hacia abajo:

---

Quiero añadir un nuevo apartado interactivo a mi dashboard Streamlit (`app.py`) llamado **"💼 Caso Iberdrola"**, que resuelva un caso práctico de optimización de la financiación de Iberdrola usando las predicciones ya generadas por el modelo. Es la parte central de la defensa de mi TFG, así que debe ser visual, interactivo y robusto.

## Contexto del proyecto

- Dashboard Streamlit en `app.py` (~2.600 líneas), con tabs principales: "🌍 Panorama global", "📊 Historico y test", "🔮 Predicciones 2026-2030", "🔬 Evaluacion del modelo". Añade "💼 Caso Iberdrola" como quinta tab en `main()`.
- Ya existe un script funcional con TODA la lógica de negocio del caso: **`run_caso_iberdrola.py`** (raíz del repo). Léelo primero y úsalo como referencia canónica de la metodología. No reinventes los cálculos: refactoriza esa lógica a un módulo `src/models/caso_iberdrola.py` con funciones puras parametrizadas, y haz que tanto `run_caso_iberdrola.py` como la nueva tab lo importen (una sola fuente de verdad).
- Datos de entrada (ya generados, en `Resultados/`): `predicciones_futuras_{banco}_{escenario}.csv` y `predicciones_futuras_{banco}_{escenario}_OIS.csv`, con banco ∈ {BCE, BoE, FED} y escenario ∈ {base, optimista, pesimista}. Columnas relevantes: índice fecha mensual 2026-2030, `tipo_predicho`, `direccion`, `P_Baja`, `P_Estable`, `P_Sube`, `Confidence_Score`.
- Estilo: el app usa paleta Iberdrola definida en CSS (variables `--verde-iberdrola: #009A44`, `--verde-oscuro`, `--azul-iberdrola: #0083CA`, etc.), tipografía Nunito y Plotly para gráficos. Sigue las convenciones existentes: helpers de carga con `@st.cache_data` y prefijo `_cargar_`, secciones como funciones `_seccion_*`, textos en español.

## El caso práctico (metodología, ya implementada en run_caso_iberdrola.py)

Iberdrola refinancia sus vencimientos reales 2026-2028 (5.392, 4.123 y 4.424 M€; datos FY2025). Se comparan 4 estrategias de refinanciación calculando el coste financiero total 2026-2030 (M€) bajo los 3 escenarios:

1. **Statu quo**: mix actual — 77,2% fijo (cuentas anuales 2025), divisas EUR 55% / GBP 25% / USD 20%.
2. **Conservadora**: 100% fijo al tipo swap del año de emisión.
3. **Agresiva**: 100% variable, concentrada en EUR (80/10/10).
4. **Guiada por el modelo**: variable mientras no haya señal de subida; gira a fijo cuando el clasificador da P(Sube) anual > umbral **o** la regresión predice subida acumulada > umbral el año siguiente. Divisas 60/25/15.

Reglas de valoración clave (respétalas exactamente):
- Coste variable del año t = tipo oficial predicho (media anual) + spread corporativo (110 pb por defecto).
- Tipo swap contratado en el año t = media de la senda OIS predicha desde t hasta 2030 + spread. **Los swaps contratados en 2026 se valoran con la curva OIS del escenario base** (es lo que cotiza el mercado hoy); a partir de 2027, con la curva del escenario evaluado.
- Pesos de los escenarios derivados del clasificador: verosimilitud direccional media de cada escenario (probabilidad media que el clasificador asigna a la dirección que sigue la senda del propio escenario, con umbral ±0,05 pp/mes), normalizada. Resultado actual: base 35,6% / optimista 34,5% / pesimista 29,9%.

## Diseño de la nueva tab

**Bloque 1 — Enunciado**: card con el planteamiento del caso (vencimientos reales, plan inversor 58.000 M€, pregunta: ¿cómo refinanciar minimizando coste esperado sin comprometer el rating BBB+?).

**Bloque 2 — Supuestos interactivos** (en un expander o columna de controles):
- Slider spread corporativo (50-200 pb, defecto 110).
- Sliders o number_inputs del mix de divisas del statu quo (defecto 55/25/20, normalizar a 100%).
- Slider % fijo del statu quo (defecto 77,2%).
- Sliders de los umbrales de la estrategia 4: P(Sube) (defecto 0,35) y subida regresión (defecto 0,25 pp).
- Radio para los pesos de escenario: "Derivados del clasificador (recomendado)" vs "Manuales" (3 sliders normalizados).
- Botón/estado que recalcule todo en vivo (Streamlit ya rerenderiza; usa cache solo en la carga de CSVs, no en el cálculo).

**Bloque 3 — Resultados**:
- Tabla de pesos de escenarios con explicación breve de cómo se derivan.
- Tabla principal: 4 estrategias × (coste base, optimista, pesimista, esperado), con la celda del mínimo por columna resaltada en verde Iberdrola y la estrategia ganadora por coste esperado destacada.
- Gráfico Plotly de barras agrupadas: coste por estrategia y escenario + marcador del coste esperado.
- Gráfico de perfil riesgo-retorno: eje X = coste en escenario pesimista (riesgo), eje Y = coste esperado; cada estrategia un punto — visualiza la dominancia media-riesgo de la estrategia 4.
- Tabla de "giros a fijo" de la estrategia 4 (banco × escenario → año de giro o "nunca"), con nota de que la señal combina clasificador y regresión.
- `st.metric`s destacados: ahorro esperado de la estrategia 4 vs statu quo (M€), y diferencia vs agresiva en el escenario pesimista.

**Bloque 4 — Interpretación**: texto breve generado a partir de los resultados actuales (qué estrategia gana por coste esperado, cuál protege mejor en el pesimista, conclusión media-riesgo). Debe actualizarse dinámicamente con los supuestos que toque el usuario.

## Requisitos técnicos

- No rompas ninguna tab existente; no cambies la firma de funciones ya usadas.
- La tab no depende del selector de banco/escenario del sidebar: usa siempre los 3 bancos y 3 escenarios (indícalo con un caption).
- Maneja con `st.error` la ausencia de los CSVs necesarios.
- Números en formato español (separador de miles con punto) y unidades M€ visibles.
- Al terminar: ejecuta `python -m py_compile app.py` y arranca el app para verificar que la tab renderiza sin errores; ejecuta también `run_caso_iberdrola.py` y comprueba que con los valores por defecto la tabla del dashboard reproduce exactamente sus resultados (statu quo 1.679, conservadora 1.659, agresiva 1.606, modelo 1.609 M€ esperados).
