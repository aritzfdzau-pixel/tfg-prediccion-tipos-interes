# CLAUDE.md

## Objetivo general del proyecto

El objetivo de este proyecto es construir un dashboard en Python para analizar y predecir la evolución de los principales tipos de interés oficiales: BCE, BoE y FED. El modelo debe combinar una predicción numérica del tipo de interés futuro con una predicción direccional del movimiento esperado.

El proyecto debe estar desarrollado de forma ordenada, modular y progresiva, permitiendo trabajar por fases independientes. Cada fase debe poder ejecutarse, revisarse y corregirse antes de avanzar a la siguiente.

## Enfoque metodológico

El modelo debe ser un modelo híbrido sencillo compuesto por dos partes:

Primero, un modelo de regresión para estimar el valor numérico esperado del tipo de interés en el siguiente periodo. Se podrán utilizar modelos como Random Forest Regressor o Regresión Lineal Múltiple.

Segundo, un modelo de clasificación para predecir la dirección del movimiento del tipo de interés. La clasificación debe determinar si el tipo de interés subirá, bajará o se mantendrá estable respecto al periodo anterior. Para esta parte se podrá utilizar Regresión Logística Multiclase u otro modelo sencillo y justificable.

El entrenamiento debe respetar siempre el orden temporal de los datos. No se debe usar división aleatoria entre entrenamiento y prueba, ya que esto mezclaría información pasada y futura. La división debe realizarse cronológicamente:

- **Train:** 1999M01 – 2023M12
- **Test:** 2024M01 – 2025M12
- **Predicción futura:** 2026 – 2030 mediante escenarios hipotéticos

## Variables principales

Las variables explicativas principales serán:

PIB.

PIB per cápita.

Inflación.

Desempleo.

Rating del país o zona económica.

VIX.

Precio del oro.

Precio de la plata.

Precio del gas.

Overnight Index Swaps, como variable representativa de las expectativas de mercado.

Los tipos de interés objetivo serán:

Tipo oficial del Banco Central Europeo.

Tipo oficial del Banco de Inglaterra.

Tipo oficial de la Reserva Federal estadounidense.

## Tratamiento de datos

El código debe organizar los datos en **frecuencia mensual**. Esta es la frecuencia base del modelo.

Reglas de transformación por tipo de variable:
- Variables diarias (tipos oficiales, OIS): agregar a media mensual.
- Variables mensuales (VIX, oro, plata, gas): usar directamente.
- Variables anuales (PIB, PIBpc, inflación, desempleo): aplicar forward-fill mensual. El valor del año X se replica en los 12 meses de ese año.
- Rating: variable estática, mismo valor para todos los meses.

Todas las variables deben quedar alineadas por fecha y por zona económica. Si alguna variable tiene una frecuencia diferente, debe transformarse de forma coherente, evitando introducir sesgos.

Los valores faltantes deben tratarse de forma explícita. No se deben eliminar datos sin justificarlo. Se deben explicar las decisiones tomadas en limpieza, interpolación, forward fill o eliminación de observaciones.

Las variables futuras necesarias para predecir 2027, 2028, 2029 y 2030 deben tratarse mediante escenarios o supuestos si no existen datos reales disponibles.

## Modelos

Para la parte de regresión, el modelo debe predecir el valor numérico del tipo de interés futuro. La variable objetivo será el tipo de interés del siguiente periodo.

Para la parte de clasificación, el modelo debe predecir la dirección del movimiento del tipo de interés. La variable objetivo debe construirse comparando el tipo de interés futuro con el tipo actual.

La clasificación debe incluir tres clases:

Sube.

Baja.

Estable.

Debe definirse un umbral razonable para considerar que el tipo se mantiene estable. Por ejemplo, una variación absoluta inferior a 0,10 puntos porcentuales podrá clasificarse como estable, aunque este umbral debe poder modificarse fácilmente.

## Sistema de scoring

El modelo de clasificación debe generar probabilidades para cada clase.

A partir de estas probabilidades, se debe crear un sistema de scoring de 0 a 100.

Deben calcularse dos puntuaciones:

Confidence Score: probabilidad de la clase predicha multiplicada por 100. Indica la confianza general del modelo en su propia predicción.

Upward Score: probabilidad de subida multiplicada por 100. Indica la probabilidad específica de que el tipo de interés suba.

El dashboard debe mostrar estos scores de forma clara y fácil de interpretar.

## Evaluación del modelo

La parte de regresión debe evaluarse con las siguientes métricas:

MAE.

RMSE.

R².

La parte de clasificación debe evaluarse con las siguientes métricas:

Accuracy.

Precision.

Recall.

F1 Score.

ROC AUC, siempre que sea aplicable para clasificación multiclase.

Debe evitarse presentar el modelo como exacto o infalible. Los resultados deben interpretarse como estimaciones basadas en datos históricos y escenarios futuros.

## Dashboard

El resultado final debe ser un dashboard en Python, preferiblemente desarrollado con Streamlit.

La pantalla principal debe mostrar los tres tipos de interés principales:

BCE.

BoE.

FED.

El usuario debe poder seleccionar uno de los tres bancos centrales. Al hacer clic o seleccionar uno, se debe desplegar el análisis específico correspondiente.

La parte izquierda del dashboard debe incluir un índice o panel de variables con las principales variables explicativas:

PIB.

PIB per cápita.

Inflación.

Desempleo.

Rating.

VIX.

Oro.

Plata.

Gas.

OIS.

La parte central del dashboard debe mostrar:

Tipo de interés actual en 2026.

Tipo de interés predicho para 2027.

Tipo de interés predicho para 2028.

Tipo de interés predicho para 2029.

Tipo de interés predicho para 2030.

Gráfico histórico del tipo de interés.

Gráfico de predicción futura del tipo de interés.

Dirección esperada del movimiento.

Confidence Score.

Upward Score.

## Estilo de programación

El código debe ser limpio, modular y fácil de entender.

Debe evitarse crear scripts excesivamente largos sin estructura.

Cuando sea posible, se deben separar las funciones en bloques claros:

Carga de datos.

Limpieza de datos.

Preparación de variables.

Entrenamiento del modelo de regresión.

Entrenamiento del modelo de clasificación.

Evaluación.

Predicción futura.

Dashboard.

Visualización.

Cada función debe tener un propósito claro.

## Entregables esperados

El proyecto debe entregar:

Un script principal del dashboard.

Scripts auxiliares para procesamiento de datos y modelos, si son necesarios.

Un archivo de requisitos con las librerías necesarias.

Un archivo README explicando cómo ejecutar el proyecto.

Tablas de resultados del modelo.

Gráficos históricos y predictivos.

Métricas de evaluación.

Predicciones para 2027, 2028, 2029 y 2030.

## Reglas de trabajo

Trabaja siempre por fases.

No avances a la siguiente fase hasta que la fase anterior esté terminada y sea funcional.

Antes de escribir código, explica brevemente qué vas a hacer.

Después de escribir código, explica cómo ejecutarlo y qué salida debería obtenerse.

Si detectas un error conceptual, metodológico o técnico, indícalo claramente antes de continuar.

Prioriza soluciones sencillas, robustas y explicables.

No sobrecompliques el modelo si una solución más simple permite cumplir el objetivo.

No inventes datos. Si faltan datos, crea una estructura preparada para recibirlos o utiliza supuestos claramente identificados como escenarios.

El objetivo no es crear un modelo perfecto, sino un modelo académico, entendible y defendible para un Trabajo de Fin de Grado.

## Uso de subagentes

Este proyecto utiliza cuatro subagentes especializados definidos en `.claude/agents/`. Cada agente tiene un dominio claro y no debe invadir el de los otros.

**REGLA FUNDAMENTAL: Ningún agente debe ejecutar ninguna fase sin que el usuario lo pida expresamente. La instrucción de avanzar siempre parte del usuario.**

### data-engineer-agent

Responsable de todo lo relacionado con datos: carga, limpieza, alineación temporal, valores faltantes, variables objetivo y escenarios futuros.

Úsalo en: Fase 0 (diagnóstico de datos), Fase 1 (estructura de carpetas), Fase 2 (carga y limpieza), Fase 3 (variables objetivo), Fase 4 (división temporal), Fase 8 (plantillas de escenarios futuros).

No entrena modelos. No crea visualizaciones. No inventa datos futuros.

### ml-model-agent

Responsable del modelo híbrido: la parte de regresión (Random Forest / Regresión Lineal) y la parte de clasificación (Regresión Logística Multiclase), así como el sistema de scoring y las predicciones futuras.

Úsalo en: Fase 5 (regresión), Fase 6 (clasificación), Fase 7 (scoring), Fase 8 (predicción futura con escenarios).

No procesa datos en bruto. No construye el dashboard. No usa modelos complejos no justificables para un TFG.

### dashboard-agent

Responsable del dashboard Streamlit: gráficos históricos y predictivos, interfaz de usuario, selector de banco central, visualización de scores y métricas.

Úsalo en: Fase 9 (gráficos), Fase 10 (dashboard completo), Fase 11 (correcciones visuales y validación de la UI).

No entrena modelos. No procesa datos. No presenta predicciones como hechos ciertos.

### review-agent

Responsable de la revisión crítica del proyecto: fuga temporal, corrección de métricas, coherencia de escenarios, calidad académica y defensa del TFG.

Úsalo en: Fase 0 (diagnóstico inicial), Fase 11 (validación final), y en cualquier momento que surjan dudas metodológicas o estadísticas.

No escribe código. No modifica datos ni modelos. Reporta problemas y propone correcciones para que los otros agentes las implementen.
