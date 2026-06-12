Quiero que trabajes como programador y analista financiero especializado en modelos predictivos sencillos en Python. El objetivo es construir un dashboard para predecir tipos de interés oficiales mediante un modelo híbrido compuesto por regresión y clasificación.

Antes de empezar, lee y respeta el archivo CLAUDE.md del proyecto. Debes trabajar por fases. No avances automáticamente. Yo te indicaré qué fase quiero realizar escribiendo, por ejemplo: “Ejecuta la Fase 1”, “Ejecuta la Fase 2” o “Corrige la Fase 3”.

El proyecto consiste en crear un modelo híbrido sencillo en Python para la predicción de tipos de interés. El modelo debe tener dos partes complementarias.

La primera parte será un modelo de regresión, preferiblemente Random Forest Regressor o Regresión Lineal Múltiple, para estimar el valor numérico esperado del tipo de interés en el siguiente periodo. El resultado debe ser un porcentaje estimado del tipo de interés.

La segunda parte será un modelo de clasificación, preferiblemente Regresión Logística Multiclase, para predecir la dirección del movimiento del tipo de interés. El modelo debe clasificar si el tipo subirá, bajará o se mantendrá estable respecto al periodo anterior.

Además, la clasificación debe incorporar un sistema de scoring de 0 a 100 calculado a partir de las probabilidades generadas por el modelo. Deben calcularse dos scores:

Confidence Score, que representa la probabilidad de la clase predicha multiplicada por 100.

Upward Score, que representa la probabilidad específica de subida multiplicada por 100.

El entrenamiento debe respetar el orden temporal de los datos. No se puede mezclar aleatoriamente pasado y futuro. La división entre train y test debe realizarse cronológicamente.

La evaluación del modelo debe incluir métricas para la parte de regresión, como MAE, RMSE y R². También debe incluir métricas para la parte de clasificación, como accuracy, precision, recall, F1 Score y ROC AUC multiclase cuando sea aplicable.

El resultado final debe ser un dashboard de Python, preferiblemente en Streamlit.

La pantalla principal del dashboard debe mostrar los tres tipos de interés principales:

BCE.

BoE.

FED.

Una vez se seleccione uno de ellos, se debe desplegar el análisis correspondiente.

La parte izquierda de la pantalla debe mostrar un índice o panel de variables:

PIB.

PIB per cápita.

Inflación.

Desempleo.

Rating del país o zona económica.

VIX.

Precio del oro.

Precio de la plata.

Precio del gas.

Overnight Index Swaps como expectativas de mercado.

La parte central de la pantalla debe mostrar:

Tipo de interés actual 2026.

Tipo de interés predicho para 2027.

Tipo de interés predicho para 2028.

Tipo de interés predicho para 2029.

Tipo de interés predicho para 2030.

Gráfico histórico del tipo de interés.

Gráfico de predicción futura del tipo de interés.

Dirección prevista del movimiento.

Confidence Score.

Upward Score.

Debes ser crítico. Si detectas un fallo conceptual, metodológico, estadístico o técnico, debes indicarlo antes de escribir código. No inventes datos. Si faltan datos futuros para 2027, 2028, 2029 y 2030, debes crear una estructura de escenarios o un archivo plantilla donde se puedan introducir esos supuestos.

FASE 0. Revisión crítica del planteamiento

Objetivo:
Analizar si el planteamiento del modelo es correcto y detectar posibles errores antes de programar.

Tareas:
Revisa la lógica del modelo híbrido.
Evalúa si las variables propuestas son adecuadas.
Explica el problema de predecir 2027, 2028, 2029 y 2030 si no existen variables futuras.
Propón cómo resolverlo mediante escenarios.
Revisa si la clasificación de subida, bajada y estabilidad está bien planteada.
Propón un umbral para definir estabilidad.
Explica cómo debe calcularse el scoring.
Indica qué estructura de carpetas recomiendas para el proyecto.

Entregable:
Un diagnóstico crítico y una estructura inicial recomendada del proyecto.

FASE 1. Estructura del proyecto y archivos base

Objetivo:
Crear la estructura inicial del proyecto en Python.

Tareas:
Diseña la estructura de carpetas.
Crea los archivos base necesarios.
Crea un requirements.txt.
Crea un README inicial.
Crea un archivo de configuración donde se puedan definir rutas, bancos centrales, variables principales y parámetros del modelo.
Prepara una carpeta para datos históricos.
Prepara una carpeta para escenarios futuros.

Entregable:
Código y estructura de carpetas lista para empezar a trabajar.

FASE 2. Carga y organización de datos

Objetivo:
Crear el módulo de carga de datos históricos.

Tareas:
Crear funciones para cargar datos desde archivos CSV o Excel.
Alinear los datos por fecha.
Organizar los datos por banco central o zona económica: BCE, BoE y FED.
Comprobar valores faltantes.
Homogeneizar nombres de columnas.
Validar que las variables principales existen.
Crear una tabla final lista para modelar.

Variables principales:
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
Tipo de interés objetivo.

Entregable:
Un módulo de carga y limpieza de datos que devuelva un DataFrame preparado para el modelo.

FASE 3. Preparación de variables objetivo

Objetivo:
Crear las variables necesarias para regresión y clasificación.

Tareas:
Crear la variable objetivo de regresión: tipo de interés del siguiente periodo.
Crear la variable objetivo de clasificación comparando el tipo futuro con el tipo actual.
Clasificar el movimiento en tres clases: sube, baja o estable.
Crear un umbral configurable para considerar que el tipo se mantiene estable.
Eliminar filas donde no pueda calcularse la variable objetivo.
Separar variables explicativas y variables objetivo.

Entregable:
Un módulo que prepare X, y_regression e y_classification.

FASE 4. División temporal train test

Objetivo:
Crear una división temporal correcta.

Tareas:
Dividir los datos respetando el orden cronológico.
No usar train_test_split aleatorio.
Permitir configurar el porcentaje de entrenamiento, por ejemplo 80 por ciento train y 20 por ciento test.
Devolver X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls.
Guardar las fechas correspondientes para poder graficar los resultados.

Entregable:
Función de división temporal lista para usar.

FASE 5. Modelo de regresión

Objetivo:
Entrenar el modelo que predice el valor numérico del tipo de interés.

Tareas:
Implementar Random Forest Regressor.
Implementar como alternativa Regresión Lineal Múltiple.
Permitir seleccionar el modelo mediante configuración.
Entrenar el modelo con los datos de entrenamiento.
Predecir sobre el conjunto de prueba.
Evaluar con MAE, RMSE y R².
Guardar resultados y predicciones.

Entregable:
Módulo de regresión con entrenamiento, predicción y evaluación.

FASE 6. Modelo de clasificación

Objetivo:
Entrenar el modelo que predice la dirección del movimiento del tipo de interés.

Tareas:
Implementar Regresión Logística Multiclase.
Entrenar el modelo respetando la división temporal.
Predecir la clase: sube, baja o estable.
Obtener probabilidades por clase.
Calcular accuracy, precision, recall, F1 Score y ROC AUC multiclase cuando sea posible.
Crear matriz de confusión.
Guardar resultados y predicciones.

Entregable:
Módulo de clasificación con predicción, probabilidades y métricas.

FASE 7. Sistema de scoring

Objetivo:
Crear el sistema de scoring de 0 a 100.

Tareas:
Calcular Confidence Score como la probabilidad de la clase predicha multiplicada por 100.
Calcular Upward Score como la probabilidad de subida multiplicada por 100.
Incorporar ambos scores a la tabla de resultados.
Explicar cómo interpretar los scores.
Preparar los scores para ser mostrados en el dashboard.

Entregable:
Función de scoring y tabla final de predicciones con scores.

FASE 8. Predicción futura 2027, 2028, 2029 y 2030

Objetivo:
Crear la predicción futura del tipo de interés.

Tareas:
Crear una plantilla para escenarios futuros con las variables necesarias para 2027, 2028, 2029 y 2030.
Permitir cargar escenarios desde CSV o Excel.
Aplicar el modelo de regresión para predecir el tipo futuro.
Aplicar el modelo de clasificación para predecir la dirección.
Calcular Confidence Score y Upward Score.
Generar una tabla final con predicciones para cada año.

Importante:
No inventes valores futuros. Si no existen datos, crea una plantilla vacía o de ejemplo claramente marcada como escenario hipotético.

Entregable:
Módulo de predicción futura y archivo plantilla de escenarios.

FASE 9. Gráficos históricos y predictivos

Objetivo:
Crear las visualizaciones principales del dashboard.

Tareas:
Crear gráfico histórico del tipo de interés.
Crear gráfico de predicción futura para 2027, 2028, 2029 y 2030.
Crear gráfico combinado con histórico y predicción.
Crear gráficos de importancia de variables si se usa Random Forest.
Crear gráficos de métricas del modelo.
Preparar todos los gráficos para integrarlos en Streamlit.

Entregable:
Funciones de visualización reutilizables.

FASE 10. Dashboard en Streamlit

Objetivo:
Construir el dashboard final.

Tareas:
Crear una pantalla principal con selector de banco central: BCE, BoE y FED.
Al seleccionar un banco central, mostrar su análisis.
Crear panel lateral izquierdo con índice y variables principales.
Mostrar en la parte central:
Tipo actual 2026.
Tipo predicho para 2027.
Tipo predicho para 2028.
Tipo predicho para 2029.
Tipo predicho para 2030.
Gráfico histórico.
Gráfico predictivo.
Dirección esperada.
Confidence Score.
Upward Score.
Métricas de regresión.
Métricas de clasificación.
Permitir seleccionar escenario futuro si existen varios.

Entregable:
Archivo app.py funcional en Streamlit.

FASE 11. Validación final y corrección de errores

Objetivo:
Revisar que todo el proyecto funciona de forma coherente.

Tareas:
Comprobar que los datos cargan correctamente.
Comprobar que los modelos entrenan correctamente.
Comprobar que no existe fuga temporal de información.
Comprobar que las métricas se calculan correctamente.
Comprobar que el dashboard se ejecuta sin errores.
Corregir bugs.
Mejorar claridad visual del dashboard.
Preparar instrucciones finales de ejecución.

Entregable:
Proyecto final revisado y listo para ejecutar.

Cuando te pida una fase, responde únicamente con el desarrollo de esa fase. No desarrolles fases posteriores salvo que te lo pida expresamente.
