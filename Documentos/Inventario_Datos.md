# Inventario de Datos — Fase 1
## Modelo Predictivo de Tipos de Interés a Largo Plazo
**Fecha de análisis:** 2026-06-02

---

## 1. Estructura de la carpeta del proyecto

```
MODELO/
├── Datos originales/        ← 8 archivos Excel con datos históricos
├── Datos procesados/        ← VACÍA (pendiente de poblar)
├── modelo/                  ← VACÍA (pendiente de poblar)
├── Resultados/              ← VACÍA (pendiente de poblar)
├── documentos_markdown/     ← 3 documentos de análisis cualitativo
├── .claude/                 ← Configuración del entorno
└── CLAUDE.md                ← Instrucciones maestras del proyecto
```

---

## 2. Inventario de archivos en /Datos originales

### Archivo 1 — `Analisis_Desempleo_1993_2024.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 25 KB |
| Hojas | Datos, Crecimiento anual, Estadística |
| **Variable principal** | Tasa de desempleo anual (% población activa) |
| **Regiones** | Eurozona (20 países, media aritmética simple), EE.UU., Reino Unido |
| **Frecuencia** | Anual |
| **Período** | 1993–2024 (32 observaciones) |
| Estructura | Año en columna, región en columna |
| Variables adicionales | Crecimiento interanual del desempleo (hoja "Crecimiento anual") |
| Estadísticos | Media, desviación típica, máx/mín por bloque |

**Países incluidos en Eurozona:** Alemania, Austria, Bélgica, Chipre, Croacia, Eslovaquia, Eslovenia, España, Estonia, Finlandia, Francia, Grecia, Irlanda, Italia, Letonia, Lituania, Luxemburgo, Malta, Países Bajos, Portugal

**Problemas detectados:**
- Encoding con caracteres especiales (tildes) — requerirá `encoding='utf-8'` o `latin-1` al importar.
- La Eurozona es una media simple, no ponderada por población.

---

### Archivo 2 — `Analisis_OIS_EUR_USD_GBP.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 228 KB |
| Hojas | Datos_Diarios, Datos_Anuales, Estadisticos, Graficos |
| **Variables** | EUR €STR (decimal), GBP SONIA (decimal), USD SOFR (%) |
| **Regiones** | Eurozona (EUR), Reino Unido (GBP), EE.UU. (USD) |
| **Frecuencia** | Diaria (hoja Datos_Diarios) / Anual (hoja Datos_Anuales) |
| **Período diario** | 1999-01-01 a ~2026-01-21 (9.886 observaciones) |
| **Período anual** | 2020–2025 (6 años) |

**Problemas críticos de datos:**
- **SOFR:** Primeros datos no-cero desde **2018-04-02** (SOFR se creó en 2018). Todos los valores 1999–2017 son cero. **No son valores reales.**
- **EUR €STR:** Primeros datos no-cero desde **2022-09-14** (€STR sustituyó a EONIA; operativo desde 2019 pero los datos en este archivo solo tienen valores reales a partir de 2022). **Datos previos son ceros.**
- **SONIA:** Disponible desde 1999 con valores reales. Sin problemas de ceros iniciales.
- Unidades heterogéneas: €STR y SONIA en formato decimal (0.03 = 3%), SOFR en porcentaje (3.00 = 3%). Requiere normalización.
- Los datos anuales solo cubren 2020–2025, insuficiente para el modelo histórico (se deberá usar los datos diarios y agregar).

---

### Archivo 3 — `Analisis_PIBpc_Inflacion_Eurozona_EEUU_UK_1993-2024.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 110 KB |
| Hojas | PORTADA, DATOS_PIBPC, DATOS_PIB, DATOS_POB, DATOS_INF, AGREGADO, CRECIMIENTO, ESTADISTICOS, GRAFICOS |
| **Frecuencia** | Anual |
| **Período** | 1993–2024 (32 años) |
| **Regiones** | 20 países Eurozona + EE.UU. + Reino Unido (22 en total) |

**Variables por hoja:**

| Hoja | Variable | Unidad |
|------|----------|--------|
| DATOS_PIBPC | PIB per cápita por país | USD constantes 2015 |
| DATOS_PIB | PIB total por país | USD constantes 2015 |
| DATOS_POB | Población por país | Personas (derivada: PIB/PIBpc) |
| DATOS_INF | Inflación anual IPC por país | % variación interanual |
| AGREGADO | PIB Eurozona, EEUU, UK + PIBpc + Inflación | USD 2015 / % |
| CRECIMIENTO | Crecimiento interanual PIBpc e inflación comparada | % |

**Estructura en hoja AGREGADO** (columnas disponibles):
- Año, PIB Eurozona (USD 2015), Población Eurozona, PIBpc Eurozona (USD 2015)
- PIBpc EEUU (USD 2015), PIBpc Reino Unido (USD 2015)
- Inflación Eurozona (%), Inflación EEUU (%), Inflación Reino Unido (%)

**Problemas detectados:**
- Estructura pivotada: países en filas, años en columnas (requiere transposición).
- La inflación de la Eurozona en AGREGADO es **media ponderada por población** (más correcta que la media simple del archivo de desempleo).

---

### Archivo 4 — `Analisis_Tipos_Interes.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 221 KB |
| Hojas | Portada, Datos diarios, Datos anuales, Crecimiento anual, Índice base 100, Estadísticos, Gráficos |
| **Variables** | BCE - Facilidad Marginal de Crédito, Banco de Inglaterra - Bank Rate, Reserva Federal - Federal Funds Rate |
| **Regiones** | Eurozona (BCE), Reino Unido (BoE), EE.UU. (Fed) |
| **Frecuencia** | Diaria (hoja "Datos diarios") / Anual (hoja "Datos anuales") |
| **Período diario** | 1999-01-01 a 2026-01-22 (9.884 observaciones) |
| **Período anual** | 1999–2025 (27 años) |
| Unidades | Decimal (0.045 = 4.5%) |

**Advertencia importante:** Estos son los **tipos oficiales de los bancos centrales** (tipos de política monetaria a corto plazo), **no** los tipos de interés a largo plazo (yield del bono soberano a 10 años). El tipo objetivo a largo plazo (variable dependiente del modelo) **no está presente** en este archivo como serie histórica numérica.

**Problemas detectados:**
- El BCE publica tres tipos (depósito, MRO, facilidad marginal de crédito). Este archivo recoge la **facilidad marginal de crédito** (el techo del corredor de tipos), no el tipo de depósito (más relevante desde 2022).

---

### Archivo 5 — `Anexo_Inflacion_Calculos.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 55 KB |
| Hojas | 1. Datos brutos, 2. Estadísticas descriptivas, 3. Mapa de calor, 4. Evolución agrupada, 5. Leyenda de fórmulas |
| **Variable** | Inflación anual (%) — IPC |
| **Período** | 1993–2025 |
| **Regiones** | Mismos 22 países/bloques que el archivo 3 |

**Problemas detectados:**
- Error de encoding al leer con Python (`charmap` no puede codificar `≤`). Requiere `engine='openpyxl'` y gestión de caracteres especiales.
- Contiene fórmulas activas en Excel que pueden no calcularse al leer con pandas (`data_only=True` en openpyxl, o recalcular con LibreOffice).
- Es un archivo auxiliar/complementario del archivo 3; puede haber solapamiento de datos.

---

### Archivo 6 — `DATASET.xlsx` ← ARCHIVO MASTER
| Campo | Detalle |
|-------|---------|
| Tamaño | 1.313 KB (el más grande) |
| Hojas | Bancos centrales, I (vacía), INFLACIÓN, DESEMPLEO, PIB A PRECIO CONST, PIB PER CAPITA PREC COST, Materias primas, Indices materias primas |

**Detalle por hoja:**

#### Hoja: Bancos centrales
| Variable | Tipo | Frecuencia | Período |
|----------|------|------------|---------|
| ECB Depo Facility | Tipo oficial BCE (tipo depósito) | Diaria | 1999-01-01 a 2026-01-21 (9.884 obs) |
| UKBASERATE | Bank Rate BoE | Diaria | 1999-01-01 a 2026-01-21 |
| FFR | Federal Funds Rate Fed | Diaria | 1999-01-01 a 2026-01-21 |
| EUROSTER | EUR €STR / EONIA | Diaria | 1999-01-01 a 2026-01-21 (muchos ceros iniciales) |
| SONIA | GBP SONIA | Diaria | 1999-01-01 a 2026-01-21 |
| SOFR | USD SOFR | Diaria | 1999-01-01 a 2026-01-21 (ceros hasta 2018-04-02) |

**Nota:** Estructura transpuesta (variables en filas, fechas en columnas). Requiere transposición.

#### Hoja: INFLACIÓN
| Variable | Período | Países |
|----------|---------|--------|
| Inflación IPC (% var. interanual) | 1960–2025 | 22 (20 Eurozona + EEUU + Reino Unido) |

#### Hoja: DESEMPLEO
| Variable | Período | Países |
|----------|---------|--------|
| Tasa de desempleo (%) | 1960–2025 | 22 |

#### Hoja: PIB A PRECIO CONST
| Variable | Período | Países |
|----------|---------|--------|
| PIB USD constantes 2015 | 1960–2025 | 22 |

#### Hoja: PIB PER CAPITA PREC COST
| Variable | Período | Países |
|----------|---------|--------|
| PIB per cápita USD constantes 2015 | 1960–2025 | 22 |

#### Hoja: Materias primas
| Variable | Código | Unidad | Período |
|----------|--------|--------|---------|
| Petróleo crudo promedio | CRUDE_PETRO | $/bbl | 1993M01–2026M04 (mensual) |
| Petróleo Brent | CRUDE_BRENT | $/bbl | 1993M01–2026M04 |
| Petróleo Dubai | CRUDE_DUBAI | $/bbl | 1993M01–2026M04 |
| Petróleo WTI | CRUDE_WTI | $/bbl | 1993M01–2026M04 |
| Gas natural EE.UU. | NGAS_US | $/mmbtu | 1993M01–2026M04 |
| Gas natural Europa | NGAS_EUR | $/mmbtu | 1993M01–2026M04 |
| Gas natural Japón (LNG) | NGAS_JP | $/mmbtu | 1993M01–2026M04 |
| Índice gas natural | iNATGAS | 2010=100 | 1993M01–2026M04 |
| Oro | GOLD | $/troy oz | 1993M01–2026M04 |
| Plata | SILVER | $/troy oz | 1993M01–2026M04 |

Fuente: World Bank Commodity Price Data (Pink Sheet). Actualizado 04/05/2026. ~401 observaciones mensuales.

#### Hoja: Indices materias primas
| Variable | Código | Base | Período |
|----------|--------|------|---------|
| Índice total | iOVERALL | 2010=100 | 1960M01–2026M04 (~797 obs) |
| Energía | iENERGY | 2010=100 | " |
| No energía | iNONFUEL | 2010=100 | " |
| Agricultura | iAGRICULTURE | 2010=100 | " |
| Bebidas | iBEVERAGES | 2010=100 | " |
| Alimentación | iFOOD | 2010=100 | " |
| Grasas y aceites | iFATS_OILS | 2010=100 | " |
| Granos | iGRAINS | 2010=100 | " |
| Fertilizantes | iFERTILIZERS | 2010=100 | " |
| Metales y minerales | iMETMIN | 2010=100 | " |
| Metales base | iBASEMET | 2010=100 | " |

**Problemas detectados en DATASET.xlsx:**
- Hoja "I": vacía, sin datos.
- Estructura transpuesta en hojas macro (filas=variables, columnas=fechas) — poco convencional.
- SOFR y EUROSTER con ceros históricos (ver detalle en Archivo 2).
- Materias primas en USD nominales, no deflactadas.

---

### Archivo 7 — `Fuentes.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 16 KB |
| Hojas | Referencias por país, Resumen ratings |

#### Hoja: Resumen ratings (calidad crediticia actual, 2026)

| País | Moody's | S&P | Fitch | Outlook | Grupo calidad | Nivel riesgo |
|------|---------|-----|-------|---------|---------------|--------------|
| Alemania | Aaa | AAA | AAA | Estable | Muy alta | Muy bajo |
| Países Bajos | Aaa | AAA | AAA | Estable | Muy alta | Muy bajo |
| Luxemburgo | Aaa | AAA | AAA | Estable | Muy alta | Muy bajo |
| Irlanda | Aa3 | AA+ | AA- | Estable/Positiva | Muy alta | Muy bajo |
| Austria | Aa1 | AA+ | AA+ | Estable/Negativa | Muy alta | Muy bajo |
| Finlandia | Aa1 | AA+ | AA+ | Negativa | Muy alta | Muy bajo |
| Estados Unidos | Aa1 | AA+ | AA+ | Estable | Muy alta | Muy bajo |
| Reino Unido | Aa3 | AA | AA- | Estable | Alta | Bajo |
| Francia | Aa3 | A+ | A+ | Negativa | Media | Moderado |
| Italia | Baa2 | BBB+ | BBB | Estable/Positiva | Media | Moderado |
| España | A3 | A+ | A- | Estable | Media | Moderado |
| Grecia | Baa3 | BBB | BB+ | Estable/Positiva | Vulnerable | Elevado |

(22 países en total)

**Problema crítico:** Solo contiene los ratings **actuales (2026)**. No existe serie histórica de ratings. Para el modelo se deberá usar una escala numérica estática o buscar series históricas externas.

---

### Archivo 8 — `PIB_1993_2024_Analisis.xlsx`
| Campo | Detalle |
|-------|---------|
| Tamaño | 102 KB |
| Hojas | Resumen, Datos_Origen_USD, Datos_EUR_Paises, Bloques_EUR, Crecimiento_Paises, Crecimiento_Bloques, Estadisticos, Volatilidad, Graficas |
| **Frecuencia** | Anual |
| **Período** | 1993–2024 (32 años) |
| **Regiones** | 20 países Eurozona + EEUU + UK |

**Variables por hoja:**

| Hoja | Variable | Unidad |
|------|----------|--------|
| Datos_Origen_USD | PIB por país | USD constantes 2015 |
| Datos_EUR_Paises | PIB por país | EUR constantes 2015 (convertido) |
| Bloques_EUR | PIB agregado Eurozona, EEUU, UK | EUR constantes 2015 |
| Crecimiento_Paises | Crecimiento interanual PIB por país | % |
| Crecimiento_Bloques | Crecimiento interanual PIB por bloque | % |
| Volatilidad | Desviación típica móvil 5 años por bloque | % |

---

## 3. Inventario de documentos markdown en /documentos_markdown

### `Analisis calidad crediticia.md`
- **Contenido:** Análisis cualitativo y tabla comparativa de ratings soberanos de 22 países (Eurozona, EEUU, RU) a mayo 2026.
- **Uso en modelo:** Base para asignar la escala numérica de calidad crediticia (scoring) y clasificar países por grupo.
- **Datos numéricos:** Solo ratings actuales (no serie histórica).

### `analisis_vix_vstoxx_vftse.md`
- **Contenido:** Definición, evolución histórica narrativa, niveles de referencia y contexto de crisis para VIX, VSTOXX y VFTSE.
- **Uso en modelo:** Marco conceptual para las reglas de scoring de volatilidad financiera.
- **Datos numéricos:** Solo niveles de referencia cualitativos (rangos, máximos históricos). **No contiene serie histórica numérica.**

---

## 4. Resumen de coberturas por variable del modelo

| Variable del modelo | Disponible | Frecuencia | Período | Archivo origen | Observaciones |
|---------------------|-----------|------------|---------|----------------|---------------|
| **VARIABLE DEPENDIENTE** | | | | | |
| Tipo de interés objetivo a largo plazo (bono 10a) | **NO** | — | — | — | **GAP CRÍTICO: Falta la variable dependiente del modelo** |
| **VARIABLES PRINCIPALES** | | | | | |
| Inflación | SÍ | Anual | 1960–2025 | DATASET / PIBpc_Inflacion | Sólido, 22 países |
| PIB real | SÍ | Anual | 1960–2025 | DATASET / PIB_Analisis | Sólido, 22 países |
| Crecimiento interanual PIB | SÍ | Anual | 1993–2024 | PIB_Analisis (Crecimiento_Bloques) | Calculado sobre PIB real |
| PIB per cápita | SÍ | Anual | 1960–2025 | DATASET / PIBpc_Inflacion | Sólido, 22 países |
| Crecimiento interanual PIBpc | SÍ | Anual | 1993–2024 | PIBpc_Inflacion (CRECIMIENTO) | Calculado |
| Tasa de desempleo | SÍ | Anual | 1960–2025 | DATASET / Desempleo | Sólido, 22 países |
| Variación interanual desempleo | SÍ | Anual | 1993–2024 | Analisis_Desempleo | Calculado |
| Deuda pública (% PIB) | **NO** | — | — | — | **GAP: Sin datos históricos** |
| Tipo oficial banco central | SÍ | Diaria/Anual | 1999–2026 | DATASET / Tipos_Interes | BCE(Depo+MRO), BoE, Fed |
| Tipo OIS (EUR €STR) | Parcial | Diaria | 1999–2026 | DATASET / OIS | Ceros hasta 2022 |
| Tipo OIS (SONIA) | SÍ | Diaria | 1999–2026 | DATASET / OIS | Datos reales desde 1999 |
| Tipo OIS (SOFR) | Parcial | Diaria | 2018–2026 | DATASET / OIS | Ceros hasta 2018 |
| **VARIABLES FINANCIERAS** | | | | | |
| Tipos a corto/medio/largo plazo (yields) | **NO** | — | — | — | **GAP: Solo tipos oficiales, no yields de mercado** |
| **VARIABLES DE RIESGO** | | | | | |
| Rating Moody's / S&P / Fitch | Solo actual | Estático | 2026 | Fuentes.xlsx | Sin histórico |
| VIX | **NO** | — | — | — | Solo documento explicativo |
| VSTOXX | **NO** | — | — | — | Solo documento explicativo |
| VFTSE | **NO** | — | — | — | Solo documento explicativo |
| Precio petróleo (Brent/WTI) | SÍ | Mensual | 1993–2026 | DATASET (Materias primas) | Sólido |
| Precio gas natural (Europa) | SÍ | Mensual | 1993–2026 | DATASET (Materias primas) | NGAS_EUR |
| Precio oro | SÍ | Mensual | 1993–2026 | DATASET (Materias primas) | GOLD |
| Precio plata | SÍ | Mensual | 1993–2026 | DATASET (Materias primas) | SILVER |
| Ratio oro/plata | Calculable | Mensual | 1993–2026 | Calculado (GOLD/SILVER) | Derivada |
| Índices materias primas | SÍ | Mensual | 1960–2026 | DATASET (Indices MP) | 15 índices disponibles |

---

## 5. Gaps críticos identificados

### Gap 1 — Variable dependiente ausente (CRÍTICO)
**Problema:** El modelo debe predecir el **tipo de interés a largo plazo** (rendimiento del bono soberano a 10 años). Este dato no existe en ningún archivo de la carpeta local.

**Impacto:** Sin la variable dependiente no es posible entrenar el modelo de regresión.

**Solución sugerida:** Obtener la serie histórica de:
- **Eurozona:** Bono alemán Bund a 10 años (Bundesbank / BCE / FRED) — referencia soberana AAA del área euro.
- **EE.UU.:** US Treasury 10-Year Note Yield (FRED, ticker `GS10`).
- **Reino Unido:** UK Gilt 10-Year Yield (BoE / FRED).

Periodo necesario: mínimo 1993–2025, frecuencia anual y mensual.

### Gap 2 — Deuda pública ausente
**Problema:** No hay datos históricos de deuda pública (% PIB) para ninguna región.

**Impacto:** Variable con peso del 8% en el modelo; su ausencia reduce la calidad del scoring.

**Solución sugerida:** Obtener de FMI (World Economic Outlook), Banco Mundial o Eurostat. Variable: General Government Debt (% of GDP), 1993–2025.

### Gap 3 — VIX/VSTOXX/VFTSE sin datos numéricos
**Problema:** Solo existe un documento explicativo. No hay serie histórica de valores.

**Impacto:** Variable con peso del 5% en el modelo.

**Solución sugerida:**
- VIX: CBOE, FRED (ticker `VIXCLS`), Investing.com — disponible desde 1990.
- VSTOXX: STOXX Ltd., Investing.com — disponible desde 1999.
- VFTSE: Investing.com — disponible desde 2000.

### Gap 4 — Ratings crediticios sin serie histórica
**Problema:** Solo disponibles los ratings actuales (2026). El modelo necesita la evolución histórica para el scoring temporal.

**Impacto:** Variable con peso del 7% en el modelo.

**Solución sugerida:** Para el modelo inicial se puede usar el rating actual como variable estática (supuesto simplificador). Para el modelo completo: Rating History de Moody's, S&P y Fitch (disponibles en sus webs o plataformas de datos financieros).

### Gap 5 — OIS con datos incompletos
**Problema:** EUR €STR tiene ceros hasta 2022 (existe desde 2019 pero este archivo no tiene datos previos). SOFR tiene ceros hasta 2018.

**Solución sugerida:**
- EUR €STR: puede complementarse con EONIA (su predecesor) desde 1999. Disponible en BCE.
- SOFR: puede complementarse con Fed Funds Effective Rate (ya disponible en DATASET) para el período anterior a 2018.

---

## 6. Matriz resumen de calidad de datos

| Archivo | Variables clave | Período real útil | Calidad | Uso en modelo |
|---------|----------------|-------------------|---------|---------------|
| Analisis_Desempleo_1993_2024.xlsx | Desempleo por bloque | 1993–2024 | Alta | Scoring + Regresión |
| Analisis_OIS_EUR_USD_GBP.xlsx | SOFR, SONIA, €STR | SONIA: 1999–2026 / SOFR: 2018–2026 | Media (gaps) | Scoring + Regresión |
| Analisis_PIBpc_Inflacion_Eurozona_EEUU_UK_1993-2024.xlsx | PIBpc, Inflación, PIB | 1993–2024 | Alta | Scoring + Regresión |
| Analisis_Tipos_Interes.xlsx | Tipos oficiales BCE/BoE/Fed | 1999–2026 | Alta | Scoring + Regresión |
| Anexo_Inflacion_Calculos.xlsx | Inflación (auxiliar) | 1993–2025 | Media (encoding) | Complementario |
| DATASET.xlsx | Variables macro + MP | Macro: 1960–2025 / MP: 1993–2026 | Alta | Master dataset |
| Fuentes.xlsx | Ratings actuales | 2026 solo | Baja (sin histórico) | Scoring estático |
| PIB_1993_2024_Analisis.xlsx | PIB bloques + crecimiento | 1993–2024 | Alta | Scoring + Regresión |
| Analisis calidad crediticia.md | Ratings cualitativos | 2026 actual | Informativa | Escala numérica scoring |
| analisis_vix_vstoxx_vftse.md | Marco VIX/VSTOXX/VFTSE | N/A (cualitativo) | Informativa | Reglas de scoring |

---

## 7. Período común de datos disponibles

El período en el que **todas las variables principales disponibles** tienen cobertura simultánea es:

**1999–2024** (anual) o **1999–2024** (mensual con interpolación)

Limitaciones:
- Los tipos oficiales y OIS comienzan en 1999 (creación del BCE/Euro).
- Los datos del DATASET macro cubren desde 1960, pero el período relevante para el modelo es 1993–2024.
- Las materias primas cubren desde 1993M01 en frecuencia mensual.

**Período de entrenamiento recomendado (sujeto a obtención de gaps):** 1999–2024 (26 años).

---

## 8. Próximos pasos recomendados (Fase 2)

1. **Obtener la variable dependiente** (yields 10 años Bund, US Treasury, UK Gilt) antes de proceder.
2. Obtener datos de **deuda pública** (FMI/WEO).
3. Obtener series históricas de **VIX** (mínimo mensual, 1993–2025).
4. Completar **EUR €STR/EONIA** para el período 1999–2021.
5. Transformar todos los datos a **estructura longitudinal** (filas = fecha × país, columnas = variables).
6. Construir las estructuras `Datos_Anuales` y `Datos_Mensuales` según CLAUDE.md.
