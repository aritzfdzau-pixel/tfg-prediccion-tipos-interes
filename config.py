"""
config.py — Configuración centralizada del proyecto TFG
Modelo predictivo de tipos de interés: BCE, BoE, FED
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "Datos originales"
PROCESSED_DIR = ROOT_DIR / "Datos procesados"
MODEL_DIR = ROOT_DIR / "Modelo"
RESULTS_DIR = ROOT_DIR / "Resultados"
SCENARIOS_DIR = ROOT_DIR / "escenarios"

DATASET_PATH = DATA_DIR / "DATASET.xlsx"
VIX_MENSUAL_PATH = DATA_DIR / "VIX_mensual_1993_2025.csv"
RATINGS_PATH = DATA_DIR / "Ratings_Numericos_Estaticos.csv"

# ---------------------------------------------------------------------------
# Índices de hojas en DATASET.xlsx
# IMPORTANTE: usar índices numéricos, NO nombres de hoja.
# La hoja de inflación contiene caracteres especiales (Ó) que se corrompen
# según la codificación del sistema. El índice es siempre seguro.
# ---------------------------------------------------------------------------
SHEET_INDICES = {
    "bancos_centrales": 0,   # 'Bancos centrales'
    "inflacion":        2,   # 'INFLACIÓN'
    "desempleo":        3,   # 'DESEMPLEO'
    "pib":              4,   # 'PIB A PRECIO CONST'
    "pibpc":            5,   # 'PIB PER CAPITA PREC COST'
    "materias_primas":  6,   # 'Materias primas'
}

# ---------------------------------------------------------------------------
# Unidades de los tipos de interés en DATASET.xlsx
# Los tipos están en forma DECIMAL: 0.02 = 2%, 0.0625 = 6.25%.
# data_loader.py multiplica ×100 al cargar para pasar a PORCENTAJE.
# Todos los cálculos del proyecto trabajan en porcentaje (%, no decimal).
# El umbral THRESHOLD_ESTABILIDAD está en puntos porcentuales (pp).
# ---------------------------------------------------------------------------
RATES_SCALE_FACTOR = 100

# Columnas de DATASET.xlsx que ya están en PORCENTAJE (no necesitan ×100).
# Anomalía del archivo fuente: SOFR está en % (e.g. 3.65 = 3.65%) mientras
# el resto de tipos están en decimal (e.g. 0.0364 = 3.64%).
RATES_ALREADY_PCT = ["SOFR"]

# ---------------------------------------------------------------------------
# Bancos centrales
# ---------------------------------------------------------------------------
BANCOS_CENTRALES = ["BCE", "BoE", "FED"]

# Nombres REALES de las columnas en DATASET.xlsx tras transponer "Bancos centrales"
# ATENCIÓN: "ECB Depo Facility" lleva espacio (no guión bajo); "UKBASERATE" es todo junto.
TARGET_COLUMNS = {
    "BCE": "ECB Depo Facility",   # con espacio — nombre real en el Excel
    "BoE": "UKBASERATE",          # nombre real en el Excel
    "FED": "FFR",
}

# Nombres reales de las series OIS en bruto (antes de imputación)
OIS_RAW_COLUMNS = {
    "BCE": "EUROSTER",   # EUR €STR; zeros antes de sep-2022
    "BoE": "SONIA",      # SONIA; cobertura completa desde 1999
    "FED": "SOFR",       # SOFR; zeros antes de abr-2018
}

# Columnas OIS unificadas (construidas en data_loader tras imputación)
OIS_COLUMNS = {
    "BCE": "OIS_EUR",
    "BoE": "OIS_GBP",
    "FED": "OIS_USD",
}

# Nombre del instrumento OIS por banco (usado para nombrar archivos del modelo OIS)
# Nota: la columna en el DataFrame siempre se llama "OIS" tras la imputación.
# Estos nombres son solo para identificación externa (archivos, labels, dashboard).
OIS_INSTRUMENT_NAMES = {
    "BCE": "ESTR",    # €STR — European Short-Term Rate
    "BoE": "SONIA",   # Sterling Overnight Index Average
    "FED": "SOFR",    # Secured Overnight Financing Rate
}

# Mapeo nombre de instrumento → columna real en el DataFrame cargado.
# Permite usar instrumento="ESTR"/"SONIA"/"SOFR" en el pipeline sin cambiar
# el nombre interno de la columna "OIS" que genera data_loader.
INSTRUMENT_TO_COL = {
    "tipo_oficial": "tipo_oficial",
    "OIS":  "OIS",    # nombre genérico (backward compat)
    "ESTR": "OIS",    # €STR → columna OIS del BCE
    "SONIA": "OIS",   # SONIA → columna OIS del BoE
    "SOFR": "OIS",    # SOFR  → columna OIS del FED
}

# Ratings estáticos (escala 1–10, valor de 2026 usado como constante)
RATINGS = {
    "BCE": 8.09,
    "BoE": 8.50,
    "FED": 9.50,
}

# ---------------------------------------------------------------------------
# Variables de materias primas — nombres reales en hoja "Materias primas"
# ---------------------------------------------------------------------------
GOLD_COL   = "GOLD"
SILVER_COL = "SILVER"

# Gas: se usa columna diferente según zona económica
# BCE/BoE → gas natural europeo (NGAS_EUR); FED → Henry Hub americano (NGAS_US)
GAS_COLUMNS = {
    "BCE": "NGAS_EUR",
    "BoE": "NGAS_EUR",
    "FED": "NGAS_US",
}

# Mapeo nombre genérico del modelo → nombre real en los DataFrames cargados
# OIS y Gas son dinámicos (dependen del banco); el resto es fijo.
COLUMN_MAP = {
    "Oro":   GOLD_COL,
    "Plata": SILVER_COL,
    "VIX":   "VIX",       # columna del CSV de VIX tras merge
    "OIS":   None,         # dinámico: OIS_COLUMNS[banco]
    "Gas":   None,         # dinámico: GAS_COLUMNS[banco]
}

# ---------------------------------------------------------------------------
# Países por zona económica
# Usados para agregar datos anuales (PIB, PIBpc, Inflación, Desempleo)
# Los nombres deben coincidir exactamente con la columna 'Codigo del pais'
# (columna índice 1) en las hojas del World Bank.
# ---------------------------------------------------------------------------
ZONA_CODIGOS = {
    "BCE": ["DEU", "FRA", "ITA", "ESP", "NLD"],   # Top-5 economías del euro
    "BoE": ["GBR"],                                 # Reino Unido
    "FED": ["USA"],                                 # Estados Unidos
}

# Pesos PIB para la media ponderada de la Eurozona (se renormalizan a 1.0)
# Aproximación basada en participación en el PIB de la zona euro ~2022
PESOS_EUROZONA = {
    "DEU": 0.30,
    "FRA": 0.20,
    "ITA": 0.15,
    "ESP": 0.11,
    "NLD": 0.07,
}

# Nombres de países tal como aparecen en la columna 0 (País) de las hojas —
# usados como fallback si la búsqueda por código no funciona
ZONA_NOMBRES_PAIS = {
    "BoE": "Reino Unido",
    "FED": "EEUU",
}

# ---------------------------------------------------------------------------
# Variables explicativas del modelo (nombres internos, tras renombrado)
# ---------------------------------------------------------------------------
VARIABLES_MODELO = [
    "PIB_growth",      # Tasa de crecimiento interanual del PIB (%), forward-filled
    "PIBpc_growth",    # Tasa de crecimiento interanual del PIB per cápita (%), forward-filled
    "Inflacion",       # Inflación (%), forward-filled mensual
    "Desempleo",       # Tasa de desempleo (%), forward-filled mensual
    "VIX",             # Índice de volatilidad implícita, mensual
    "Oro",             # Precio del oro (USD/oz), mensual
    "Plata",           # Precio de la plata (USD/oz), mensual
    "Gas",             # Precio del gas (NGAS_EUR o NGAS_US según banco), mensual
    "OIS_taylor",      # Regla de Taylor: r* + 1.5*(π-π*) + 0.5*(y-y*). Tipo neutral
                       # calculado en feature_engineering._calcular_OIS_taylor().
    "brecha_taylor",   # Brecha política = OIS_taylor_t - tipo_oficial_t (pp).
                       # Positiva → tipos por debajo del equilibrio (presión alcista).
                       # Negativa → tipos por encima del equilibrio (presión bajista).
                       # Calculada en feature_engineering.preparar_features().
    "meses_sin_cambio", # Inercia de política: meses consecutivos sin mover el tipo.
                        # Señal de pausa sostenida: valores altos → el banco lleva
                        # tiempo sin actuar → probabilidad de cambio baja.
                        # Calculada en feature_engineering.preparar_features().
    "delta_tipo_lag1",  # Momentum: cambio del tipo el mes anterior (pp).
                        # Señal de dirección reciente: negativo → ciclo de bajadas;
                        # positivo → ciclo de subidas; cero → pausa.
                        # Calculada en feature_engineering.preparar_features().
]
# NOTA: Rating (calificación crediticia estática) fue eliminado del modelo en la
# Fase A de mejoras al presentar importancia = 0% en los tres bancos centrales.
# Al ser constante por banco (mismo valor todos los meses), el Random Forest
# nunca puede usarlo para dividir datos — no aporta información discriminativa.

# ---------------------------------------------------------------------------
# Parametros de la Regla de Taylor por banco central
# OIS_taylor = max(0, r_star + 1.5*(π - pi_star) + 0.5*(PIB_growth - y_star))
#
# r_star  : tipo de interes neutral real estimado (%)
# pi_star : objetivo de inflacion del banco (%)
# y_star  : tasa de crecimiento potencial del PIB (%)
#
# Fuentes de calibracion:
#   BCE: Holston-Laubach-Williams (2023) r*~1.0%; objetivo inflacion BCE 2%
#   BoE: Bank Underground (2022) r*~1.5%; objetivo inflacion BoE 2%
#   FED: Laubach-Williams (2023) r*~2.5%; objetivo inflacion FED 2%
# ---------------------------------------------------------------------------
TAYLOR_PARAMS = {
    "BCE": {"r_star": 1.0, "pi_star": 2.0, "y_star": 1.5},
    "BoE": {"r_star": 1.5, "pi_star": 2.0, "y_star": 2.0},
    "FED": {"r_star": 2.5, "pi_star": 2.0, "y_star": 2.0},
}

# ---------------------------------------------------------------------------
# División temporal
# ---------------------------------------------------------------------------
TRAIN_END   = "2023-12-31"   # Último mes de entrenamiento
TEST_START  = "2024-01-01"   # Primer mes de test
TEST_END    = "2025-12-31"   # Último mes de test
PREDICT_START = 2026          # Primer año de predicción futura
PREDICT_END   = 2030          # Último año de predicción futura

# ---------------------------------------------------------------------------
# Parámetros del modelo de clasificación
# ---------------------------------------------------------------------------
# Umbral mensual en puntos porcentuales (pp).
# Justificación: el paso mínimo de los bancos centrales es 0.25 pp.
# Cualquier variación mensual < 0.125 pp se clasifica como "Estable".
# ASUME que los tipos ya están en porcentaje (tras conversión ×100 en data_loader).
THRESHOLD_ESTABILIDAD = 0.125

CLASES = ["Baja", "Estable", "Sube"]   # orden fijo para matrices de confusión

# ---------------------------------------------------------------------------
# Selección de modelo de regresión
# ---------------------------------------------------------------------------
REGRESSOR = "random_forest"   # "random_forest" (principal) o "ridge" (comparativo)

# Hiperparámetros Random Forest Regressor
RF_N_ESTIMATORS    = 200
RF_MAX_DEPTH       = 6
RF_MIN_SAMPLES_LEAF = 10
RF_RANDOM_STATE    = 42
RF_MAX_FEATURES    = 'sqrt'   # sqrt(10)≈3 features por split → evita dominio del VIX
                               # y fuerza uso equilibrado de variables macro fundamentales

# Hiperparámetros Logistic Regression (clasificación)
LR_SOLVER      = "lbfgs"
LR_MAX_ITER    = 1000
LR_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Colores del dashboard (uno por banco central)
# ---------------------------------------------------------------------------
COLORES = {
    "BCE": "#003399",   # azul BCE
    "BoE": "#CC0000",   # rojo BoE
    "FED": "#228B22",   # verde FED
}
