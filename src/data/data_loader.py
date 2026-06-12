"""
data_loader.py — Carga y limpieza de datos históricos (Fase 2)

NOTAS DE ESTRUCTURA DEL DATASET.xlsx
-------------------------------------
Hoja "Bancos centrales" (sheet 0):
  - TRANSPUESTA: las variables son FILAS, las fechas son COLUMNAS.
  - Shape bruta: (8 filas × ~9885 columnas).
  - Fila 0: vacía. Fila 1: "Fecha " + fechas diarias. Filas 2–7: variables.
  - Requiere transposición explícita antes de usar como DataFrame normal.
  - Las tasas están en DECIMAL (0.02 = 2%). Multiplicar × config.RATES_SCALE_FACTOR.

Hoja "Materias primas" (sheet 6):
  - Formato normal (fechas = filas). Saltar las 6 primeras filas descriptivas.
  - Las fechas están en formato texto "1993M01" → requiere parse_fecha_mensual().

Hojas PIB / PIBpc / Inflación / Desempleo (sheets 4, 5, 2, 3):
  - Formato wide: filas = países, columnas = años (1960–2025).
  - Usar índices numéricos (config.SHEET_INDICES) para evitar problemas de encoding.
  - BCE: media ponderada de DEU, FRA, ITA, ESP, NLD (top-5 zona euro).
  - BoE: GBR. FED: USA.

Punto de entrada principal: cargar_dataset_completo(banco)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def parse_fecha_mensual(fecha_str: str) -> pd.Timestamp:
    """Convierte '1993M01' a pd.Timestamp('1993-01-01')."""
    s = str(fecha_str).strip()
    anio = int(s[:4])
    mes = int(s[5:7])
    return pd.Timestamp(year=anio, month=mes, day=1)


def _extraer_columnas_anio(df: pd.DataFrame, anio_min: int = 1993) -> dict:
    """
    Devuelve {col_original: anio_int} para columnas que representan años >= anio_min.
    Maneja columnas como '1960', '1960.0', 1960 y 1960.0 indistintamente.
    """
    resultado = {}
    for col in df.columns:
        try:
            anio = int(float(str(col)))
            if anio >= anio_min:
                resultado[col] = anio
        except (ValueError, TypeError):
            pass
    return resultado


# ---------------------------------------------------------------------------
# Cargadores por fuente
# ---------------------------------------------------------------------------

def cargar_bancos_centrales() -> pd.DataFrame:
    """
    Lee la hoja "Bancos centrales" (sheet 0) de DATASET.xlsx.

    La hoja está TRANSPUESTA: variables como filas, fechas como columnas.
      - Fila 0: vacía.
      - Fila 1: "Fecha " en col 0 + fechas diarias en cols 1+.
      - Filas 2–7: nombre de variable en col 0 + valores diarios en cols 1+.

    Devuelve:
        DataFrame diario con columnas:
        "ECB Depo Facility", "UKBASERATE", "FFR", "EUROSTER", "SONIA", "SOFR"
        Valores en PORCENTAJE (ya multiplicados × RATES_SCALE_FACTOR).
        Índice: DatetimeIndex diario, 1999-01-01 … 2026-01-21.
    """
    df_raw = pd.read_excel(
        config.DATASET_PATH,
        sheet_name=config.SHEET_INDICES["bancos_centrales"],
        header=None,
        engine="openpyxl",
    )

    # Fila 1: fechas (col 0 tiene "Fecha ", cols 1+ tienen las fechas)
    fechas = pd.to_datetime(df_raw.iloc[1, 1:].values)

    # Filas 2+: col 0 = nombre variable, cols 1+ = valores diarios
    nombres = [str(v).strip() for v in df_raw.iloc[2:, 0].values]
    datos = df_raw.iloc[2:, 1:].apply(pd.to_numeric, errors="coerce").values

    df = pd.DataFrame(datos.T, index=fechas, columns=nombres)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Fecha"

    # Decimal → porcentaje (0.02 → 2.0 %).
    # EXCEPCIÓN: SOFR ya está en porcentaje en el archivo fuente (3.65 = 3.65%);
    # el resto de columnas están en decimal (0.0364 = 3.64%).
    cols_decimal = [c for c in df.columns if c not in config.RATES_ALREADY_PCT]
    df[cols_decimal] = df[cols_decimal] * config.RATES_SCALE_FACTOR

    return df


def cargar_materias_primas() -> pd.DataFrame:
    """
    Lee la hoja "Materias primas" (sheet 6) de DATASET.xlsx.

    Salta las 6 primeras filas de metadatos. La columna 0 contiene fechas
    en formato "1993M01", que se parsean con parse_fecha_mensual().

    Devuelve:
        DataFrame mensual con columnas: CRUDE_PETRO, CRUDE_BRENT, CRUDE_DUBAI,
        CRUDE_WTI, NGAS_US, NGAS_EUR, NGAS_JP, iNATGAS, GOLD, SILVER.
        Índice: DatetimeIndex mensual (primer día del mes), 1993-01-01 … 2026-04-01.
    """
    df = pd.read_excel(
        config.DATASET_PATH,
        sheet_name=config.SHEET_INDICES["materias_primas"],
        skiprows=6,
        header=0,
        engine="openpyxl",
    )

    # Columna 0 tiene fechas en formato "1993M01"
    col_fecha = df.columns[0]
    df[col_fecha] = df[col_fecha].apply(parse_fecha_mensual)
    df = df.rename(columns={col_fecha: "Fecha"}).set_index("Fecha")
    df.index = pd.DatetimeIndex(df.index)

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def cargar_macro_anual(tipo: str) -> pd.DataFrame:
    """
    Lee una hoja de datos macroeconómicos anuales del World Bank.

    Parámetro tipo: "pib", "pibpc", "inflacion" o "desempleo".
    Usa config.SHEET_INDICES[tipo] para evitar problemas de encoding
    en el nombre de la hoja de inflación (Ó).

    Devuelve:
        DataFrame con columnas ["codigo", 1993, 1994, ..., 2025] (años como int).
        Valores numéricos; NaN donde no hay dato.
    """
    sheet_idx = config.SHEET_INDICES[tipo]
    df = pd.read_excel(
        config.DATASET_PATH,
        sheet_name=sheet_idx,
        skiprows=3,
        header=0,
        engine="openpyxl",
    )

    # Las 3 primeras columnas son: País, Codigo del pais, Indicador
    cols = df.columns.tolist()
    df = df.rename(columns={cols[0]: "pais", cols[1]: "codigo", cols[2]: "indicador"})

    # Extraer solo columnas de año ≥ 1993
    mapa_anios = _extraer_columnas_anio(df, anio_min=1993)
    df = df[["codigo"] + list(mapa_anios.keys())].copy()
    df = df.rename(columns=mapa_anios)   # Renombrar a enteros: '1993' → 1993

    # Asegurar valores numéricos
    anio_cols = list(mapa_anios.values())
    df[anio_cols] = df[anio_cols].apply(pd.to_numeric, errors="coerce")

    # El World Bank almacena 0 como "sin datos" para años recientes no publicados.
    # PIB, PIBpc, inflación y desempleo nunca son literalmente 0 en economías avanzadas.
    # Reemplazar 0 por NaN para que forward_fill_anual los propague desde el último año válido.
    df[anio_cols] = df[anio_cols].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Transformaciones de datos
# ---------------------------------------------------------------------------

def agregar_zona_economica(df_macro: pd.DataFrame, banco: str) -> pd.Series:
    """
    Agrega datos de países al nivel de zona económica.

    BCE: media ponderada de DEU, FRA, ITA, ESP, NLD (pesos en config.PESOS_EUROZONA,
         renormalizados a 1.0 si algún país no tiene dato para ese año).
    BoE: selección directa de GBR.
    FED: selección directa de USA.

    Devuelve:
        pd.Series con índice = año (int) y valor = dato agregado de la zona.
    """
    codigos = config.ZONA_CODIGOS[banco]
    df_zona = df_macro[df_macro["codigo"].isin(codigos)].copy()
    anio_cols = [c for c in df_zona.columns if isinstance(c, int)]

    if banco == "BCE":
        pesos = config.PESOS_EUROZONA
        resultado = {}
        for anio in anio_cols:
            total_peso = 0.0
            total_valor = 0.0
            for codigo in codigos:
                if codigo not in pesos:
                    continue
                fila = df_zona[df_zona["codigo"] == codigo]
                if fila.empty:
                    continue
                valor = float(fila[anio].values[0])
                if pd.isna(valor):
                    continue
                total_peso += pesos[codigo]
                total_valor += pesos[codigo] * valor
            resultado[anio] = total_valor / total_peso if total_peso > 0 else np.nan
        return pd.Series(resultado)

    else:
        # Single country
        codigo = codigos[0]
        fila = df_zona[df_zona["codigo"] == codigo]
        if fila.empty:
            return pd.Series(dtype=float)
        serie = fila.iloc[0][anio_cols].copy()
        serie.index = anio_cols
        return serie.astype(float)


def imputar_OIS(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye OIS_EUR, OIS_GBP, OIS_USD a partir de las series brutas.
    Opera sobre el DataFrame DIARIO de cargar_bancos_centrales().

    OIS_EUR: €STR cuando != 0 y no NaN; ECB Depo Facility en el resto.
    OIS_GBP: SONIA directamente (cobertura completa desde 1999).
    OIS_USD: SOFR cuando != 0 y no NaN; FFR en el resto.

    Justificación de las imputaciones: Decisiones 2 y 3 en
    Documentos/decisiones_metodologicas.md.
    """
    # OIS_EUR
    df["OIS_EUR"] = df["EUROSTER"].copy()
    mask_eur = (df["OIS_EUR"] == 0) | df["OIS_EUR"].isna()
    df.loc[mask_eur, "OIS_EUR"] = df.loc[mask_eur, "ECB Depo Facility"]

    # OIS_GBP — SONIA sin imputación
    df["OIS_GBP"] = df["SONIA"].copy()

    # OIS_USD
    df["OIS_USD"] = df["SOFR"].copy()
    mask_usd = (df["OIS_USD"] == 0) | df["OIS_USD"].isna()
    df.loc[mask_usd, "OIS_USD"] = df.loc[mask_usd, "FFR"]

    return df


def agregar_mensual(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Agrega las columnas indicadas de frecuencia diaria a media mensual.
    Usa resample("MS") → primer día de cada mes como índice.
    """
    return df[columnas].resample("MS").mean()


def forward_fill_anual(
    df_mensual: pd.DataFrame,
    serie_anual: pd.Series,
    col_destino: str,
) -> pd.DataFrame:
    """
    Replica el valor anual a los 12 meses del año correspondiente.

    Para cada mes del df_mensual, busca su año en serie_anual y asigna
    ese valor a col_destino. Resultado: mismo valor para todos los meses
    de un año (forward-fill estándar para variables anuales).

    Meses en años sin dato quedan como NaN.
    """
    anio_a_valor = serie_anual.to_dict()
    df_mensual[col_destino] = df_mensual.index.year.map(anio_a_valor)
    # Propagar el último valor disponible a años sin dato (p.ej. 2025 si WB aún no lo publicó).
    # Esto es el estándar en modelos macroeconómicos: usar la última observación disponible.
    df_mensual[col_destino] = df_mensual[col_destino].ffill()
    return df_mensual


# ---------------------------------------------------------------------------
# Enriquecimiento del dataset
# ---------------------------------------------------------------------------

def anadir_vix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lee VIX_mensual_1993_2025.csv y hace left join con df por fecha mensual.
    Meses sin dato VIX (ej. 2026) quedan como NaN.
    """
    vix = pd.read_csv(config.VIX_MENSUAL_PATH)
    vix["Fecha"] = pd.to_datetime(vix["Fecha"])
    vix = vix.set_index("Fecha")[["VIX"]]
    return df.join(vix, how="left")


def anadir_ratings(df: pd.DataFrame, banco: str) -> pd.DataFrame:
    """
    Añade la columna "Rating" con el valor estático de config.RATINGS[banco]
    replicado en todas las filas (Decisión 4 — rating estático de 2026).
    """
    df["Rating"] = config.RATINGS[banco]
    return df


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def validar_columnas(df: pd.DataFrame, banco: str) -> None:
    """
    Verifica que el DataFrame contiene todas las columnas obligatorias.

    Columnas requeridas (estado intermedio antes de feature_engineering):
    tipo_oficial, OIS, Oro, Plata, Gas, VIX, PIB, PIBpc, Inflacion,
    Desempleo, Rating.

    Lanza ValueError con detalle de columnas faltantes si no están todas.
    """
    requeridas = [
        "tipo_oficial", "OIS", "Oro", "Plata", "Gas", "VIX",
        "PIB", "PIBpc", "Inflacion", "Desempleo", "Rating",
    ]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Dataset [{banco}]: faltan columnas obligatorias → {faltantes}"
        )


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def guardar_datos_procesados(banco: str, df: pd.DataFrame) -> Path:
    """
    Guarda el dataset procesado de un banco central en Datos procesados/.
    Nombre del archivo: dataset_{banco}.csv
    Devuelve la ruta del archivo guardado.
    """
    ruta = config.PROCESSED_DIR / f"dataset_{banco}.csv"
    df.to_csv(ruta)
    return ruta


def cargar_datos_procesados(banco: str) -> pd.DataFrame:
    """
    Carga el dataset procesado previamente guardado en Datos procesados/.
    Más rápido que ejecutar cargar_dataset_completo() en cada ejecución.
    """
    ruta = config.PROCESSED_DIR / f"dataset_{banco}.csv"
    df = pd.read_csv(ruta, index_col="Fecha", parse_dates=True)
    return df


def cargar_dataset_completo(banco: str) -> pd.DataFrame:
    """
    Pipeline completo de carga para un banco central.

    Orden de operaciones:
      1. Carga tipos/OIS diarios de DATASET.xlsx (hoja Bancos centrales).
      2. Imputa OIS_EUR y OIS_USD (gaps pre-€STR / pre-SOFR).
      3. Agrega a media mensual (tipos oficiales + OIS del banco).
      4. Incorpora Gold, Silver, Gas desde hoja Materias primas.
      5. Incorpora VIX desde CSV.
      6. Carga PIB, PIBpc, Inflación, Desempleo anuales; agrega por zona;
         aplica forward-fill mensual.
      7. Añade Rating estático.
      8. Valida columnas obligatorias.

    Nota sobre enero 2026: los datos de Bancos centrales llegan hasta 2026-01-21.
    La media mensual de enero 2026 usa solo esos 21 días (parcial). Materias
    primas cubre hasta 2026-04 con datos completos.

    Parámetros:
        banco: "BCE", "BoE" o "FED"

    Devuelve:
        DataFrame mensual con DatetimeIndex (primer día del mes), periodo
        1999-01-01 en adelante. Columnas: tipo_oficial, OIS, Oro, Plata,
        Gas, VIX, PIB, PIBpc, Inflacion, Desempleo, Rating.
    """
    if banco not in config.BANCOS_CENTRALES:
        raise ValueError(
            f"Banco no reconocido: '{banco}'. Usar uno de {config.BANCOS_CENTRALES}"
        )

    # 1 & 2 — Tipos diarios + imputación OIS
    df_bc = cargar_bancos_centrales()
    df_bc = imputar_OIS(df_bc)

    # 3 — Agregar a mensual: tipo oficial + OIS del banco
    tipo_col = config.TARGET_COLUMNS[banco]
    ois_col  = config.OIS_COLUMNS[banco]    # OIS_EUR / OIS_GBP / OIS_USD
    df = agregar_mensual(df_bc, [tipo_col, ois_col])
    df = df.rename(columns={tipo_col: "tipo_oficial", ois_col: "OIS"})

    # 4 — Materias primas: Gold, Silver, Gas
    df_mat = cargar_materias_primas()
    gas_col = config.GAS_COLUMNS[banco]
    df_mat_sel = df_mat[[config.GOLD_COL, config.SILVER_COL, gas_col]].rename(
        columns={config.GOLD_COL: "Oro", config.SILVER_COL: "Plata", gas_col: "Gas"}
    )
    df = df.join(df_mat_sel, how="left")

    # 5 — VIX
    df = anadir_vix(df)

    # 6 — Variables macroeconómicas anuales → forward-fill mensual
    for tipo, col_destino in [
        ("pib",       "PIB"),
        ("pibpc",     "PIBpc"),
        ("inflacion", "Inflacion"),
        ("desempleo", "Desempleo"),
    ]:
        df_macro = cargar_macro_anual(tipo)
        serie_anual = agregar_zona_economica(df_macro, banco)
        df = forward_fill_anual(df, serie_anual, col_destino)

    # 7 — Rating estático
    df = anadir_ratings(df, banco)

    # 8 — Validación
    validar_columnas(df, banco)

    return df
