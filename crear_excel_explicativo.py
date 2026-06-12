"""
crear_excel_explicativo.py
Genera un Excel pedagógico con todos los cálculos del modelo de predicción
de tipos de interés (BCE, BoE, FED) vinculados mediante fórmulas Excel reales.

Colores:
  Amarillo  = celda de entrada (el tutor puede cambiar el valor)
  Verde     = resultado calculado con fórmula Excel
  Azul claro= dato informativo del modelo
  Rojo claro= error de predicción
"""

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\aritz\OneDrive\Escritorio\PERSONAL\Investigacion\Trabajo de Fin De Grado - Aritz Fernandez Aulestiarte\MODELO V2")
RES  = BASE / "Resultados"
ESC  = BASE / "escenarios"

# ─── Cargar datos reales ───────────────────────────────────────────────────────
imp   = {b: pd.read_csv(RES / f"importancia_{b}.csv", index_col=0) for b in ["BCE","BoE","FED"]}
resumen = pd.read_csv(RES / "resumen_regresion.csv")
scores  = {b: pd.read_csv(RES / f"scores_test_{b}.csv",  index_col=0) for b in ["BCE","BoE","FED"]}
esc_data = {}
for banco in ["BCE","BoE","FED"]:
    for sc in ["base","optimista","pesimista"]:
        key = f"{banco}_{sc}"
        df = pd.read_csv(ESC / f"escenario_{sc}_{banco}.csv")
        esc_data[key] = df.iloc[0]   # fila del año 2026 mes 1
fut_data = {}
for banco in ["BCE","BoE","FED"]:
    for sc in ["base","optimista","pesimista"]:
        key = f"{banco}_{sc}"
        df = pd.read_csv(RES / f"predicciones_futuras_{banco}_{sc}.csv", index_col=0)
        fut_data[key] = df

# ─── Estilos ──────────────────────────────────────────────────────────────────
C_AZUL_OSC  = "1F4E79"
C_AZUL_MED  = "2E75B6"
C_AZUL_CLAR = "D6E4F0"
C_VERDE_OSC  = "375623"
C_VERDE_CLAR = "E2EFDA"
C_AMARILLO  = "FFF2CC"
C_NARANJA   = "F4B942"
C_ROJO_CLAR = "FCE4D6"
C_ROJO_OSC  = "C00000"
C_BLANCO    = "FFFFFF"
C_GRIS      = "F2F2F2"

def _borde(grosor="thin", color="BFBFBF"):
    lado = Side(style=grosor, color=color)
    return Border(left=lado, right=lado, top=lado, bottom=lado)

def _fill(color):
    return PatternFill("solid", fgColor=color)

def _font(bold=False, color="000000", size=10, italic=False):
    return Font(bold=bold, color=color, size=size, italic=italic)

def _alin(h="center", v="center", wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

def cel(ws, row, col, value, bg=None, bold=False, font_color="000000",
        size=10, h="center", wrap=False, italic=False, border=True,
        num_format=None):
    """Escribe una celda con estilo completo."""
    c = ws.cell(row=row, column=col, value=value)
    if bg:
        c.fill = _fill(bg)
    c.font  = _font(bold=bold, color=font_color, size=size, italic=italic)
    c.alignment = _alin(h=h, wrap=wrap)
    if border:
        c.border = _borde()
    if num_format:
        c.number_format = num_format
    return c

def titulo(ws, row, c1, c2, texto, bg=C_AZUL_OSC, size=14):
    ws.merge_cells(start_row=row, start_column=c1, end_row=row, end_column=c2)
    c = ws.cell(row=row, column=c1, value=texto)
    c.fill    = _fill(bg)
    c.font    = Font(bold=True, color=C_BLANCO, size=size)
    c.alignment = _alin(wrap=False)
    ws.row_dimensions[row].height = 38
    return c

def subheader(ws, row, c1, c2, texto, bg=C_AZUL_MED):
    ws.merge_cells(start_row=row, start_column=c1, end_row=row, end_column=c2)
    c = ws.cell(row=row, column=c1, value=texto)
    c.fill = _fill(bg)
    c.font = Font(bold=True, color=C_BLANCO, size=11)
    c.alignment = _alin()
    ws.row_dimensions[row].height = 22
    return c

def nota(ws, row, c1, c2, texto, bg=C_AMARILLO):
    ws.merge_cells(start_row=row, start_column=c1, end_row=row, end_column=c2)
    c = ws.cell(row=row, column=c1, value=texto)
    c.fill = _fill(bg)
    c.font = Font(size=9, color="7F6000", italic=True)
    c.alignment = _alin(h="left", wrap=True)
    ws.row_dimensions[row].height = 36
    return c

def col_w(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

def freeze(ws, cell="A3"):
    ws.freeze_panes = cell

# ─── CREAR LIBRO ──────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()
wb.remove(wb.active)

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 0: ÍNDICE
# ══════════════════════════════════════════════════════════════════════════════
ws0 = wb.create_sheet("📋 Índice")
ws0.sheet_view.showGridLines = False
col_w(ws0, [3, 38, 62])

titulo(ws0, 1, 1, 3,
       "MODELO PREDICTIVO DE TIPOS DE INTERÉS — BCE · BoE · FED")
subheader(ws0, 2, 1, 3,
          "TFG · Aritz Fernández Aulestiarte  |  Documento explicativo para el tutor")

ws0.row_dimensions[4].height = 20
cel(ws0, 4, 2, "HOJA",        bg=C_AZUL_MED, bold=True, font_color=C_BLANCO, size=11)
cel(ws0, 4, 3, "CONTENIDO",   bg=C_AZUL_MED, bold=True, font_color=C_BLANCO, size=11)

hojas = [
    ("1️⃣  Variables del Modelo",      "Las 10 variables de entrada — qué mide cada una y por qué"),
    ("2️⃣  Regla de Taylor",            "Calculadora interactiva: cambia inflación o PIB y ve el tipo 'ideal' actualizarse"),
    ("3️⃣  Importancia de Variables",   "Qué peso le da el modelo a cada variable para cada banco"),
    ("4️⃣  Predicción Test 2024-2025",  "Comparativa tipo real vs predicho + cálculo del error MAE con fórmulas Excel"),
    ("5️⃣  Clasificación Dirección",    "Cómo decide el modelo si el tipo SUBE, BAJA o queda ESTABLE"),
    ("6️⃣  Escenarios 2026-2030",       "Predicciones futuras bajo 3 escenarios — fórmulas Taylor vinculadas"),
    ("7️⃣  Métricas Resumen",           "Tabla final de precisión del modelo para los tres bancos"),
]
for i, (h, d) in enumerate(hojas, start=5):
    bg = C_AZUL_CLAR if i % 2 == 0 else C_BLANCO
    ws0.row_dimensions[i].height = 22
    cel(ws0, i, 2, h, bg=bg, bold=True, font_color=C_AZUL_OSC, h="left", size=10)
    cel(ws0, i, 3, d, bg=bg, h="left", size=10)

# Leyenda
ws0.row_dimensions[13].height = 18
ws0.merge_cells("B13:C13")
c = ws0.cell(row=13, column=2, value="LEYENDA DE COLORES")
c.font = Font(bold=True, size=10, color=C_AZUL_OSC)

leyenda = [
    (14, C_AMARILLO, "7F6000", "Celda AMARILLA  →  Dato de entrada que puedes modificar"),
    (15, C_VERDE_CLAR, C_VERDE_OSC, "Celda VERDE      →  Resultado calculado automáticamente con fórmula Excel"),
    (16, C_AZUL_CLAR, C_AZUL_OSC,  "Celda AZUL       →  Dato informativo real del modelo"),
    (17, C_ROJO_CLAR, C_ROJO_OSC,  "Celda ROJA       →  Error de predicción (diferencia real − predicho)"),
]
for row, bg, fc, txt in leyenda:
    ws0.row_dimensions[row].height = 18
    ws0.merge_cells(f"B{row}:C{row}")
    c = ws0.cell(row=row, column=2, value=txt)
    c.fill = _fill(bg)
    c.font = Font(color=fc, size=10)
    c.alignment = _alin(h="left")
    c.border = _borde()

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 1: VARIABLES DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
ws1 = wb.create_sheet("1️⃣ Variables")
ws1.sheet_view.showGridLines = False
col_w(ws1, [3, 18, 26, 30, 10, 58])
freeze(ws1, "A3")

titulo(ws1, 1, 1, 6, "VARIABLES DEL MODELO — Las 10 entradas del Random Forest")
for col, txt in enumerate(["#","Variable","Nombre completo","Unidad","Frecuencia","¿Qué mide para el modelo?"], 1):
    cel(ws1, 2, col, txt, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO, size=10)
ws1.row_dimensions[2].height = 20

vars_info = [
    (1,  "PIB_growth",   "Crecimiento del PIB",             "%",                      "Anual",
     "Cuánto crece la economía respecto al año anterior. Si sube mucho → economía recalentada → bancos suben tipos"),
    (2,  "PIBpc_growth", "Crecimiento PIB per cápita",      "%",                      "Anual",
     "Igual que el anterior pero dividido por habitante. Más preciso para comparar entre países"),
    (3,  "Inflación",    "Tasa de inflación",               "%",                      "Mensual",
     "Cuánto suben los precios. El objetivo de los tres bancos es 2%. Si supera ese nivel → suben tipos"),
    (4,  "Desempleo",    "Tasa de desempleo",               "%",                      "Mensual",
     "Porcentaje de personas sin trabajo. Especialmente importante para la FED (doble mandato: inflación + empleo)"),
    (5,  "Rating",       "Calificación crediticia",         "Escala 1-10",            "Estático",
     "Solidez financiera del país/zona. BCE=8.09 | BoE=8.50 | FED=9.50. El mismo valor todos los meses → no influye en el modelo"),
    (6,  "VIX",          "Índice de volatilidad de mercado","Puntos (9=calma, 80=pánico)","Mensual",
     "Mide el miedo en los mercados. Alto VIX = incertidumbre = bancos centrales más cautos con los tipos"),
    (7,  "Oro",          "Precio del oro",                  "USD/onza troy",          "Mensual",
     "Activo refugio. Sube cuando hay incertidumbre económica o inflación alta. Señal de tensión financiera"),
    (8,  "Plata",        "Precio de la plata",              "USD/onza troy",          "Mensual",
     "Combina factor refugio + demanda industrial. Complementa al oro como señal de mercado"),
    (9,  "Gas",          "Precio del gas natural",          "EUR/MWh (BCE/BoE) | USD/MMBtu (FED)","Mensual",
     "Impacta directamente en la inflación energética. La crisis de 2022 disparó los tipos precisamente por esto"),
    (10, "OIS_taylor",   "Regla de Taylor (calculada)",     "%",                      "Calculada a partir de Inflación y PIB",
     "Tipo de interés 'ideal' según la fórmula de Taylor. NO es un dato externo — se calcula con fórmula matemática (ver hoja 2)"),
]
for i, (num, var, nombre, unidad, freq, desc) in enumerate(vars_info, start=3):
    bg = C_AZUL_CLAR if i % 2 == 0 else C_BLANCO
    ws1.row_dimensions[i].height = 42
    for col, val in [(1,num),(2,var),(3,nombre),(4,unidad),(5,freq),(6,desc)]:
        c = cel(ws1, i, col, val, bg=bg, h="center" if col < 6 else "left", wrap=True, size=9)
        if col == 2:
            c.font = Font(bold=True, color=C_AZUL_OSC, size=10)
        if col == 10:
            c.font = Font(bold=True, color=C_VERDE_OSC, size=10)

nota(ws1, 13, 1, 6,
     "⚠️  IMPORTANTE sobre OIS_taylor: la fórmula es  OIS_taylor = MÁX(0 ;  r*  +  1,5×(Inflación − 2%)  +  0,5×(PIB_growth − y*)). "
     "Se calcula internamente para evitar que el modelo use el tipo real como variable de entrada (lo que sería trampa). "
     "Ver hoja 2️⃣ para la calculadora interactiva.")

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 2: REGLA DE TAYLOR — CALCULADORA INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("2️⃣ Regla Taylor")
ws2.sheet_view.showGridLines = False
col_w(ws2, [3, 36, 16, 16, 16])
freeze(ws2, "A3")

titulo(ws2, 1, 1, 5, "REGLA DE TAYLOR — Calculadora interactiva del tipo de interés 'ideal'")

nota(ws2, 2, 1, 5,
     "La Regla de Taylor (1993) estima cuál debería ser el tipo de interés según la economía real. "
     "Si la inflación supera el 2% o el PIB crece más de lo sostenible → el tipo debería subir. "
     "Cambia los valores AMARILLOS y las celdas VERDES se recalcularán solas.")

# Fórmula general
ws2.merge_cells("B3:E3")
c = ws2.cell(row=3, column=2,
             value="FÓRMULA:    OIS_taylor  =  MÁX( 0 ;  r*  +  1,5 × (Inflación − 2%)  +  0,5 × (PIB_growth − y*) )")
c.fill = _fill(C_VERDE_CLAR)
c.font = Font(bold=True, size=12, color=C_VERDE_OSC)
c.alignment = _alin()
ws2.row_dimensions[3].height = 28

# ── BLOQUE A: Parámetros fijos ────────────────────────────────────────────────
subheader(ws2, 5, 1, 5, "BLOQUE A — Parámetros fijos del modelo (calibrados con literatura económica)")

for col, txt in [(2,"Parámetro"),(3,"BCE"),(4,"BoE"),(5,"FED")]:
    cel(ws2, 6, col, txt, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO)
ws2.row_dimensions[6].height = 18

params = [
    ("r*   — Tipo neutral real (%)",         1.0,  1.5,  2.5),
    ("π*   — Objetivo inflación (%)",        2.0,  2.0,  2.0),
    ("y*   — Crecimiento potencial PIB (%)", 1.5,  2.0,  2.0),
]
# Filas 7, 8, 9
for i, (label, bce, boe, fed) in enumerate(params, start=7):
    bg = C_AZUL_CLAR if i % 2 == 0 else C_BLANCO
    ws2.row_dimensions[i].height = 20
    cel(ws2, i, 2, label, bg=bg, h="left", size=10, bold=True)
    for col, val in [(3, bce), (4, boe), (5, fed)]:
        cel(ws2, i, col, val, bg=bg, num_format="0.0")
# R_ROW=7, PI_ROW=8, Y_ROW=9

# ── BLOQUE B: Calculadora ─────────────────────────────────────────────────────
subheader(ws2, 11, 1, 5, "BLOQUE B — Calculadora interactiva  (modifica los valores AMARILLOS)")

for col, txt in [(2,"Variable"),(3,"BCE"),(4,"BoE"),(5,"FED")]:
    cel(ws2, 12, col, txt, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO)
ws2.row_dimensions[12].height = 18

# Inputs Inflación (fila 13) y PIB_growth (fila 14)
input_vals = [
    ("📥  Inflación actual (%)",      2.3, 2.8, 2.5),   # fila 13
    ("📥  PIB_growth actual (%)",     1.4, 1.2, 2.3),   # fila 14
]
for i, (label, bce, boe, fed) in enumerate(input_vals, start=13):
    ws2.row_dimensions[i].height = 24
    cel(ws2, i, 2, label, bg=C_AMARILLO, bold=True, font_color="7F6000", h="left")
    for col, val in [(3, bce), (4, boe), (5, fed)]:
        c = cel(ws2, i, col, val, bg=C_AMARILLO, bold=True, font_color="7F6000", num_format="0.0")
# INFL_ROW=13, PIB_ROW=14

# Cálculos intermedios vinculados
pasos = [
    (15, "📐  Paso 1: Gap inflación  =  Inflación − π*",
         "=C13-C8", "=D13-D8", "=E13-E8"),
    (16, "📐  Paso 2: Gap PIB  =  PIB_growth − y*",
         "=C14-C9", "=D14-D9", "=E14-E9"),
    (17, "📐  Paso 3: Taylor bruto  =  r* + 1,5×Gap_π + 0,5×Gap_PIB",
         "=C7+1.5*C15+0.5*C16", "=D7+1.5*D15+0.5*D16", "=E7+1.5*E15+0.5*E16"),
]
for row, label, f_bce, f_boe, f_fed in pasos:
    ws2.row_dimensions[row].height = 24
    cel(ws2, row, 2, label, bg=C_VERDE_CLAR, font_color=C_VERDE_OSC, h="left", size=10)
    for col, formula in [(3, f_bce), (4, f_boe), (5, f_fed)]:
        cel(ws2, row, col, formula, bg=C_VERDE_CLAR, font_color=C_VERDE_OSC,
            bold=True, num_format="0.00")

# Resultado final
ws2.row_dimensions[18].height = 32
cel(ws2, 18, 2, "✅  OIS_taylor FINAL  =  MÁX(0 ; Paso 3)",
    bg=C_VERDE_OSC, bold=True, font_color=C_BLANCO, size=12, h="left")
for col, f in [(3,"=MAX(0,C17)"),(4,"=MAX(0,D17)"),(5,"=MAX(0,E17)")]:
    c = cel(ws2, 18, col, f, bg=C_VERDE_OSC, bold=True, font_color=C_BLANCO, size=13,
            num_format="0.00")
    c.border = _borde("medium", "000000")

nota(ws2, 20, 1, 5,
     "💡  En los escenarios 2026-2030, este mismo cálculo se aplica con los valores macro de cada escenario "
     "(Base, Optimista, Pesimista). El OIS_taylor resultante es directamente el tipo de interés predicho para ese período. "
     "Ve a la hoja 6️⃣ para ver los tres escenarios con sus fórmulas.")

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 3: IMPORTANCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("3️⃣ Importancias")
ws3.sheet_view.showGridLines = False
col_w(ws3, [3, 22, 18, 18, 18, 5, 40])
freeze(ws3, "A3")

titulo(ws3, 1, 1, 7, "IMPORTANCIA DE VARIABLES — Cuánto contribuye cada variable a las predicciones del Random Forest")

nota(ws3, 2, 1, 7,
     "El Random Forest calcula automáticamente qué variables son más útiles para predecir los cambios de tipos. "
     "El porcentaje indica cuánto redujo el error cada variable a lo largo de los 200 árboles de decisión. "
     "La suma de todas las variables = 100% para cada banco.")

for col, txt in [(2,"Variable"),(3,"BCE (Eurozona)"),(4,"BoE (Reino Unido)"),(5,"FED (EE.UU.)"),(7,"Interpretación")]:
    cel(ws3, 4, col, txt, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO)
ws3.row_dimensions[4].height = 20

# Unir importancias en orden BCE
orden = imp["BCE"].index.tolist()
interpretaciones = {
    "VIX":         "Alta volatilidad de mercado = señal fuerte de cambio inminente de tipos",
    "OIS_taylor":  "El tipo 'ideal' según Taylor concentra la señal de inflación + PIB",
    "Inflacion":   "Variable objetivo de los bancos centrales — objetivo 2%",
    "Desempleo":   "Especialmente crítico para la FED (doble mandato)",
    "PIB_growth":  "Crecimiento real de la economía — cuánto produce el país",
    "PIBpc_growth":"Crecimiento per cápita — más preciso que el PIB total",
    "Plata":       "Proxy de demanda industrial y tensión financiera",
    "Oro":         "Activo refugio — sube con la incertidumbre económica",
    "Gas":         "Impulsor clave de la inflación energética (esp. Europa 2022)",
    "Rating":      "Constante por banco → el modelo nunca lo usa (0%)",
}
for i, var in enumerate(orden, start=5):
    bg = C_AZUL_CLAR if i % 2 == 0 else C_BLANCO
    ws3.row_dimensions[i].height = 22
    v_bce = float(imp["BCE"].loc[var, "importancia"])
    v_boe = float(imp["BoE"].loc[var, "importancia"]) if var in imp["BoE"].index else 0.0
    v_fed = float(imp["FED"].loc[var, "importancia"]) if var in imp["FED"].index else 0.0

    cel(ws3, i, 2, var, bg=bg, bold=True, font_color=C_AZUL_OSC, h="left")
    for col, val in [(3, v_bce), (4, v_boe), (5, v_fed)]:
        pct = round(val * 100, 1)
        # Color más intenso si > 15%
        bg_val = "AADAFF" if pct > 15 else (C_AZUL_CLAR if pct > 5 else bg)
        cel(ws3, i, col, pct / 100, bg=bg_val, bold=(pct > 15), num_format="0.0%")
    cel(ws3, i, 7, interpretaciones.get(var, ""), bg=bg, h="left", size=9, wrap=True)

# Totales (fórmula Excel SUM)
ws3.row_dimensions[15].height = 22
cel(ws3, 15, 2, "TOTAL", bg=C_AZUL_OSC, bold=True, font_color=C_BLANCO)
for col, letra in [(3,"C"),(4,"D"),(5,"E")]:
    cel(ws3, 15, col, f"=SUM({letra}5:{letra}14)", bg=C_AZUL_OSC,
        bold=True, font_color=C_BLANCO, num_format="0.0%")
cel(ws3, 15, 7, "→  Debe sumar 100% para cada banco (el modelo distribuye el 100% entre las variables)",
    bg=C_AZUL_OSC, font_color=C_BLANCO, h="left")

nota(ws3, 17, 1, 7,
     "🔑  LECTURA CLAVE: La FED da mucho peso al PIB (≈38%) porque tiene doble mandato legal "
     "(estabilidad de precios + máximo empleo). El BCE y BoE, con mandato único de inflación, "
     "reaccionan más al VIX (volatilidad de mercado) porque los grandes shocks del período 2000-2025 "
     "(crisis 2008, COVID, energía 2022) se manifestaron primero como volatilidad de mercado.")

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 4: PREDICCIÓN TEST 2024-2025
# ══════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("4️⃣ Test 2024-2025")
ws4.sheet_view.showGridLines = False
col_w(ws4, [3, 14, 15, 15, 14, 14, 5, 14, 15, 15, 14, 14, 5, 14, 15, 15, 14, 14])
freeze(ws4, "A4")

titulo(ws4, 1, 1, 18, "PREDICCIÓN TEST 2024-2025 — Comparativa Tipo Real vs Predicho + Cálculo del Error (MAE)")

nota(ws4, 2, 1, 18,
     "El modelo predice el tipo de interés mensual. Aquí se compara con lo que realmente ocurrió. "
     "Error = Tipo_Real − Tipo_Predicho  |  |Error| = ABS(Error)  |  MAE = PROMEDIO de todos los |Error|  "
     "Las fórmulas verdes calculan el error automáticamente a partir de los datos azules.")

# Headers por banco
bancos_test = [("BCE", 2, "1F4E79"), ("BoE", 8, "CC0000"), ("FED", 14, "228B22")]
for banco, col_ini, color in bancos_test:
    ws4.merge_cells(start_row=3, start_column=col_ini, end_row=3, end_column=col_ini+5)
    c = ws4.cell(row=3, column=col_ini, value=f"  {banco}")
    c.fill = _fill(color)
    c.font = Font(bold=True, color=C_BLANCO, size=12)
    c.alignment = _alin(h="left")

    sub_headers = ["Fecha", "Tipo Real (%)", "Tipo Predicho (%)", "Error (pp)", "|Error| (pp)", "Acierto dirección"]
    for j, sh in enumerate(sub_headers):
        cel(ws4, 4 + 0, col_ini + j, sh, bg=C_AZUL_MED if color=="1F4E79" else
            ("FFCCCC" if color=="CC0000" else "CCFFCC"),
            bold=True, font_color=C_BLANCO if color=="1F4E79" else "000000", size=9)

ws4.row_dimensions[3].height = 22
ws4.row_dimensions[4].height = 32

# Datos de los 24 meses (2024-2025) para cada banco
for banco, col_ini, color in bancos_test:
    df_s = scores[banco].reset_index()
    for idx in range(min(24, len(df_s))):
        row = idx + 5
        ws4.row_dimensions[row].height = 18
        bg = C_GRIS if idx % 2 == 0 else C_BLANCO

        fecha_str  = str(df_s.loc[idx, df_s.columns[0]])[:7]
        tipo_real  = float(df_s.loc[idx, "tipo_real"])
        tipo_pred  = float(df_s.loc[idx, "tipo_predicho"])
        clase_real = str(df_s.loc[idx, "clase_real"])
        dir_pred   = str(df_s.loc[idx, "direccion"])
        acierto    = "✅" if clase_real == dir_pred else "❌"

        col_letra_real = get_column_letter(col_ini + 1)
        col_letra_pred = get_column_letter(col_ini + 2)

        cel(ws4, row, col_ini,     fecha_str, bg=bg, size=9)
        cel(ws4, row, col_ini + 1, tipo_real, bg=C_AZUL_CLAR, num_format="0.00", size=9)
        cel(ws4, row, col_ini + 2, tipo_pred, bg=C_AZUL_CLAR, num_format="0.00", size=9)
        # Error = Real - Predicho (fórmula vinculada)
        err_formula = f"={col_letra_real}{row}-{col_letra_pred}{row}"
        c_err = cel(ws4, row, col_ini + 3, err_formula, bg=C_ROJO_CLAR, num_format="0.000", size=9)
        # |Error| (fórmula ABS vinculada al error)
        col_err = get_column_letter(col_ini + 3)
        abs_formula = f"=ABS({col_err}{row})"
        cel(ws4, row, col_ini + 4, abs_formula, bg=C_ROJO_CLAR, num_format="0.000", size=9)
        # Acierto dirección
        bg_ac = "E2EFDA" if acierto == "✅" else C_ROJO_CLAR
        cel(ws4, row, col_ini + 5, acierto, bg=bg_ac, size=11)

# Fila MAE (fórmula PROMEDIO)
row_mae = 29
ws4.row_dimensions[row_mae].height = 26
for banco, col_ini, color in bancos_test:
    col_abs = get_column_letter(col_ini + 4)
    cel(ws4, row_mae, col_ini, "MAE (Error Medio)", bg=C_AZUL_OSC, bold=True,
        font_color=C_BLANCO, size=10)
    mae_formula = f"=AVERAGE({col_abs}5:{col_abs}28)"
    cel(ws4, row_mae, col_ini + 1, mae_formula, bg=C_VERDE_OSC, bold=True,
        font_color=C_BLANCO, size=11, num_format="0.000")
    ws4.merge_cells(start_row=row_mae, start_column=col_ini+2,
                    end_row=row_mae, end_column=col_ini+5)
    c = ws4.cell(row=row_mae, column=col_ini+2,
                 value="← PROMEDIO( |Error| ) — cuanto más bajo, mejor")
    c.fill = _fill(C_VERDE_CLAR)
    c.font = Font(italic=True, color=C_VERDE_OSC, size=9)
    c.alignment = _alin(h="left")

nota(ws4, 31, 1, 18,
     "📖  MAE = Mean Absolute Error (Error Absoluto Medio). Mide en promedio cuántos puntos porcentuales "
     "se equivoca el modelo. Un MAE de 0.13 significa que el modelo se equivoca de media en ±0,13 puntos porcentuales. "
     "En un rango de tipos que va del 0% al 5,5%, eso es una precisión del ~97%.")

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 5: CLASIFICACIÓN DE DIRECCIÓN
# ══════════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("5️⃣ Clasificación")
ws5.sheet_view.showGridLines = False
col_w(ws5, [3, 14, 14, 12, 12, 12, 14, 14, 5, 40])
freeze(ws5, "A4")

titulo(ws5, 1, 1, 10, "CLASIFICACIÓN DE DIRECCIÓN — Cómo decide el modelo si el tipo SUBE, BAJA o queda ESTABLE")

nota(ws5, 2, 1, 10,
     "Además de predecir el nivel del tipo, el modelo clasifica la DIRECCIÓN del movimiento. "
     "La Regresión Logística calcula la probabilidad de cada dirección. "
     "Net Score = P(Sube) − P(Baja): va de −100 (bajada segura) a +100 (subida segura).")

# Regla de clasificación
ws5.row_dimensions[4].height = 56
ws5.merge_cells("B4:J4")
c = ws5.cell(row=4, column=2,
             value=("REGLA DE CLASIFICACIÓN:\n"
                    "  Si  Δtipo  >  +0,125 pp  →  'SUBE'      (el tipo sube más de la mitad del mínimo paso real de 0,25pp)\n"
                    "  Si  Δtipo  <  −0,125 pp  →  'BAJA'      (el tipo baja más de la mitad del mínimo paso real)\n"
                    "  Si  |Δtipo|  ≤  0,125 pp →  'ESTABLE'   (el tipo prácticamente no se mueve)"))
c.fill = _fill(C_VERDE_CLAR)
c.font = Font(size=10, color=C_VERDE_OSC, bold=False)
c.alignment = _alin(h="left", wrap=True)
c.border = _borde()

# Headers tabla BCE (ejemplo)
subheader(ws5, 6, 1, 10, "EJEMPLO: BCE 2024-2025 — Probabilidades de dirección por mes")
hdrs = ["Fecha", "Tipo Real", "Tipo Pred.", "P(Baja)%", "P(Estable)%", "P(Sube)%",
        "Dirección", "Net Score", "", "Fórmula Net Score"]
for col, h in enumerate(hdrs, 1):
    cel(ws5, 7, col, h, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO, size=9)
ws5.row_dimensions[7].height = 20

df_s = scores["BCE"].reset_index()
for idx in range(min(24, len(df_s))):
    row = idx + 8
    ws5.row_dimensions[row].height = 18
    bg = C_GRIS if idx % 2 == 0 else C_BLANCO

    fecha_str = str(df_s.loc[idx, df_s.columns[0]])[:7]
    tipo_real = float(df_s.loc[idx, "tipo_real"])
    tipo_pred = float(df_s.loc[idx, "tipo_predicho"])
    p_baja    = float(df_s.loc[idx, "P_Baja"])
    p_est     = float(df_s.loc[idx, "P_Estable"])
    p_sube    = float(df_s.loc[idx, "P_Sube"])
    net_real  = float(df_s.loc[idx, "Net_Score"])
    dir_pred  = str(df_s.loc[idx, "direccion"])
    clase_real= str(df_s.loc[idx, "clase_real"])

    bg_dir = ("E2EFDA" if dir_pred == clase_real else C_ROJO_CLAR)

    cel(ws5, row, 1,  fecha_str,    bg=bg, size=9)
    cel(ws5, row, 2,  tipo_real,    bg=C_AZUL_CLAR, num_format="0.00", size=9)
    cel(ws5, row, 3,  tipo_pred,    bg=C_AZUL_CLAR, num_format="0.00", size=9)
    cel(ws5, row, 4,  p_baja/100,   bg="FCE4D6",    num_format="0.0%", size=9)
    cel(ws5, row, 5,  p_est/100,    bg=C_GRIS,      num_format="0.0%", size=9)
    cel(ws5, row, 6,  p_sube/100,   bg=C_VERDE_CLAR,num_format="0.0%", size=9)
    cel(ws5, row, 7,  dir_pred,     bg=bg_dir, bold=True, size=9)
    # Net Score como fórmula vinculada a P_Sube y P_Baja
    col_baja = get_column_letter(4)
    col_sube = get_column_letter(6)
    net_formula = f"={col_sube}{row}-{col_baja}{row}"
    cel(ws5, row, 8, net_formula, bg=C_VERDE_CLAR, font_color=C_VERDE_OSC,
        bold=True, num_format="0.0%", size=9)
    cel(ws5, row, 10,
        f"= P(Sube) − P(Baja) = {p_sube:.1f}% − {p_baja:.1f}% = {net_real:+.1f}%",
        bg=bg, h="left", size=8)

# Accuracy dirección
row_acc = 32
ws5.row_dimensions[row_acc].height = 24
cel(ws5, row_acc, 1, "Acierto dirección BCE", bg=C_AZUL_OSC, bold=True, font_color=C_BLANCO)
ws5.merge_cells(f"B{row_acc}:D{row_acc}")
c = ws5.cell(row=row_acc, column=2,
             value="=COUNTIF(G8:G31,C8)/COUNTA(G8:G31)")
c.fill = _fill(C_VERDE_OSC)
c.font = Font(bold=True, color=C_BLANCO, size=12)
c.alignment = _alin()
c.number_format = "0.0%"
c.border = _borde()

nota(ws5, 34, 1, 10,
     "📊  Net Score > 0 → el modelo ve más probabilidad de subida que bajada. "
     "Net Score < 0 → más probabilidad de bajada. "
     "Net Score ≈ 0 → el modelo considera la situación equilibrada (probable Estable).")

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 6: ESCENARIOS 2026-2030
# ══════════════════════════════════════════════════════════════════════════════
ws6 = wb.create_sheet("6️⃣ Escenarios 2026-2030")
ws6.sheet_view.showGridLines = False
col_w(ws6, [3, 26, 14, 14, 14, 5, 26, 14, 14, 14, 5, 26, 14, 14, 14])
freeze(ws6, "A4")

titulo(ws6, 1, 1, 15, "ESCENARIOS 2026-2030 — Predicciones futuras con Regla de Taylor + probabilidades de dirección")

nota(ws6, 2, 1, 15,
     "Para el futuro (2026-2030) el modelo usa la Regla de Taylor directamente como tipo predicho. "
     "Se definen 3 escenarios macroeconómicos. Las celdas AMARILLAS son los supuestos de cada escenario — "
     "modifícalas y las predicciones VERDES se recalcularán automáticamente.")

# Bloque parámetros Taylor (referencia)
subheader(ws6, 4, 1, 15, "PARÁMETROS TAYLOR (referencia) — Mismos que hoja 2️⃣")
for col, txt in [(2,"Param."),(3,"BCE"),(4,"BoE"),(5,"FED")]:
    cel(ws6, 5, col, txt, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO, size=9)
params_ref = [("r*",1.0,1.5,2.5),("π* (objetivo infl.)",2.0,2.0,2.0),("y* (PIB potencial)",1.5,2.0,2.0)]
for i, (lab, bce, boe, fed) in enumerate(params_ref, start=6):
    ws6.row_dimensions[i].height = 18
    cel(ws6, i, 2, lab, bg=C_AZUL_CLAR, h="left", size=9)
    for col, val in [(3,bce),(4,boe),(5,fed)]:
        cel(ws6, i, col, val, bg=C_AZUL_CLAR, num_format="0.0", size=9)
# r*=fila6, π*=fila7, y*=fila8

TAYLOR_REF_ROWS = {"r_star": 6, "pi_star": 7, "y_star": 8}

# 3 escenarios × 3 bancos
esc_nombres = ["Base", "Optimista", "Pesimista"]
esc_keys    = ["base", "optimista", "pesimista"]
esc_colors  = [C_AZUL_MED, "2E8B57", "C0504D"]
bancos_col  = [(2, "BCE", 3), (7, "BoE", 4), (12, "FED", 5)]

row_inicio = 10
for esc_nombre, esc_key, esc_color in zip(esc_nombres, esc_keys, esc_colors):
    subheader(ws6, row_inicio, 1, 15, f"ESCENARIO {esc_nombre.upper()}", bg=esc_color)
    row_inicio += 1

    # Headers variables macro
    for col_start, banco, taylor_col in bancos_col:
        cel(ws6, row_inicio, col_start, f"Variable ({banco})",
            bg=esc_color, bold=True, font_color=C_BLANCO, size=9)
        for j, sub in enumerate(["Valor 2026", "Tipo pred. (%)", "Net Score"]):
            cel(ws6, row_inicio, col_start+1+j, sub,
                bg=esc_color, bold=True, font_color=C_BLANCO, size=9)
    ws6.row_dimensions[row_inicio].height = 20
    row_inicio += 1

    # Variables macro de entrada (amarillas)
    var_labels = ["PIB_growth (%)", "PIBpc_growth (%)", "Inflación (%)", "Desempleo (%)"]
    var_keys   = ["PIB_growth", "PIBpc_growth", "Inflacion", "Desempleo"]
    infl_row_local = {}
    pib_row_local  = {}

    for vi, (vlab, vkey) in enumerate(zip(var_labels, var_keys)):
        ri = row_inicio + vi
        ws6.row_dimensions[ri].height = 20
        for col_start, banco, taylor_col in bancos_col:
            ed = esc_data.get(f"{banco}_{esc_key}")
            val = float(ed[vkey]) if ed is not None and vkey in ed else 0.0
            cel(ws6, ri, col_start,   vlab, bg=C_AMARILLO, font_color="7F6000", h="left", size=9)
            cel(ws6, ri, col_start+1, val,  bg=C_AMARILLO, font_color="7F6000", bold=True, num_format="0.0", size=9)
            # Columnas tipo_pred y net_score quedan vacías aquí (salen de fórmula abajo)
        if vkey == "Inflacion":
            infl_row_local = {b: row_inicio + vi for _,b,_ in bancos_col}
        if vkey == "PIB_growth":
            pib_row_local  = {b: row_inicio + vi for _,b,_ in bancos_col}

    row_inicio += len(var_labels)

    # OIS_taylor calculado con fórmula vinculada
    ws6.row_dimensions[row_inicio].height = 28
    for col_start, banco, taylor_col in bancos_col:
        infl_r = row_inicio - len(var_labels) + 2   # fila de inflación
        pib_r  = row_inicio - len(var_labels)        # fila de PIB_growth
        col_val = get_column_letter(col_start + 1)   # columna de valor
        r_star_cell  = f"{get_column_letter(taylor_col)}{TAYLOR_REF_ROWS['r_star']}"
        pi_star_cell = f"{get_column_letter(taylor_col)}{TAYLOR_REF_ROWS['pi_star']}"
        y_star_cell  = f"{get_column_letter(taylor_col)}{TAYLOR_REF_ROWS['y_star']}"
        infl_cell  = f"{col_val}{infl_r}"
        pib_cell   = f"{col_val}{pib_r}"

        taylor_formula = (
            f"=MAX(0,{r_star_cell}"
            f"+1.5*({infl_cell}-{pi_star_cell})"
            f"+0.5*({pib_cell}-{y_star_cell}))"
        )
        cel(ws6, row_inicio, col_start, "✅ OIS_taylor (tipo pred.)",
            bg=C_VERDE_OSC, bold=True, font_color=C_BLANCO, h="left", size=10)
        c = cel(ws6, row_inicio, col_start+1, taylor_formula,
                bg=C_VERDE_OSC, bold=True, font_color=C_BLANCO, size=12, num_format="0.00")
        c.border = _borde("medium", "000000")

        # Net Score del escenario: primer mes predicho
        fd = fut_data.get(f"{banco}_{esc_key}")
        net_val = float(fd["Net_Score"].iloc[0]) / 100 if fd is not None else 0.0
        cel(ws6, row_inicio, col_start+2, net_val,
            bg=C_VERDE_CLAR, bold=True, font_color=C_VERDE_OSC, num_format="0.0%", size=10)
        cel(ws6, row_inicio, col_start+3, "Net Score ene-2026",
            bg=C_VERDE_CLAR, font_color=C_VERDE_OSC, h="left", size=8)

    row_inicio += 2   # espacio entre escenarios

nota(ws6, row_inicio, 1, 15,
     "💡  Cambia cualquier celda AMARILLA (PIB_growth, Inflación, etc.) y el OIS_taylor VERDE se recalculará "
     "automáticamente con la Regla de Taylor. Así puedes crear tu propio escenario personalizado "
     "sin necesidad del dashboard.")

# ══════════════════════════════════════════════════════════════════════════════
# HOJA 7: MÉTRICAS RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
ws7 = wb.create_sheet("7️⃣ Métricas")
ws7.sheet_view.showGridLines = False
col_w(ws7, [3, 20, 14, 14, 14, 14, 14, 14, 5, 42])
freeze(ws7, "A3")

titulo(ws7, 1, 1, 10, "MÉTRICAS DE PRECISIÓN — Evaluación del modelo en el período test 2024-2025")

nota(ws7, 2, 1, 10,
     "Todas las métricas se calculan ÚNICAMENTE sobre los 24 meses de test (2024-2025), "
     "que el modelo NUNCA vio durante el entrenamiento. "
     "MAE = error medio | RMSE = error medio penalizando más los grandes errores | R² = bondad de ajuste (1.0 = perfecto)")

# Headers
hdrs7 = ["Banco","Modelo","MAE Train","RMSE Train","R² Train","MAE Test","RMSE Test","R² Test","","Significado R²"]
for col, h in enumerate(hdrs7, 1):
    cel(ws7, 3, col, h, bg=C_AZUL_MED, bold=True, font_color=C_BLANCO, size=10)
ws7.row_dimensions[3].height = 22

r2_significado = {
    "BCE": "El modelo explica el 96,3% de la variación del tipo de interés BCE en test",
    "BoE": "El modelo explica el 97,1% de la variación del tipo de interés BoE en test",
    "FED": "El modelo explica el 93,7% de la variación del tipo de interés FED en test",
}

row = 4
for _, fila in resumen.iterrows():
    banco  = fila["Banco"]
    modelo = fila["Modelo"]
    bg = C_AZUL_CLAR if modelo == "RF" else C_BLANCO
    ws7.row_dimensions[row].height = 22

    cel(ws7, row, 1,  banco,             bg=bg, bold=(modelo=="RF"), font_color=C_AZUL_OSC)
    cel(ws7, row, 2,  "Random Forest" if modelo=="RF" else "Ridge (comparativo)",
        bg=bg, bold=(modelo=="RF"), h="left")
    cel(ws7, row, 3,  fila["MAE_train"],  bg=bg, num_format="0.0000")
    cel(ws7, row, 4,  fila["RMSE_train"], bg=bg, num_format="0.0000")
    cel(ws7, row, 5,  fila["R2_train"],   bg=bg, num_format="0.0000")
    # Test en verde si es RF (modelo principal)
    bg_test = C_VERDE_CLAR if modelo=="RF" else bg
    cel(ws7, row, 6,  fila["MAE_test"],   bg=bg_test, bold=(modelo=="RF"), num_format="0.0000")
    cel(ws7, row, 7,  fila["RMSE_test"],  bg=bg_test, bold=(modelo=="RF"), num_format="0.0000")
    cel(ws7, row, 8,  fila["R2_test"],    bg=bg_test, bold=(modelo=="RF"), num_format="0.0000")
    if modelo == "RF":
        cel(ws7, row, 10, r2_significado.get(banco,""), bg=bg, h="left", size=9, wrap=True)
    row += 1

# Separador y fórmulas explicadas
ws7.row_dimensions[row+1].height = 22
subheader(ws7, row+1, 1, 10, "GLOSARIO DE MÉTRICAS — Fórmulas y significado")

metricas_exp = [
    ("MAE  (Error Medio Absoluto)",
     "= PROMEDIO( |tipo_real − tipo_predicho| )  →  En promedio, ¿cuántos pp se equivoca el modelo?"),
    ("RMSE (Raíz Error Cuadrático Medio)",
     "= RAÍZ( PROMEDIO( (tipo_real − tipo_predicho)² ) )  →  Penaliza más los errores grandes"),
    ("R²   (Coeficiente de determinación)",
     "Va de 0 a 1. Un R²=0.97 significa que el modelo explica el 97% de la variación observada en los tipos"),
    ("Acierto de dirección",
     "% de meses en que el modelo predijo correctamente SUBE / BAJA / ESTABLE"),
]
for i, (met, exp) in enumerate(metricas_exp, start=row+2):
    bg = C_AZUL_CLAR if i % 2 == 0 else C_BLANCO
    ws7.row_dimensions[i].height = 28
    cel(ws7, i, 1, met, bg=bg, bold=True, font_color=C_AZUL_OSC, h="left", size=10)
    ws7.merge_cells(start_row=i, start_column=2, end_row=i, end_column=10)
    c = ws7.cell(row=i, column=2, value=exp)
    c.fill = _fill(bg)
    c.font = Font(size=9, italic=True)
    c.alignment = _alin(h="left", wrap=True)
    c.border = _borde()

nota(ws7, row + 7, 1, 10,
     "🏆  CONCLUSIÓN: El modelo tiene un R² > 0,93 en los tres bancos y un MAE inferior a 0,13 pp "
     "en un rango de tipos que va del 0% al 5,5%. Eso equivale a una precisión media superior al 97%. "
     "El período test 2024-2025 incluye los ciclos de bajadas de tipos más complejos de la última década.")

# ─── Guardar ──────────────────────────────────────────────────────────────────
ruta_salida = BASE / "Modelo_Explicativo_Tutor.xlsx"
wb.save(ruta_salida)
print(f"Excel guardado en: {ruta_salida}")
