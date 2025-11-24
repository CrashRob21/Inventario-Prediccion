# app.py (versión robusta: detecta logos aunque tengan doble extensión o distinto formato)
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# ---------------------------
# Configuración inicial
# ---------------------------
st.set_page_config(page_title="Forecast Inventarios - Daimay", layout="wide")

# Ruta base (carpeta del script)
BASE_DIR = os.path.dirname(__file__) or os.getcwd()
LOGOS_DIR = os.path.join(BASE_DIR, "logos")

# ---------------------------
# Funciones utilitarias para buscar logos
# ---------------------------
def find_logo_file(keywords, folder=LOGOS_DIR):
    """
    Busca en folder un archivo cuyo nombre contenga cualquiera de las palabras en keywords.
    Devuelve la ruta completa del primer archivo válido o None.
    """
    if not os.path.isdir(folder):
        return None

    files = os.listdir(folder)
    lower_files = {f.lower(): f for f in files}
    for fname_lower, fname_real in lower_files.items():
        for kw in keywords:
            if kw.lower() in fname_lower:
                candidate_path = os.path.join(folder, fname_real)
                try:
                    with Image.open(candidate_path) as im:
                        im.verify()
                    return candidate_path
                except Exception:
                    continue

    for fname_real in files:
        p = os.path.join(folder, fname_real)
        try:
            with Image.open(p) as im:
                im.verify()
            return p
        except Exception:
            continue

    return None

# ---------------------------
# Buscar logos
# ---------------------------
logo_daimay_path = find_logo_file(["daimay", "daymai", "daimai", "daimay", "daimai"])
logo_insunte_path = find_logo_file(["insunte", "insun", "insut", "insute", "insute"], LOGOS_DIR)

# Mostrar logos en dos columnas
col_left, col_right = st.columns([1,1])

with col_left:
    if logo_daimay_path:
        try:
            st.image(Image.open(logo_daimay_path), width=180)
        except Exception as e:
            st.warning(f"No se pudo mostrar logo Daimay: {e}")
    else:
        st.info("Logo Daimay no encontrado automáticamente (no es obligatorio).")

with col_right:
    if logo_insunte_path:
        try:
            st.image(Image.open(logo_insunte_path), width=180)
        except Exception as e:
            st.warning(f"No se pudo mostrar logo Insunte: {e}")
    else:
        st.info("Logo Insunte no encontrado automáticamente (no es obligatorio).")

st.title("📦 Modelo Predictivo de Demanda - Área de Almacén")
st.markdown("Sube un Excel con columnas `Mes` y `Ventas` o usa el ejemplo. La app permite descargar reporte en Excel y PDF.")

# ---------------------------
# Carga de datos
# ---------------------------
archivo = st.file_uploader("📤 Sube tu archivo Excel (.xlsx) o CSV con columnas: Mes, Ventas", type=["xlsx", "csv"])

if archivo is not None:
    try:
        if str(archivo).lower().endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
        st.success("✅ Archivo cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()
else:
    st.info("Usando datos de ejemplo. Puedes subir tu archivo para ver tus datos reales.")
    df = pd.DataFrame({
        'Mes': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio'],
        'Ventas': [520, 480, 550, 600, 630, 610]
    })

# ---------------------------
# Mostrar datos y preparar
# ---------------------------
st.subheader("📊 Histórico de ventas")
st.dataframe(df)

df = df.reset_index(drop=True)
df['Mes_Num'] = np.arange(1, len(df) + 1)

X = df[['Mes_Num']]
y = df['Ventas']

# ---------------------------
# Entrenar modelo simple
# ---------------------------
modelo = LinearRegression()
modelo.fit(X, y)

horizon = st.sidebar.number_input("Meses a predecir", min_value=1, max_value=12, value=3)
futuros = pd.DataFrame({'Mes_Num': [len(df) + i for i in range(1, horizon+1)]})
prediccion = modelo.predict(futuros)
futuros['Ventas_Predichas'] = np.round(prediccion, 2)
futuros['Mes'] = [f'Mes {i}' for i in futuros['Mes_Num']]

# ---------------------------
# Gráficos
# ---------------------------
st.subheader("📈 Proyección de demanda")
fig, ax = plt.subplots()
ax.plot(df['Mes_Num'], df['Ventas'], 'bo-', label='Histórico')
ax.plot(futuros['Mes_Num'], futuros['Ventas_Predichas'], 'r--o', label='Pronóstico')
ax.set_xlabel('Mes (num)')
ax.set_ylabel('Ventas / Unidades')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("🔮 Pronóstico")
st.dataframe(futuros[['Mes','Ventas_Predichas']])

# ---------------------------
# Validación
# ---------------------------
pred_hist = modelo.predict(X)
mae = mean_absolute_error(y, pred_hist)
rmse = np.sqrt(mean_squared_error(y, pred_hist))

st.subheader("📉 Comparativa Real vs Predicho (histórico)")
comparacion = pd.DataFrame({
    'Mes': df['Mes'],
    'Ventas_Real': df['Ventas'],
    'Ventas_Predicha': np.round(pred_hist,2),
    'Error': np.round(df['Ventas'] - pred_hist,2)
})
st.dataframe(comparacion)

fig2, ax2 = plt.subplots()
width = 0.35
x = np.arange(len(df))
ax2.bar(x - width/2, df['Ventas'], width, label='Real')
ax2.bar(x + width/2, pred_hist, width, label='Predicho')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Mes'], rotation=45)
ax2.set_ylabel('Ventas')
ax2.set_title('Real vs Predicho')
ax2.legend()
st.pyplot(fig2)

# ---------------------------
# Exportar excel y pdf
# ---------------------------
st.subheader("💾 Exportar reporte")

def generar_excel(df_hist, df_forecast, comparacion):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_hist.to_excel(writer, sheet_name='Histórico', index=False)
        df_forecast.to_excel(writer, sheet_name='Pronóstico', index=False)
        comparacion.to_excel(writer, sheet_name='Comparación', index=False)
    output.seek(0)
    return output

excel_bytes = generar_excel(df[['Mes','Ventas']], futuros[['Mes','Ventas_Predichas']], comparacion)

st.download_button(
    label="📥 Descargar reporte (Excel)",
    data=excel_bytes,
    file_name=f"reporte_inventario_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

def generar_pdf_simple(fig, tabla_df, titulo="Reporte de Predicción"):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    return buf

pdf_buf = generar_pdf_simple(fig, comparacion, titulo="Reporte de Predicción - Daimay")

st.download_button(
    label="📥 Descargar reporte (PDF)",
    data=pdf_buf,
    file_name=f"reporte_inventario_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf"
)

st.markdown(f"**MAE:** {mae:.2f} &nbsp;&nbsp; **RMSE:** {rmse:.2f}")
st.success("✅ Reporte listo para descargar.")