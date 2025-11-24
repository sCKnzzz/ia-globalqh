import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io
import plotly

# DEFINIR FUNCIONES GLOBALES
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

# FUNCIÓN PARA PREPARAR DATOS (SIMPLIFICADA)
def preparar_datos(df):
    df_procesado = df.copy()
    
    # Mapear nombres de columnas
    mapeo_columnas = {
        'NIVEL DE AFORO (m)': 'NIVEL_AFORO',
        'CAUDAL (m3/s)': 'CAUDAL', 
        'AREA (m2)': 'AREA',
        'ANCHO RIO (m)': 'ANCHO_RIO',
        'PERIMETRO (m)': 'PERIMETRO',
        'VELOCIDAD (m/s)': 'VELOCIDAD',
        'FECHA AFORO': 'FECHA'
    }
    
    for col_original, col_nuevo in mapeo_columnas.items():
        if col_original in df_procesado.columns:
            df_procesado[col_nuevo] = df_procesado[col_original]
    
    # CALCULAR PERÍMETRO SI ES NECESARIO
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any() or (df_procesado['PERIMETRO'] <= 0).any():
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['PERIMETRO'] = df_procesado['ANCHO_RIO'] + 2 * df_procesado['TIRANTE_MEDIO']
        st.info("📏 Perímetro calculado automáticamente")
    
    # Calcular variables hidráulicas
    df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    
    # Año
    if 'FECHA' in df_procesado.columns:
        try:
            df_procesado['FECHA'] = pd.to_datetime(df_procesado['FECHA'], errors='coerce')
            df_procesado['YEAR'] = df_procesado['FECHA'].dt.year.fillna(2024).astype(int)
        except:
            df_procesado['YEAR'] = 2024
    else:
        df_procesado['YEAR'] = 2024
    
    return df_procesado

# FUNCIÓN PARA CLASIFICAR GRUPOS (SIMPLIFICADA)
def clasificar_grupos(df):
    """Clasificar datos en grupos según características hidráulicas"""
    df_clasificado = df.copy()
    
    # Clasificación más simple y robusta
    condiciones = [
        (df_clasificado['RADIO_HIDRAULICO'] > 0.6),  # GRUPO_ALTO_RH
        (df_clasificado['YEAR'] >= 2023),            # GRUPO_RECIENTE (más flexible)
    ]
    
    grupos = ['GRUPO_ALTO_RH', 'GRUPO_RECIENTE']
    df_clasificado['GRUPO_PREDICHO'] = np.select(condiciones, grupos, default='GRUPO_ESTANDAR')
    
    return df_clasificado

# FUNCIÓN MEJORADA PARA AJUSTAR CURVAS
def ajustar_curva_grupo(datos_grupo, nombre_grupo):
    try:
        H = datos_grupo['NIVEL_AFORO'].values
        Q = datos_grupo['CAUDAL'].values
        
        st.write(f"🔍 Analizando {nombre_grupo}: {len(H)} puntos")
        st.write(f"   - Niveles: {H}")
        st.write(f"   - Caudales: {Q}")
        
        if len(H) < 3:
            st.warning(f"   ⚠️ {nombre_grupo}: Solo {len(H)} puntos (mínimo 3)")
            return None
        
        # Ordenar datos
        sort_idx = np.argsort(H)
        H_sorted = H[sort_idx]
        Q_sorted = Q[sort_idx]
        
        modelos = [
            ('Polinómico G2', func_poly2),
            ('Polinómico G3', func_poly3),
            ('Potencial', func_pot)
        ]
        
        mejor_r2 = -np.inf
        mejor_modelo = None
        
        for nombre, funcion in modelos:
            try:
                st.write(f"   🔧 Probando modelo: {nombre}")
                
                if nombre == 'Potencial':
                    # Para función potencial, asegurar valores positivos
                    H_pos = np.maximum(H_sorted, 0.01)
                    Q_pos = np.maximum(Q_sorted, 0.01)
                    params, _ = curve_fit(funcion, H_pos, Q_pos, p0=[1.0, 2.0], maxfev=5000)
                    Q_pred = funcion(H_pos, *params)
                else:
                    params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
                    Q_pred = funcion(H_sorted, *params)
                
                # Calcular R²
                ss_res = np.sum((Q_sorted - Q_pred)**2)
                ss_tot = np.sum((Q_sorted - np.mean(Q_sorted))**2)
                
                if ss_tot > 0:
                    r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = 0
                
                st.write(f"     - R² = {r2:.3f}")
                
                # Aceptar modelos con R² razonable (bajamos el umbral a 0.5)
                if r2 > mejor_r2 and r2 > 0.5:
                    mejor_r2 = r2
                    mejor_modelo = {
                        'nombre': nombre,
                        'funcion': funcion,
                        'parametros': params,
                        'r2': round(r2, 3),
                        'n_puntos': len(H_sorted),
                        'rango_niveles': (min(H_sorted), max(H_sorted)),
                        'rango_caudales': (min(Q_sorted), max(Q_sorted)),
                        'grupo': nombre_grupo
                    }
                    
            except Exception as e:
                st.write(f"     ❌ Error en {nombre}: {str(e)}")
                continue
        
        if mejor_modelo:
            st.success(f"   ✅ {nombre_grupo}: {mejor_modelo['nombre']} (R²={mejor_modelo['r2']:.3f})")
            return mejor_modelo
        else:
            st.warning(f"   ❌ {nombre_grupo}: Ningún modelo tuvo R² > 0.5")
            return None
            
    except Exception as e:
        st.error(f"   💥 Error en {nombre_grupo}: {str(e)}")
        return None

# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
def procesar_datos(df):
    """Función principal para procesar datos y generar curvas"""
    
    st.write("📊 **PASO 1: Preparando datos...**")
    df_procesado = preparar_datos(df)
    
    st.write("📊 **PASO 2: Clasificando grupos...**")
    df_clasificado = clasificar_grupos(df_procesado)
    
    # Mostrar distribución de grupos
    distribucion = df_clasificado['GRUPO_PREDICHO'].value_counts()
    st.write("📈 **Distribución de grupos:**")
    for grupo, count in distribucion.items():
        st.write(f"   - {grupo}: {count} puntos")
    
    st.write("📊 **PASO 3: Ajustando curvas...**")
    resultados = {}
    
    # Procesar TODOS los grupos (incluyendo GRUPO_ESTANDAR)
    for grupo in df_clasificado['GRUPO_PREDICHO'].unique():
        grupo_data = df_clasificado[df_clasificado['GRUPO_PREDICHO'] == grupo]
        
        if len(grupo_data) >= 3:
            curva = ajustar_curva_grupo(grupo_data, grupo)
            if curva:
                resultados[grupo] = curva
        else:
            st.warning(f"⚠️ {grupo}: Solo {len(grupo_data)} puntos (se necesitan al menos 3)")
    
    return resultados, df_clasificado

# FUNCIÓN PARA GRÁFICO
def crear_grafico_curvas(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    
    # Graficar puntos por grupo
    for grupo in df['GRUPO_PREDICHO'].unique():
        color = colores.get(grupo, 'orange')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, s=60, label=grupo, alpha=0.7)
    
    # Graficar curvas ajustadas
    for grupo, curva in curvas.items():
        color = colores.get(grupo, 'orange')
        H_range = np.linspace(curva['rango_niveles'][0], curva['rango_niveles'][1], 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        
        ax.plot(H_range, Q_curve, color=color, linewidth=2, 
               label=f'{grupo} (R²={curva["r2"]:.3f})')
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Curvas Altura-Caudal", page_icon="🌊", layout="wide")
st.title("🌊 Generador de Curvas Altura-Caudal")
st.markdown("Sistema de clasificación por grupos y ajuste de curvas")

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Generador de Curvas Altura-Caudal")
    st.info("Sube tus datos de aforos o ingrésalos manualmente para generar curvas altura-caudal")
    
    st.subheader("📋 Datos necesarios:")
    st.markdown("""
    - Nivel de aforo (m)
    - Caudal (m³/s) 
    - Área (m²)
    - Ancho del río (m)
    - Velocidad (m/s)
    - Fecha (opcional)
    """)

elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")
    
    archivo_subido = st.file_uploader("Selecciona archivo CSV", type=['csv'])
    
    if archivo_subido is not None:
        try:
            df = pd.read_csv(archivo_subido)
            st.success(f"✅ {len(df)} aforos cargados exitosamente")
            
            # Mostrar vista previa
            st.subheader("📋 Vista previa de datos")
            st.dataframe(df.head())
            
            # Verificar columnas básicas
            columnas_necesarias = ['CAUDAL (m3/s)', 'VELOCIDAD (m/s)', 'AREA (m2)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)']
            columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
            
            if not columnas_faltantes:
                st.success("✅ Todas las columnas necesarias están presentes")
                
                if st.button("🚀 GENERAR CURVAS ALTURA-CAUDAL", type="primary", use_container_width=True):
                    
                    # PROCESAR DATOS
                    curvas, datos_procesados = procesar_datos(df)
                    
                    if curvas:
                        st.success(f"🎉 ¡ÉXITO! Se generaron {len(curvas)} curvas")
                        
                        # Mostrar gráfico
                        st.subheader("📈 Curvas Altura-Caudal Generadas")
                        fig = crear_grafico_curvas(datos_procesados, curvas, "Curvas Altura-Caudal por Grupo")
                        st.pyplot(fig)
                        
                        # Mostrar ecuaciones
                        st.subheader("📐 Ecuaciones por Grupo")
                        for grupo, curva in curvas.items():
                            with st.expander(f"{grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
                                st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                                st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.2f} - {curva['rango_caudales'][1]:.2f} m³/s")
                                
                                if curva['nombre'] == 'Polinómico G2':
                                    a, b, c = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                elif curva['nombre'] == 'Polinómico G3':
                                    a, b, c, d = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                                elif curva['nombre'] == 'Potencial':
                                    a, b = curva['parametros']
                                    st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                        
                        # Mostrar datos procesados
                        st.subheader("📊 Datos Procesados")
                        st.dataframe(datos_procesados[['NIVEL_AFORO', 'CAUDAL', 'AREA', 'ANCHO_RIO', 'RADIO_HIDRAULICO', 'GRUPO_PREDICHO']])
                        
                    else:
                        st.error("❌ No se pudieron generar curvas. Revisa los mensajes de arriba para identificar el problema.")
                        
            else:
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    st.info("💡 Ingresa los datos de aforo manualmente")
    
    num_aforos = st.number_input("Número de aforos:", min_value=1, max_value=20, value=3)
    datos_manual = []
    
    for i in range(num_aforos):
        with st.expander(f"Aforo {i+1}", expanded=True if i == 0 else False):
            col1, col2 = st.columns(2)
            with col1:
                nivel = st.number_input("Nivel (m)", min_value=0.1, value=1.0 + i*0.5, key=f"n{i}")
                caudal = st.number_input("Caudal (m³/s)", min_value=0.1, value=2.0 + i*1.0, key=f"q{i}")
                area = st.number_input("Área (m²)", min_value=0.1, value=3.0 + i*2.0, key=f"a{i}")
            with col2:
                ancho = st.number_input("Ancho Río (m)", min_value=0.1, value=8.0, key=f"w{i}")
                velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, value=0.7, key=f"v{i}")
            
            datos_manual.append({
                'FECHA AFORO': '2024-01-01',
                'NIVEL DE AFORO (m)': nivel,
                'CAUDAL (m3/s)': caudal,
                'AREA (m2)': area,
                'ANCHO RIO (m)': ancho,
                'VELOCIDAD (m/s)': velocidad
            })
    
    if st.button("🚀 GENERAR CURVAS CON DATOS MANUALES", type="primary", use_container_width=True) and datos_manual:
        with st.spinner("Procesando..."):
            df_manual = pd.DataFrame(datos_manual)
            curvas, datos_procesados = procesar_datos(df_manual)
            
            if curvas:
                st.success(f"🎉 ¡ÉXITO! Se generaron {len(curvas)} curvas")
                
                # Mostrar gráfico
                st.subheader("📈 Curvas Generadas")
                fig = crear_grafico_curvas(datos_procesados, curvas, "Curvas Altura-Caudal")
                st.pyplot(fig)
                
                # Mostrar ecuaciones
                st.subheader("📐 Ecuaciones")
                for grupo, curva in curvas.items():
                    with st.expander(f"{grupo} - {curva['nombre']} - R² = {curva['r2']:.3f}"):
                        if curva['nombre'] == 'Polinómico G2':
                            a, b, c = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                        elif curva['nombre'] == 'Polinómico G3':
                            a, b, c, d = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                        elif curva['nombre'] == 'Potencial':
                            a, b = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
            else:
                st.error("❌ No se pudieron generar curvas con los datos ingresados")

st.markdown("---")
st.markdown("**🌊 Sistema de Generación de Curvas Altura-Caudal**")